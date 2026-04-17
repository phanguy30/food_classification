"""
helpers.py — shared utilities for all fine-tuning experiments
"""

import os
import copy
import time
import loralib as lora
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
import torchvision.models as models
from torchvision.models import ResNet18_Weights, EfficientNet_V2_S_Weights


# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────

def get_food101_loaders(transform, batch_size: int = 256, num_workers: int = 4):
    """Return (train_loader, val_loader) for Food-101."""
    train_dataset = Food101(root="data", split="train", download=True, transform=transform)
    val_dataset   = Food101(root="data", split="test",  download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


# ──────────────────────────────────────────────
# Train / Validate
# ──────────────────────────────────────────────

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def validate_with_time(model, dataloader, criterion, device):
    """Like validate(), but also returns per-image latency and throughput."""
    model.eval()
    total_loss, correct, total, total_time = 0.0, 0, 0, 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            outputs = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.time() - t0

            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds   = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += images.size(0)

    return (
        total_loss / total,
        correct / total,
        total_time / total,          # seconds per image
        total / total_time,          # images per second
    )


# ──────────────────────────────────────────────
# Training loop with checkpointing
# ──────────────────────────────────────────────

def run_training(
    model,
    model_name: str,
    train_dataloader,
    val_dataloader,
    epochs: int = 8,
    lr: float = 1e-3,
    optimizer=None,                  # pass a custom optimizer, or None to use Adam
    checkpoint_dir: str = "model_weights",
):
    """
    Full training loop with resume-from-checkpoint support.

    Returns: model, train_losses, val_losses, train_accs, val_accs
    """
    device = get_device()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable, lr=lr)

    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path       = os.path.join(checkpoint_dir, f"checkpoint_{model_name}.pt")
    final_ckpt_path = os.path.join(checkpoint_dir, f"final_checkpoint_{model_name}.pt")

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0.0
    best_state   = None
    curr_epoch   = 0

    # ── If final checkpoint exists, just load metrics and return ──
    if os.path.exists(final_ckpt_path):
        print(f"[{model_name}] Final checkpoint found — loading metrics for plotting.")
        ckpt = torch.load(final_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        return (
            model,
            ckpt.get("train_losses", []),
            ckpt.get("val_losses",   []),
            ckpt.get("train_accs",   []),
            ckpt.get("val_accs",     []),
        )

    # ── Resume from mid-run checkpoint ──
    if os.path.exists(ckpt_path):
        print(f"[{model_name}] Resuming from checkpoint …")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        best_val_acc = ckpt["best_val_acc"]
        train_losses = ckpt.get("train_losses", [])
        val_losses   = ckpt.get("val_losses",   [])
        train_accs   = ckpt.get("train_accs",   [])
        val_accs     = ckpt.get("val_accs",     [])
        curr_epoch   = ckpt["epoch"] + 1
        print(f"  → epoch {curr_epoch}, best val acc {best_val_acc:.4f}")
    else:
        print(f"[{model_name}] Starting from scratch.")

    print(f"  Trainable params: {count_trainable_params(model):,}")

    # ── Main loop ──
    while curr_epoch < epochs:
        train_loss, train_acc = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        val_loss,   val_acc   = validate(model, val_dataloader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"  Epoch {curr_epoch+1:>2}/{epochs} — "
            f"train loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": curr_epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "train_losses": train_losses,
                    "val_losses":   val_losses,
                    "train_accs":   train_accs,
                    "val_accs":     val_accs,
                },
                ckpt_path,
            )

        curr_epoch += 1

    print(f"[{model_name}] Training complete. Best val acc: {best_val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "epoch": curr_epoch - 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "train_losses": train_losses,
            "val_losses":   val_losses,
            "train_accs":   train_accs,
            "val_accs":     val_accs,
        },
        final_ckpt_path,
    )

    return model, train_losses, val_losses, train_accs, val_accs


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def plot_results(train_losses, val_losses, train_accs, val_accs, title_prefix: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label="Train Loss")
    axes[0].plot(val_losses,   label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title_prefix}: Loss")
    axes[0].legend()

    axes[1].plot(train_accs, label="Train Acc")
    axes[1].plot(val_accs,   label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title_prefix}: Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# Misc
# ──────────────────────────────────────────────

def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def replace_module_by_name(model, module_name: str, new_module: nn.Module):
    """Replace a submodule addressed by dotted name, e.g. 'layer1.0.conv1'."""
    parts  = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def to_int(x):
    """Unwrap single-element tuples returned by Conv2d attribute inspection."""
    return x[0] if isinstance(x, tuple) else x


# ──────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────

def build_resnet18_lora(num_classes: int = 101, r: int = 8, alpha: int = 16):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    target_layers = [
        "layer1.0.conv1", "layer1.0.conv2", "layer1.1.conv1", "layer1.1.conv2",
        "layer2.0.conv1", "layer2.0.conv2", "layer2.1.conv1", "layer2.1.conv2",
        "layer3.0.conv1", "layer3.0.conv2", "layer3.1.conv1", "layer3.1.conv2",
        "layer4.0.conv1", "layer4.0.conv2", "layer4.1.conv1", "layer4.1.conv2",
    ]

    for name, module in list(model.named_modules()):
        if name in target_layers and isinstance(module, nn.Conv2d):
            new_layer = lora.Conv2d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=to_int(module.kernel_size),
                stride=to_int(module.stride),
                padding=to_int(module.padding),
                dilation=to_int(module.dilation),
                groups=module.groups,
                bias=(module.bias is not None),
                r=r, lora_alpha=alpha,
            )
            new_layer.conv.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new_layer.conv.bias.data.copy_(module.bias.data)
            replace_module_by_name(model, name, new_layer)

    lora.mark_only_lora_as_trainable(model)
    for p in model.fc.parameters():
        p.requires_grad = True

    return model


def build_efficientnet_v2_s_lora(num_classes=101, r=8, alpha=16):
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Conv2d) and name.startswith("features"):
            if module.groups != 1:
                continue

            new_layer = lora.Conv2d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=to_int(module.kernel_size),
                stride=to_int(module.stride),
                padding=to_int(module.padding),
                dilation=to_int(module.dilation),
                groups=module.groups,
                bias=(module.bias is not None),
                r=r,
                lora_alpha=alpha
            )

            new_layer.conv.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new_layer.conv.bias.data.copy_(module.bias.data)

            replace_module_by_name(model, name, new_layer)

    lora.mark_only_lora_as_trainable(model)

    for p in model.classifier[1].parameters():
        p.requires_grad = True

    return model


    

class ResNetWithAdapters(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.adapters = nn.ModuleDict({
            "layer1": CNNAdapter(64),
            "layer2": CNNAdapter(128),
            "layer3": CNNAdapter(256),
            "layer4": CNNAdapter(512),
        })

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.adapters["layer1"](x)

        x = self.model.layer2(x)
        x = self.adapters["layer2"](x)

        x = self.model.layer3(x)
        x = self.adapters["layer3"](x)

        x = self.model.layer4(x)
        x = self.adapters["layer4"](x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)

        return x
    


class CNNAdapter(nn.Module):
    def __init__(self, channels, reduction_factor=8):
        super().__init__()
        bottleneck = channels // reduction_factor
        self.adapter = nn.Sequential(
            nn.Conv2d(channels, bottleneck, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, channels, kernel_size=1)
        )

    def forward(self, x):
        return x + self.adapter(x) 
    


class EfficientNetWithAdapters(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        # pick certain stages to add adapters to. 
        self.adapter_indices = [3, 5, 6]  

        self.adapters = nn.ModuleDict({
            str(i): CNNAdapter(self._get_channels(i))
            for i in self.adapter_indices
        })

    def _get_channels(self, idx):
        # get output channels of that stage
        return self.model.features[idx][-1].out_channels

    def forward(self, x):
        for i, layer in enumerate(self.model.features):
            x = layer(x)
            if str(i) in self.adapters:
                x = self.adapters[str(i)](x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        return x

base_model = models.efficientnet_v2_s(
    weights=models.EfficientNet_V2_S_Weights.DEFAULT
)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        # x: (B, C, H, W)

        avg = torch.mean(x, dim=1, keepdim=True)      # (B, 1, H, W)
        max, _ = torch.max(x, dim=1, keepdim=True)    # (B, 1, H, W)

        x_cat = torch.cat([avg, max], dim=1)          # (B, 2, H, W)
        mask = torch.sigmoid(self.conv(x_cat))        # (B, 1, H, W)

        return x * mask

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        avg = torch.mean(x, dim=(2, 3))               # (B, C)
        max, _ = torch.max(x.view(b, c, -1), dim=2)   # (B, C)

        attn = self.mlp(avg) + self.mlp(max)
        attn = torch.sigmoid(attn).view(b, c, 1, 1)

        return x * attn

class CBAMAdapter(nn.Module):
    def __init__(self, channels, reduction_factor=8):
        super().__init__()
        
        bottleneck = channels // reduction_factor
        self.conv_adapter = nn.Sequential(
            nn.Conv2d(channels, bottleneck, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, channels, kernel_size=1)
        )
        # make sure conv adapter starts as identity function
        nn.init.zeros_(self.conv_adapter[-1].weight)
        
        self.spatial_attn = SpatialAttention()
        self.channel_attn = ChannelAttention(channels)

    def forward(self, x):
        x = x + self.conv_adapter(x)
        x = x + self.channel_attn(x)
        x = x + self.spatial_attn(x)
        
        return x

class ResNetWithCBAMAdapters(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.adapters = nn.ModuleDict({
            "layer1": CBAMAdapter(64),
            "layer2": CBAMAdapter(128),
            "layer3": CBAMAdapter(256),
            "layer4": CBAMAdapter(512),
        })

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.adapters["layer1"](x)

        x = self.model.layer2(x)
        x = self.adapters["layer2"](x)

        x = self.model.layer3(x)
        x = self.adapters["layer3"](x)

        x = self.model.layer4(x)
        x = self.adapters["layer4"](x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)

        return x