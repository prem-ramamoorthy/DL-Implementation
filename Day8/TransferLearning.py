import torch
import torch.nn as nn
from torchvision import models

# ──────────────────────────────────────────────────────────────
# TYPE 1: Feature Extraction
# Freeze entire backbone → only train the new classifier head
# ──────────────────────────────────────────────────────────────
def feature_extraction(num_classes=10):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():       # freeze all
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)   # new head (unfrozen by default)
    return model


# ──────────────────────────────────────────────────────────────
# TYPE 2: Partial Fine-Tuning
# Freeze early layers → unfreeze layer4 + head
# ──────────────────────────────────────────────────────────────
def partial_finetune(num_classes=10, unfreeze_from="layer4"):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():       # freeze all first
        param.requires_grad = False

    unfreeze = False
    for name, module in model.named_children():
        if name == unfreeze_from:
            unfreeze = True
        if unfreeze:
            for param in module.parameters():
                param.requires_grad = True  # unfreeze from layer4 onward

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ──────────────────────────────────────────────────────────────
# TYPE 3: Full Fine-Tuning (differential learning rates)
# All layers train — backbone gets a lower LR than the head
# ──────────────────────────────────────────────────────────────
def full_finetune(num_classes=10):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_optimizer(model, backbone_lr=1e-4, head_lr=1e-3):
    head_ids     = {id(p) for p in model.fc.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return torch.optim.Adam([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": model.fc.parameters(), "lr": head_lr},
    ])


# ──────────────────────────────────────────────────────────────
# TYPE 4: Multi-Task Transfer
# One backbone → two task-specific heads
# ──────────────────────────────────────────────────────────────
class MultiTaskResNet(nn.Module):
    def __init__(self, num_fine=10, num_coarse=2):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone  = nn.Sequential(*list(base.children())[:-1])  # strip fc
        self.head_fine   = nn.Linear(512, num_fine)
        self.head_coarse = nn.Linear(512, num_coarse)

    def forward(self, x):
        feats = self.backbone(x).flatten(1)          # (B, 512)
        return self.head_fine(feats), self.head_coarse(feats)


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)

    m1 = feature_extraction();         print("Type 1 out:", m1(x).shape)
    m2 = partial_finetune();           print("Type 2 out:", m2(x).shape)
    m3 = full_finetune();              print("Type 3 out:", m3(x).shape)
    m4 = MultiTaskResNet()
    f, c = m4(x);                      print("Type 4 out:", f.shape, c.shape)