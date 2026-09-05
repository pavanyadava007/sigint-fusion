import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, c: int, k: int = 7):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(c, c, k, padding=k // 2), nn.BatchNorm1d(c), nn.ReLU(),
                                 nn.Conv1d(c, c, k, padding=k // 2), nn.BatchNorm1d(c))

    def forward(self, x):
        return torch.relu(x + self.net(x))


class ResNet1D(nn.Module):
    """Length-agnostic I/Q classifier: conv backbone -> adaptive pool -> `emb`-d embedding -> linear head.

    The head is swappable (`reset_head`) and the backbone freezable (`freeze_backbone`) for transfer learning;
    `embed` exposes the embedding for prototypical few-shot classification.
    """

    def __init__(self, n_classes: int, width: int = 64, depth: int = 4, emb: int = 128):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv1d(2, width, 7, padding=3), nn.BatchNorm1d(width), nn.ReLU()]
        for _ in range(depth):
            layers += [Block(width), nn.MaxPool1d(2)]
        self.backbone = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(width, emb), nn.ReLU())
        self.head = nn.Linear(emb, n_classes)

    def embed(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.head(self.embed(x))

    def reset_head(self, n_classes: int):
        self.head = nn.Linear(self.head.in_features, n_classes)

    def freeze_backbone(self, freeze: bool = True):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
