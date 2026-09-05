"""STFT spectrogram -> pretrained ViT (timm) transfer learning.

The frontend turns I/Q [B,2,L] into a 3-channel 224x224 dB spectrogram image (min-max normalised per sample) so an
ImageNet-pretrained vit_tiny_patch16_224 can be fine-tuned. Not ONNX-exportable via the legacy exporter (torch.stft);
kept as the comparison model for the ResNet-1D that is served.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class STFTFrontend(nn.Module):
    def __init__(self, n_fft: int = 64, hop: int = 8, out: int = 224):
        super().__init__()
        self.n_fft, self.hop, self.out = n_fft, hop, out
        self.register_buffer("win", torch.hann_window(n_fft))

    def forward(self, x):  # x:[B,2,L] -> [B,3,out,out]
        z = torch.complex(x[:, 0], x[:, 1])
        S = torch.stft(z, self.n_fft, self.hop, window=self.win, return_complex=True, onesided=False)
        S = torch.fft.fftshift(S, dim=1)
        img = 20 * torch.log10(S.abs() + 1e-6)
        lo, hi = img.amin(dim=(1, 2), keepdim=True), img.amax(dim=(1, 2), keepdim=True)
        img = (img - lo) / (hi - lo + 1e-6)
        img = nn.functional.interpolate(img[:, None], size=(self.out, self.out), mode="bilinear", align_corners=False)
        return img.repeat(1, 3, 1, 1)


class ViTSpec(nn.Module):
    def __init__(self, n_classes: int, name: str = "vit_tiny_patch16_224", pretrained: bool = True):
        super().__init__()
        import timm

        self.front = STFTFrontend()
        self.vit = timm.create_model(name, pretrained=pretrained, num_classes=0)
        self.head = nn.Linear(self.vit.num_features, n_classes)

    def embed(self, x):
        return self.vit(self.front(x))

    def forward(self, x):
        return self.head(self.embed(x))

    def reset_head(self, n: int):
        self.head = nn.Linear(self.vit.num_features, n)

    def freeze_backbone(self, freeze: bool = True):
        for p in self.vit.parameters():
            p.requires_grad = not freeze
