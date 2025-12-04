"""
DINOv3-based segmentation model definition.

This module exposes a segmentation model that:
- Uses a pretrained DINOv3 ViT backbone from Hugging Face.
- Freezes the backbone parameters.
- Adds a lightweight convolutional decoder that upsamples patch tokens
  back to pixel space to produce a 1-channel segmentation mask.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from huggingface_hub import login


class DINOv3SegmentationModel(nn.Module):
    """
    Semantic segmentation model with a frozen DINOv3 backbone and
    a small CNN decoder on top.

    The model:
    - Accepts RGB images of shape (B, 3, H, W).
    - Produces logits of shape (B, 1, H, W) corresponding to the mask.
    """

    def __init__(
        self,
        dino_model_id: str,
        hf_token: Optional[str] = None,
        freeze_backbone: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        dino_model_id : str
            Hugging Face model ID for the DINOv3 backbone
            (e.g., "facebook/dinov3-vits16-pretrain-lvd1689m").
        hf_token : Optional[str]
            Optional Hugging Face access token. If provided, login() is called.
            For public repos you should NOT hardcode this token in the code.
        freeze_backbone : bool
            If True, backbone parameters are frozen and only the decoder is trained.
        """
        super().__init__()

        if hf_token is not None:
            # Log in to the Hugging Face Hub programmatically.
            # On Colab or local machines, you can alternatively run
            # `huggingface-cli login` instead of passing a token here.
            login(token=hf_token)

        # Load the pretrained DINOv3 model as a feature extractor.
        self.backbone = AutoModel.from_pretrained(dino_model_id, token=hf_token)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # DINOv3 ViT exposes:
        # - `config.hidden_size` for embedding dimension
        # - optionally `config.num_register_tokens` for extra tokens
        self.num_registers = getattr(self.backbone.config, "num_register_tokens", 0)
        self.embed_dim = self.backbone.config.hidden_size

        # Simple convolutional decoder that maps patch tokens to a 1-channel mask.
        self.decoder = nn.Sequential(
            nn.Conv2d(self.embed_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of images, shape (B, 3, H, W).

        Returns
        -------
        torch.Tensor
            Logits for the segmentation mask, shape (B, 1, H, W).
        """
        b, c, h, w = x.shape

        # Extract token embeddings from the DINOv3 backbone.
        with torch.no_grad():
            outputs = self.backbone(pixel_values=x)
            # Skip [CLS] token and any register tokens, keep only patch tokens.
            patch_tokens = outputs.last_hidden_state[
                :, (1 + self.num_registers) :, :
            ]  # (B, N_patches, D)

        # Convert patch sequence to a 2D feature map.
        n = patch_tokens.shape[1]
        grid_size = int(n**0.5)
        features = (
            patch_tokens.permute(0, 2, 1)
            .reshape(b, self.embed_dim, grid_size, grid_size)
        )

        # Decode to low-resolution logits and upsample to the original image size.
        logits_lowres = self.decoder(features)
        logits = F.interpolate(
            logits_lowres, size=(h, w), mode="bilinear", align_corners=False
        )

        return logits
