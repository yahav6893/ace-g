# Copyright © Niantic Spatial, Inc. 2025. All rights reserved.
"""Image encoders for camera localization."""

from __future__ import annotations

# Default SelaVPR repo root (override via env var SELAVPR_REPO)

import logging
import pathlib
import os
from typing import TypedDict

import torch
import torch.nn.functional as F
from torch import nn

from ace_g import utils

_logger = logging.getLogger(__name__)


class EncoderConfig(TypedDict):
    """Configuration for an encoder network."""

    obj_type: str
    """Type of the encoder; has to be resolvable by ace_g.utils.str_to_object (fully-qualified python name will work
    from anywhere "ace_g.encoders.DINOv2Encoder")"""
    kwargs: dict
    """Keyword arguments passed to the encoder constructor."""
    path: pathlib.Path | None
    """Path to the encoder weights; None if no weights are available."""


class Encoder(nn.Module):
    """Encoder base class."""

    dim_out: int
    """Number of output channels of the encoder."""
    subsample_factor: int
    """Spatial subsampling factor of the encoder."""
    supports_rgb: bool
    """Whether the encoder supports RGB images."""

class ListMultiEncoder(Encoder):
    """Runs a list of encoders and concatenates their patch features along channel dim.

    This is a simpler alternative to MultiDinoEncoder(main + fusion):
      - you provide one list: encoders=[...]
      - output: (..., K*C, H', W') where K=len(encoders)

    Constraints:
      - all sub-encoders must have identical subsample_factor and dim_out
      - all must produce identical spatial output size for a given input
    """

    supports_rgb = True

    def __init__(self, encoders: list[EncoderConfig]) -> None:
        super().__init__()
        assert encoders and len(encoders) >= 1, "ListMultiEncoder requires a non-empty encoders list."

        self.encoders = nn.ModuleList([create_encoder(c) for c in encoders])
        self.num_encoders = len(self.encoders)

        sf = self.encoders[0].subsample_factor
        dim = self.encoders[0].dim_out
        for i, e in enumerate(self.encoders):
            assert e.subsample_factor == sf, f"subsample_factor mismatch at enc[{i}]"
            assert e.dim_out == dim, f"dim_out mismatch at enc[{i}]"

        self.subsample_factor = sf
        self.dim_out_single = dim
        self.dim_out = self.num_encoders * self.dim_out_single
        self.supports_rgb = all(getattr(e, "supports_rgb", False) for e in self.encoders)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = []
        h0 = w0 = None
        for i, e in enumerate(self.encoders):
            f = e(images)
            if h0 is None:
                h0, w0 = f.shape[-2], f.shape[-1]
            elif f.shape[-2] != h0 or f.shape[-1] != w0:
                raise ValueError(
                    f"ListMultiEncoder spatial mismatch: enc[0]=({h0},{w0}) enc[{i}]=({f.shape[-2]},{f.shape[-1]})"
                )
            feats.append(f)
        return torch.cat(feats, dim=-3)


class MultiDinoEncoder(Encoder):
    """
    Encoder that runs multiple DINOv2-family encoders and concatenates their patch features.

    This is intended for *fully buffered* multi-encoder training where the buffer stores concatenated feature vectors,
    and all learnable fusion happens in the head.

    Constraints:
      - All sub-encoders must have identical `subsample_factor` (patch size) and identical output channel dim.
      - All sub-encoders must produce identical spatial output sizes for a given input (enforced at runtime).

    The output shape is (..., N*dim_out_single, H', W') where N is the number of encoders.
    """

    supports_rgb = True  # Effective value is computed in __init__.

    def __init__(self, main: EncoderConfig, fusion: list[EncoderConfig]) -> None:
        """
        Args:
            main: EncoderConfig for the 'main' encoder. This encoder's features will typically be used as the query
                input in the fusion head (but this is decided by the head).
            fusion: List of EncoderConfig for the remaining encoders.
        """
        super(MultiDinoEncoder, self).__init__()

        if fusion is None:
            fusion = []

        # Instantiate sub-encoders via the same factory, so existing configs keep working.
        self.main_encoder = create_encoder(main)
        self.fusion_encoders = nn.ModuleList([create_encoder(c) for c in fusion])

        self.encoders = nn.ModuleList([self.main_encoder, *list(self.fusion_encoders)])
        self.num_encoders = len(self.encoders)

        assert self.num_encoders >= 1, "MultiDinoEncoder requires at least one encoder (main)."

        # Validate shared contract.
        sf = self.main_encoder.subsample_factor
        dim = self.main_encoder.dim_out
        for i, e in enumerate(self.encoders):
            assert (
                e.subsample_factor == sf
            ), f"All encoders must share subsample_factor. main={sf}, enc[{i}]={e.subsample_factor}"
            assert e.dim_out == dim, f"All encoders must share dim_out. main={dim}, enc[{i}]={e.dim_out}"

        self.subsample_factor = sf
        self.dim_out_single = dim
        self.dim_out = self.num_encoders * self.dim_out_single

        # supports_rgb is true only if all sub-encoders support RGB.
        self.supports_rgb = all(getattr(e, "supports_rgb", False) for e in self.encoders)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute concatenated patch features.

        Args:
            images: Input images. Shape (..., 3 or 1, H, W).

        Returns:
            Patch features. Shape (..., N*dim_out_single, H', W').
        """
        feats = []
        h0 = w0 = None
        for i, e in enumerate(self.encoders):
            f = e(images)
            if h0 is None:
                h0, w0 = f.shape[-2], f.shape[-1]
            else:
                if f.shape[-2] != h0 or f.shape[-1] != w0:
                    raise ValueError(
                        "MultiDinoEncoder spatial output mismatch across encoders. "
                        f"enc[0]=({h0},{w0}) enc[{i}]=({f.shape[-2]},{f.shape[-1]})"
                    )
            feats.append(f)

        # Concatenate channels: (..., N*C, H', W')
        return torch.cat(feats, dim=-3)


class DINOv2Encoder(Encoder):
    """DINOv2 encoder, used to extract features from the input images.

    The number of output channels is not modified and depends on the chosen model.

    See https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md#model-details.

    Support model names:
        dinov2_vits14
        dinov2_vits14_reg
        dinov2_vitb14
        dinov2_vitb14_reg
        dinov2_vitl14
        dinov2_vitl14_reg
        dinov2_vitg14
        dinov2_vitg14_reg
    """

    supports_rgb = True

    def __init__(self, model_name: str = "dinov2_vits14_reg") -> None:
        """Initialize the DINOv2 encoder.

        Args:
            model_name: Name of the model to load using torch.hub.load. See class docstring for supported models.
        """
        super(DINOv2Encoder, self).__init__()

        hub = "facebookresearch/dinov2"

        # Load the DINOv2 model.
        self.dinov2 = torch.hub.load(
            hub, model_name, verbose=True, trust_repo=True, source="github", skip_validation=True
        )
        self.subsample_factor = self.dinov2.patch_embed.patch_size[0]  # type: ignore
        self.dim_out = self.dinov2.embed_dim  # type: ignore

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Compute features for images.

        The image is cropped to a size that is a multiple of the patch size (i.e., right and bottom might be cropped).
        Spatial dimensions are reduced by the patch size (e.g., for patches of size 14, H' = H // 14).
        See subsample_factor.

        For grayscsale images the input is broadcasted to 3 channels.

        Args:
            images: Input images. Shape (..., 3 or 1, H, W).

        Returns:
            Features for image patches. Shape (..., dim_feat, H', W').
        """
        leading_dims = images.shape[:-3]
        c, h, w = images.shape[-3:]
        images = images.view(-1, c, h, w)
        if c == 1:  # for grayscale images, pass the same channel three times
            images = images.expand(-1, 3, -1, -1)
        images = images[
            ...,
            : self.subsample_factor * (h // self.subsample_factor),
            : self.subsample_factor * (w // self.subsample_factor),
        ]
        features = self.dinov2.forward_features(images)  # type: ignore
        patch_features = (
            features["x_norm_patchtokens"]
            .permute(0, 2, 1)
            .reshape(*leading_dims, -1, h // self.subsample_factor, w // self.subsample_factor)
        )
        return patch_features
class DPTv2Encoder(Encoder):
    """Depth Anything V2 (DINOv2-DPT) encoder.

    This encoder loads a Depth Anything V2 checkpoint and exposes the underlying DINOv2 backbone
    (fine-tuned for depth estimation) through the same patch-token interface as :class:`DINOv2Encoder`.

    The forward pass is intentionally identical to :class:`DINOv2Encoder` except that it calls
    ``self.dinov2_dpt.forward_features(images)``.

    Notes for training stability / gotchas:
    - **Input normalization**: Depth Anything V2 models are typically used with ImageNet normalization
      (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]). This encoder does **not** apply any
      normalization, mirroring :class:`DINOv2Encoder`. Ensure your dataset transform matches what you
      want (otherwise convergence and final accuracy may suffer).
    - **RGB vs grayscale**: the backbone expects 3 channels. If you feed grayscale, we broadcast 1→3.
    - **Spatial sizes**: images are cropped to a multiple of the patch size (usually 14).

    Supported encoder variants (matching the official DA-V2 checkpoints):
        vits, vitb, vitl

    References:
        Depth Anything V2 repo + checkpoints: https://github.com/DepthAnything/Depth-Anything-V2
    """

    supports_rgb = True

    # Official checkpoint URLs (Hugging Face "resolve" links used by the upstream README).
    _CKPT_URLS: dict[str, str] = {
        "vits": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true",
        "vitb": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true",
        "vitl": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true",
    }

    def __init__(
        self,
        model_name: str = "vitb",
        checkpoint: str | pathlib.Path | None = None,
    ) -> None:
        """Initialize the Depth Anything V2 encoder.

        Args:
            encoder: One of {"vits","vitb","vitl"}.
            checkpoint: Optional path to a local ``depth_anything_v2_*.pth`` checkpoint. If None, the
                official checkpoint is downloaded via ``torch.hub.load_state_dict_from_url`` (cached under
                TORCH_HOME / default torch hub cache).
        """
        super().__init__()

        try:
            # pip package: `depth-anything-v2`  -> import name: `depth_anything_v2`
            from depth_anything_v2.dinov2 import DINOv2 as DepthAnythingDINOv2  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Missing dependency `depth_anything_v2`. Install it (e.g. `pip install depth-anything-v2`) "
                "or vendor the Depth-Anything-V2 repo into your PYTHONPATH."
            ) from e

        if model_name not in self._CKPT_URLS:
            raise ValueError(f"Unsupported Depth Anything V2 encoder='{model_name}'. Expected one of {sorted(self._CKPT_URLS)}.")

        # Instantiate the DA-V2 DINOv2 backbone (this class provides .forward_features()).
        self.dinov2_dpt = DepthAnythingDINOv2(model_name=model_name)

        # Load checkpoint.
        if checkpoint is None:
            url = self._CKPT_URLS[model_name]
            state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=True)  # type: ignore
        else:
            state_dict = torch.load(pathlib.Path(checkpoint), map_location="cpu", weights_only=False)

        # Common wrappers: sometimes the actual dict is under "state_dict" or "model".
        if isinstance(state_dict, dict):
            if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict and isinstance(state_dict["model"], dict):
                state_dict = state_dict["model"]

        assert isinstance(state_dict, dict), "Expected a state-dict-like mapping from the checkpoint."

        # DepthAnythingV2 checkpoints are for the full model (pretrained backbone + DPT head).
        # We only need the backbone: keys are typically prefixed with 'pretrained.'.
        pretrained_sd = {k.replace("pretrained.", "", 1): v for k, v in state_dict.items() if k.startswith("pretrained.")}

        # Handle DataParallel prefixes if needed.
        if not pretrained_sd:
            pretrained_sd = {
                k.replace("module.pretrained.", "", 1): v for k, v in state_dict.items() if k.startswith("module.pretrained.")
            }

        # If the checkpoint looks like *just* a backbone checkpoint (no 'pretrained.' prefix), fall back.
        if not pretrained_sd:
            pretrained_sd = state_dict

        incompatible = self.dinov2_dpt.load_state_dict(pretrained_sd, strict=False)
        if getattr(incompatible, "missing_keys", None):
            _logger.warning(f"DPTv2Encoder: missing keys while loading backbone (showing up to 10): {incompatible.missing_keys[:10]}")
        if getattr(incompatible, "unexpected_keys", None):
            _logger.warning(
                f"DPTv2Encoder: unexpected keys while loading backbone (showing up to 10): {incompatible.unexpected_keys[:10]}"
            )

        # Infer output/stride from the backbone.
        ps = getattr(getattr(self.dinov2_dpt, "patch_embed", None), "patch_size", 14)
        if isinstance(ps, (tuple, list)):
            ps = ps[0]
        self.subsample_factor = int(ps)
        self.dim_out = int(getattr(self.dinov2_dpt, "embed_dim"))
        _logger.info(f"DPTv2Encoder: dim_out (attr):{self.dim_out}")
        _logger.info(f"DPTv2Encoder: dim_out (field):{self.dinov2_dpt.embed_dim}")


    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Compute features for images.

        Identical to :meth:`DINOv2Encoder.forward`, except it calls ``self.dinov2_dpt.forward_features``.

        Args:
            images: Input images. Shape (..., 3 or 1, H, W).

        Returns:
            Features for image patches. Shape (..., dim_feat, H', W').
        """
        leading_dims = images.shape[:-3]
        c, h, w = images.shape[-3:]
        images = images.view(-1, c, h, w)
        if c == 1:  # for grayscale images, pass the same channel three times
            images = images.expand(-1, 3, -1, -1)
        images = images[
            ...,
            : self.subsample_factor * (h // self.subsample_factor),
            : self.subsample_factor * (w // self.subsample_factor),
        ]
        features = self.dinov2_dpt.forward_features(images)  # type: ignore
        patch_features = (
            features["x_norm_patchtokens"]
            .permute(0, 2, 1)
            .reshape(*leading_dims, -1, h // self.subsample_factor, w // self.subsample_factor)
        )
        return patch_features

class SALADEncoder(Encoder):
    """SALAD (DINOv2-SALAD) encoder.

    Loads a vanilla DINOv2 backbone via torch.hub and then loads the SALAD checkpoint
    (trained for VPR) into the backbone weights only, ignoring the aggregator.

    Forward pass is intentionally identical to :class:`DINOv2Encoder`:
      - accepts (..., 1|3, H, W)
      - broadcasts 1->3
      - crops right/bottom to multiples of patch size
      - calls `.forward_features()` and reads `x_norm_patchtokens`
      - reshapes to (..., dim_out, H//patch, W//patch)

    Notes:
      - This encoder does NOT apply any normalization; keep normalization explicit in the dataset transform
        if you want ImageNet mean/std (SALAD training default).
      - Official released checkpoint is for `dinov2_vitb14`.
    """

    supports_rgb = True

    _DEFAULT_CKPT_URL = "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        checkpoint: str | pathlib.Path | None = None,
        checkpoint_url: str = _DEFAULT_CKPT_URL,
    ) -> None:
        super().__init__()

        # 1) Build vanilla DINOv2 backbone (same as DINOv2Encoder)
        hub = "facebookresearch/dinov2"
        self.dinov2 = torch.hub.load(
            hub, model_name, verbose=True, trust_repo=True, source="github", skip_validation=True
        )
        self.subsample_factor = self.dinov2.patch_embed.patch_size[0]  # type: ignore
        self.dim_out = self.dinov2.embed_dim  # type: ignore

        # 2) Load SALAD checkpoint (lightning .ckpt / state_dict wrappers)
        if checkpoint is None:
            sd = torch.hub.load_state_dict_from_url(
                checkpoint_url, map_location="cpu", progress=True
            )
        else:
            sd = torch.load(pathlib.Path(checkpoint), map_location="cpu", weights_only=False)

        if isinstance(sd, dict):
            if "state_dict" in sd and isinstance(sd["state_dict"], dict):
                sd = sd["state_dict"]
            elif "model" in sd and isinstance(sd["model"], dict):
                sd = sd["model"]

        if not isinstance(sd, dict):
            raise ValueError("SALADEncoder: expected checkpoint to yield a state_dict-like mapping.")

        # 3) Extract backbone weights from a larger checkpoint by trying common prefixes
        target_keys = set(self.dinov2.state_dict().keys())

        prefixes = [
            "backbone.model.",
            "module.backbone.model.",
            "model.backbone.model.",
            "module.model.backbone.model.",
            "backbone.",
            "module.backbone.",
            "model.backbone.",
            "module.model.backbone.",
            "dinov2.",
            "module.dinov2.",
            "encoder.",
            "module.encoder.",
        ]

        best_sd = None
        best_score = -1

        for pref in prefixes:
            cand = {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}
            if not cand:
                continue
            score = sum(1 for k in cand.keys() if k in target_keys)
            if score > best_score:
                best_score = score
                best_sd = cand

        # If no prefix matched, fall back to trying to load the whole dict with strict=False.
        load_sd = best_sd if (best_sd is not None and best_score > 0) else sd

        incompatible = self.dinov2.load_state_dict(load_sd, strict=False)
        if getattr(incompatible, "missing_keys", None):
            _logger.warning(
                f"SALADEncoder: missing keys while loading backbone (showing up to 10): "
                f"{incompatible.missing_keys[:10]}"
            )
        if getattr(incompatible, "unexpected_keys", None):
            _logger.warning(
                f"SALADEncoder: unexpected keys while loading backbone (showing up to 10): "
                f"{incompatible.unexpected_keys[:10]}"
            )

        _logger.info(
            f"SALADEncoder: loaded SALAD weights into {model_name} | "
            f"patch={self.subsample_factor} dim_out={self.dim_out} | "
            f"prefix_score={best_score}"
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        leading_dims = images.shape[:-3]
        c, h, w = images.shape[-3:]
        images = images.view(-1, c, h, w)

        if c == 1:
            images = images.expand(-1, 3, -1, -1)

        images = images[
            ...,
            : self.subsample_factor * (h // self.subsample_factor),
            : self.subsample_factor * (w // self.subsample_factor),
        ]

        features = self.dinov2.forward_features(images)  # type: ignore
        patch_features = (
            features["x_norm_patchtokens"]
            .permute(0, 2, 1)
            .reshape(*leading_dims, -1, h // self.subsample_factor, w // self.subsample_factor)
        )
        return patch_features

# Module-level default (override via env var if you want)
ACEG_DEFAULT_SELAVPR_REPO = pathlib.Path(os.environ.get('SELAVPR_REPO', '/content/drive/MyDrive/ace_cache/SelaVPR'))

class SELAEncoder(Encoder):
    """SelaVPR encoder (DINOv2 ViT-L/14 adapted for VPR via SelaVPR).

    Same as your current class, but `selavpr_repo` is optional:
      - if not provided, falls back to ACEG_DEFAULT_SELAVPR_REPO (or env var SELAVPR_REPO).
    """

    supports_rgb = True

    def __init__(
        self,
        checkpoint: str | pathlib.Path,
        selavpr_repo: str | pathlib.Path | None = None,
        registers: bool = False,
        img_size: int = 518,
    ) -> None:
        super().__init__()

        if selavpr_repo is None:
            selavpr_repo = ACEG_DEFAULT_SELAVPR_REPO

        repo = pathlib.Path(selavpr_repo)
        if not repo.exists():
            raise FileNotFoundError(f"SELAEncoder: selavpr_repo does not exist: {repo}")

        ckpt_path = pathlib.Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"SELAEncoder: checkpoint does not exist: {ckpt_path}")

        # Lazy import SelaVPR backbone code (keeps ace_g import clean).
        import sys
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        try:
            from backbone.vision_transformer import vit_large  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "SELAEncoder: failed importing SelaVPR backbone. "
                "Ensure you cloned Lu-Feng/SelaVPR and passed its repo root as selavpr_repo "
                "(or set env var SELAVPR_REPO)."
            ) from e

        num_register_tokens = 4 if registers else 0

        # Build SelaVPR's modified ViT-L/14 backbone
        self.backbone = vit_large(
            patch_size=14,
            img_size=img_size,
            init_values=1,
            block_chunks=0,
            num_register_tokens=num_register_tokens,
        )

        # Load checkpoint; SelaVPR checkpoints commonly store weights under ["model_state_dict"].
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
                sd = obj["model_state_dict"]
            elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
                sd = obj["state_dict"]
            elif "model" in obj and isinstance(obj["model"], dict):
                sd = obj["model"]
            else:
                sd = obj
        else:
            raise ValueError("SELAEncoder: unexpected checkpoint format (expected a dict-like checkpoint).")

        assert isinstance(sd, dict), "SELAEncoder: expected a state_dict-like mapping."

        prefixes = [
            "module.backbone.",
            "backbone.",
            "module.model.backbone.",
            "model.backbone.",
        ]

        backbone_sd = {}
        for pref in prefixes:
            if any(k.startswith(pref) for k in sd.keys()):
                backbone_sd = {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}
                break

        load_sd = backbone_sd if backbone_sd else sd

        if any(k.startswith("module.") for k in load_sd.keys()):
            load_sd = {k[len("module."):]: v for k, v in load_sd.items()}

        incompatible = self.backbone.load_state_dict(load_sd, strict=False)
        if getattr(incompatible, "missing_keys", None):
            _logger.warning(
                "SELAEncoder: missing keys while loading backbone (showing up to 10): "
                f"{incompatible.missing_keys[:10]}"
            )
        if getattr(incompatible, "unexpected_keys", None):
            _logger.warning(
                "SELAEncoder: unexpected keys while loading backbone (showing up to 10): "
                f"{incompatible.unexpected_keys[:10]}"
            )

        self.subsample_factor = 14
        self.dim_out = int(getattr(self.backbone, "embed_dim", 1024))
        _logger.info(
            f"SELAEncoder: loaded | registers={registers} | img_size={img_size} | "
            f"subsample_factor={self.subsample_factor} | dim_out={self.dim_out} | repo={repo}"
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        leading_dims = images.shape[:-3]
        c, h, w = images.shape[-3:]
        images = images.view(-1, c, h, w)

        if c == 1:
            images = images.expand(-1, 3, -1, -1)

        ps = self.subsample_factor
        images = images[..., : ps * (h // ps), : ps * (w // ps)]
        hp, wp = images.shape[-2] // ps, images.shape[-1] // ps

        features = self.backbone(images)  # dict with "x_norm_patchtokens"
        patch_tokens = features["x_norm_patchtokens"]  # (B, hp*wp, dim)
        patch_features = patch_tokens.permute(0, 2, 1).reshape(*leading_dims, -1, hp, wp)
        return patch_features

class FCNEncoder(Encoder):
    """FCN encoder, used to extract features from the input images.

    The number of output channels is configurable, the default used in the paper is 512.
    """

    supports_rgb = False  # Whether the encoder supports RGB images.

    def __init__(self, dim_out: int = 512) -> None:
        """Initialize the FCN encoder."""
        super().__init__()

        self.dim_out = dim_out

        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)

        self.res1_conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.res1_conv2 = nn.Conv2d(256, 256, 1, 1, 0)
        self.res1_conv3 = nn.Conv2d(256, 256, 3, 1, 1)

        self.res2_conv1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.res2_conv2 = nn.Conv2d(512, 512, 1, 1, 0)
        self.res2_conv3 = nn.Conv2d(512, self.dim_out, 3, 1, 1)

        self.res2_skip = nn.Conv2d(256, self.dim_out, 1, 1, 0)

        # con2, conv3, conv4 all downsample by a factor of 2, so the subsample factor is 8.
        self.subsample_factor = 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute features for images.

        Args:
            images: Gray-scale image. Shape (..., 1, H, W).

        Returns:
            Features for image patches. Shape (..., 512, H', W').
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        res = F.relu(self.conv4(x))

        x = F.relu(self.res1_conv1(res))
        x = F.relu(self.res1_conv2(x))
        x = F.relu(self.res1_conv3(x))

        res += x

        x = F.relu(self.res2_conv1(res))
        x = F.relu(self.res2_conv2(x))
        x = F.relu(self.res2_conv3(x))

        x = self.res2_skip(res) + x

        return x


def create_encoder(encoder_config: EncoderConfig) -> Encoder:
    """Create an encoder from a configuration."""
    path = encoder_config.get("path", None)
    if path is not None:
        encoder_state_dict = torch.load(path, map_location="cpu", weights_only=False)
        encoder = encoder_from_state_dict(encoder_state_dict)
        _logger.info(f"Loaded pretrained encoder from: {path}")
    else:
        obj_type = encoder_config["obj_type"]
        kwargs = encoder_config.get("kwargs", {})
        encoder_cls = utils.str_to_object(obj_type)
        encoder = encoder_cls(**kwargs)  # type: ignore
    return encoder


def encoder_from_state_dict(state_dict: dict) -> Encoder:
    """Extract the encoder type and configuration from a state dictionary.

    Args:
        state_dict: State dictionary of the encoder.

    Returns: The encoder object with weights loaded from the state dictionary.
    """
    _logger.info("Creating encoder from state dictionary.")

    # TODO support DINOv2Encoder

    # Number of output channels of the last encoder layer.
    num_encoder_features = state_dict["res2_conv3.weight"].shape[0]

    encoder = FCNEncoder(dim_out=num_encoder_features)

    # Load encoder weights.
    encoder.load_state_dict(state_dict)

    return encoder
