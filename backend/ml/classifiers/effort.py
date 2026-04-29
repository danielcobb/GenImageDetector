import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import CLIPVisionConfig, CLIPVisionModel

from ml.classifiers.pytorch_base import PyTorchClassifier

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# ViT-L/14 self-attn projections are 1024x1024. Upstream uses r=1023, leaving
# a rank-1 trainable residual. See YZY-stack/Effort-AIGI-Detection.
RESIDUAL_RANK = 1


class SVDResidualLinear(nn.Module):
    """nn.Linear split into a frozen `weight_main` plus a trainable
    U @ diag(S) @ V residual. Inference only — training-only buffers
    (S_r/U_r/V_r/weight_original_fnorm) are intentionally omitted."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight_main = nn.Parameter(
            torch.zeros(out_features, in_features), requires_grad=False
        )
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.S_residual = nn.Parameter(torch.zeros(RESIDUAL_RANK))
        self.U_residual = nn.Parameter(torch.zeros(out_features, RESIDUAL_RANK))
        self.V_residual = nn.Parameter(torch.zeros(RESIDUAL_RANK, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        return F.linear(x, self.weight_main + residual, self.bias)


def _swap_self_attn_linears(module: nn.Module) -> None:
    for name, child in module.named_children():
        if "self_attn" in name:
            for sub_name, sub in list(child.named_modules()):
                if isinstance(sub, nn.Linear):
                    parent = child
                    parts = sub_name.split(".")
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    setattr(
                        parent,
                        parts[-1],
                        SVDResidualLinear(
                            sub.in_features,
                            sub.out_features,
                            bias=sub.bias is not None,
                        ),
                    )
        else:
            _swap_self_attn_linears(child)


class _EffortModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)["pooler_output"]
        return self.head(feat)


class EffortClassifier(PyTorchClassifier):
    """Effort detector — CLIP ViT-L/14 with SVD-residual self-attention,
    fine-tuned for AI-generated image detection (label 0 = real, 1 = fake)."""

    def __init__(self, model_path: str, device: str = None, quiet: bool = False):
        self.quiet = quiet
        super().__init__(model_path, device)

    def get_model_architecture(self) -> nn.Module:
        config = CLIPVisionConfig.from_pretrained("openai/clip-vit-large-patch14")
        backbone = CLIPVisionModel(config).vision_model
        _swap_self_attn_linears(backbone)
        head = nn.Linear(config.hidden_size, 2)
        return _EffortModel(backbone, head)

    def load_weights(self):
        if not self.quiet:
            print(f"Loading Effort weights from {self.model_path}")
        state_dict = torch.load(
            self.model_path, map_location=self.device, weights_only=True
        )
        cleaned = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if not self.quiet:
            print(
                f"[Effort] load_state_dict missing={len(missing)} unexpected={len(unexpected)}"
            )

    def get_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ]
        )

    def postprocess(self, output: torch.Tensor) -> float:
        # output: [1, 2] logits — col 0 = real, col 1 = fake
        prob_real = torch.softmax(output, dim=1)[0, 0].item()
        confidence = round(prob_real * 100, 1)
        if not self.quiet:
            print(f"[Effort Debug] Prob(real): {prob_real:.4f}")
        return confidence
