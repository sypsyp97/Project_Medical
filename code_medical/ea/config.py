from dataclasses import dataclass
from typing import Tuple


@dataclass
class EvalCfg:
    # 3 architecturally diverse models for chest X-ray
    model_names: Tuple[str, str, str] = ("densenet121", "resnet50", "vit_small")

    # Evaluation seeds for stability measurement
    seeds: Tuple[int, int] = (2025, 2026)

    num_classes: int = 6
    steps: int = 500
    batch_size: int = 256         # A100 80GB can handle 224x224 grayscale at this batch size
    train_lr: float = 1e-3          # AdamW typical LR (was 0.01 for SGD)

    # Robustness score coefficients
    alpha: float = 1.0   # CVaR weight
    beta: float = 1.0    # variance penalty weight
    gamma: float = 0.05  # distribution matching reward weight
