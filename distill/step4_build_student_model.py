"""
Step 4: DistillPRM Student Model Architecture.

This file defines the student model and loss functions for the distillation pipeline.
Currently: only builds the model and verifies forward pass (no training yet).

Architecture:
  Backbone : Qwen2.5-Math-1.5B  (AutoModel, last-token pooling)
  Head 1   : score_head     → scalar reward score (sigmoid)
  Head 2   : error_head     → error type logits (7 classes)

Loss functions:
  adaptive_score_loss: difficulty-weighted blend of CE (hard label) + KL (soft label)
    - difficulty = 1 - 2 * |teacher_score - 0.5|
    - simple steps (teacher confident) → more weight on CE (hard label)
    - hard steps (teacher uncertain)   → more weight on KL (soft label)

  error_type_loss: cross-entropy on error type classification (auxiliary task)

  total_loss = adaptive_score_loss + λ_error * error_type_loss

Input format:
  The backbone receives a tokenized string:
    "Question: {question}\n\nContext:\n{context}\n\nCurrent step:\n{current_step}"
  We take the hidden state of the LAST non-padding token as the step representation.

Note: Qwen2.5-Math-1.5B needs to be downloaded separately.
  pip install huggingface_hub
  huggingface-cli download Qwen/Qwen2.5-Math-1.5B --local-dir models/Qwen2.5-Math-1.5B
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT             = Path(__file__).resolve().parents[1]
STUDENT_MODEL_PATH = str(ROOT / "models" / "Qwen2.5-Math-1.5B")

# ─── Error type taxonomy (simplified 4-class) ────────────────────────────────
# Collapsed from original 7-class based on empirical distribution analysis:
#   calculation_error + algebraic_error → computation_error  (75% of wrong steps)
#   wrong_reference                     → propagation_error  (22% of wrong steps)
#   logical_gap + conceptual + irrelev  → reasoning_error    ( 2% of wrong steps)
ERROR_TYPES = {
    0: "correct",
    1: "computation_error",   # arithmetic / algebraic mistake
    2: "propagation_error",   # error from a prior wrong step
    3: "reasoning_error",     # wrong concept / logical gap / irrelevant
}
NUM_ERROR_TYPES = len(ERROR_TYPES)


# ─── Model ────────────────────────────────────────────────────────────────────

class DistillPRM(nn.Module):
    """
    Lightweight discriminative PRM distilled from GenPRM.

    Backbone takes the full input sequence (question + context + current step).
    We pool the last non-padding token's hidden state and pass it through two heads.
    """

    def __init__(
        self,
        model_name_or_path: str = STUDENT_MODEL_PATH,
        num_error_types: int = NUM_ERROR_TYPES,
        head_hidden_dim: int = 256,
        dropout: float = 0.1,
        attn_implementation: str = "eager",
    ):
        super().__init__()

        from transformers import AutoModel, AutoConfig
        config  = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )
        hidden_dim = config.hidden_size

        # Head 1: reward score (0..1)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )

        # Head 2: error type classification (7 classes)
        self.error_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_error_types),
        )

        # Initialize heads with small weights
        for head in [self.score_head, self.error_head]:
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def get_last_token_hidden(self, input_ids, attention_mask):
        """
        Run backbone and return the hidden state of the last non-padding token.
        This is the standard approach for LLM-based reward models.
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state   # [batch, seq_len, hidden]

        # Find last non-padding position for each sample in the batch
        # attention_mask: 1 for real tokens, 0 for padding
        seq_lengths = attention_mask.sum(dim=1) - 1   # [batch] — index of last real token
        batch_size  = input_ids.size(0)
        last_hidden = hidden[torch.arange(batch_size), seq_lengths]  # [batch, hidden]
        return last_hidden

    def forward(self, input_ids, attention_mask):
        """
        Returns:
          score        : [batch]     — predicted step correctness probability (0..1)
          error_logits : [batch, 7]  — unnormalized error type scores
        """
        last_hidden  = self.get_last_token_hidden(input_ids, attention_mask)
        # Cast heads to float32 for numerical stability (backbone may be bfloat16)
        last_hidden_f = last_hidden.float()

        score        = torch.sigmoid(self.score_head(last_hidden_f)).squeeze(-1)
        error_logits = self.error_head(last_hidden_f)
        return score, error_logits


# # ─── Loss functions ──────────────────────────────────────────────────────────

# def compute_difficulty(
#     teacher_score: torch.Tensor,
#     temperature:   float = 1.0,
# ) -> torch.Tensor:
#     """
#     Compute per-step difficulty from teacher's soft score.

#     temperature=1.0 (default):
#       difficulty = 1 - 2 * |teacher_score - 0.5|
#       teacher_score ≈ 0.95 → difficulty ≈ 0.10  (easy)
#       teacher_score ≈ 0.50 → difficulty ≈ 1.00  (hard)

#     temperature>1.0 (temperature-scaled logit):
#       clamped = clamp(teacher_score, 1e-6, 1-1e-6)
#       logit   = log(clamped / (1 - clamped))
#       scaled  = sigmoid(logit / temperature)
#       difficulty = 1 - 2 * |scaled - 0.5|
#       Higher T → flatter sigmoid → more steps treated as "hard" →
#       more KL weight overall, softening the bimodal CE-dominated regime.
#     """
#     if temperature == 1.0:
#         return 1.0 - 2.0 * (teacher_score - 0.5).abs()
#     clamped    = teacher_score.clamp(1e-6, 1.0 - 1e-6)
#     logit      = torch.log(clamped / (1.0 - clamped))
#     scaled     = torch.sigmoid(logit / temperature)
#     return 1.0 - 2.0 * (scaled - 0.5).abs()
def compute_adaptive_temperature(
    teacher_score: torch.Tensor,
    t_min:         float = 1.0,
    t_max:         float = 5.0,
) -> torch.Tensor:
    """
    Compute per-step adaptive temperature based on raw score extremity.
    
    Extreme scores (near 0 or 1) → large T (压平分布)
    Uncertain scores (near 0.5)  → small T (保留信息)
    """
    raw_difficulty = 1.0 - 2.0 * (teacher_score - 0.5).abs()
    T = t_max - (t_max - t_min) * raw_difficulty
    return T


def compute_difficulty(
    teacher_score:        torch.Tensor,
    temperature:          float = 1.0,
    # 方向2：自适应温度
    adaptive_temperature: bool  = False,
    t_min:                float = 1.0,
    t_max:                float = 5.0,
    # 方向1：多维度难度
    step_index:           Optional[torch.Tensor] = None,
    total_steps:          Optional[torch.Tensor] = None,
    step_length:          Optional[torch.Tensor] = None,
    w_score:              float = 1.0,
    w_position:           float = 0.0,
    w_length:             float = 0.0,
) -> torch.Tensor:
    """
    Compute per-step difficulty from teacher's soft score,
    optionally incorporating adaptive temperature (方向2)
    and multi-dimensional difficulty signals (方向1).

    默认参数 = 原始行为，完全向后兼容：
      temperature=1.0, adaptive_temperature=False,
      w_score=1.0, w_position=0.0, w_length=0.0

    方向2（自适应温度）：
      adaptive_temperature=True, t_min=1.0, t_max=5.0
      每个样本根据自身score的极端程度获得不同的T

    方向1（多维度难度）：
      w_score=0.6, w_position=0.2, w_length=0.2
      需要同时传入 step_index, total_steps, step_length
    """
    # ── Step 1: 计算score-based difficulty ──────────────────────────────────
    if not adaptive_temperature and temperature == 1.0:
        # 原始公式，不做任何温度缩放
        score_difficulty = 1.0 - 2.0 * (teacher_score - 0.5).abs()
    else:
        clamped = teacher_score.clamp(1e-6, 1.0 - 1e-6)
        logit   = torch.log(clamped / (1.0 - clamped))

        if adaptive_temperature:
            # 方向2：每个样本有自己的T
            T = compute_adaptive_temperature(teacher_score, t_min, t_max)
            scaled = torch.sigmoid(logit / T)
        else:
            # 原始固定温度
            scaled = torch.sigmoid(logit / temperature)

        score_difficulty = 1.0 - 2.0 * (scaled - 0.5).abs()

    # ── Step 2: 如果只用score信号，直接返回 ─────────────────────────────────
    if w_position == 0.0 and w_length == 0.0:
        return score_difficulty

    # ── Step 3: 多维度混合（方向1）──────────────────────────────────────────
    difficulty = w_score * score_difficulty

    # 位置信号：越靠后越难
    if w_position > 0.0 and step_index is not None and total_steps is not None:
        position_difficulty = step_index.float() / total_steps.float().clamp(min=1)
        difficulty = difficulty + w_position * position_difficulty

    # 长度信号：越长越难（500字符归一化）
    if w_length > 0.0 and step_length is not None:
        length_difficulty = (step_length.float() / 500.0).clamp(0.0, 1.0)
        difficulty = difficulty + w_length * length_difficulty

    return difficulty.clamp(0.0, 1.0)

def adaptive_score_loss(
    student_score:          torch.Tensor,
    teacher_score:          torch.Tensor,
    hard_label:             torch.Tensor,
    alpha_easy:             float = 0.9,
    alpha_hard:             float = 0.1,
    difficulty_temperature: float = 1.0,
) -> torch.Tensor:
    """
    Difficulty-adaptive blend of hard-label CE loss and soft-label KL loss.

    For easy steps (teacher confident): alpha ≈ alpha_easy → primarily CE
    For hard steps (teacher uncertain): alpha ≈ alpha_hard → primarily KL

    Loss = alpha * CE(student, hard_label) + (1 - alpha) * KL(student || teacher)

    Args:
      student_score          : [batch]  float, P(correct) from student (after sigmoid)
      teacher_score          : [batch]  float, P(correct) from GenPRM (soft label)
      hard_label             : [batch]  float (0.0 or 1.0), binary correctness label
      alpha_easy             : CE weight when difficulty = 0 (easy)
      alpha_hard             : CE weight when difficulty = 1 (hard)
      difficulty_temperature : temperature for logit-scaled difficulty (1.0 = original)
    """
    difficulty = compute_difficulty(teacher_score, temperature=difficulty_temperature)   # [batch]

    # Difficulty-weighted CE coefficient (per step)
    alpha = alpha_easy * (1.0 - difficulty) + alpha_hard * difficulty  # [batch]

    # CE loss: standard binary cross-entropy with hard labels
    loss_ce = F.binary_cross_entropy(
        student_score, hard_label.float(), reduction="none"
    )   # [batch]

    # KL divergence: KL(student || teacher)
    # Treat as binary distributions: [P(wrong), P(correct)]
    student_dist = torch.stack([1.0 - student_score, student_score], dim=-1)   # [batch, 2]
    teacher_dist = torch.stack([1.0 - teacher_score, teacher_score], dim=-1)   # [batch, 2]
    loss_kl = F.kl_div(
        torch.log(student_dist + 1e-8),
        teacher_dist,
        reduction="none",
    ).sum(dim=-1)   # [batch]

    # Weighted combination
    loss = alpha * loss_ce + (1.0 - alpha) * loss_kl   # [batch]
    return loss.mean()


def error_type_loss(
    error_logits: torch.Tensor,
    error_labels: torch.Tensor,
) -> torch.Tensor:
    """Plain cross-entropy (backward-compatible). Prefer error_type_focal_loss."""
    return F.cross_entropy(error_logits, error_labels)


def error_type_focal_loss(
    error_logits:  torch.Tensor,
    error_labels:  torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    gamma:         float = 2.0,
) -> torch.Tensor:
    """
    Class-weighted focal loss for error type classification.

    FL(p_t) = -w_t * (1 - p_t)^gamma * log(p_t)

    Args:
      error_logits  : [batch, K]  unnormalized logits from error_head
      error_labels  : [batch]     int64 class indices
      class_weights : [K] float   per-class weights on the same device as logits;
                                  if None, falls back to plain CE
      gamma         : float       focal parameter — 0 disables focal modulation,
                                  2 is standard (Lin et al. 2017)
    """
    # Weighted CE per sample (no aggregation yet)
    ce = F.cross_entropy(
        error_logits, error_labels,
        weight=class_weights, reduction="none",
    )  # [batch]

    if gamma > 0.0:
        # p_t = probability assigned to the correct class
        pt = torch.exp(-ce)                          # [batch], in (0, 1]
        focal_weight = (1.0 - pt) ** gamma           # [batch]
        return (focal_weight * ce).mean()

    return ce.mean()


def total_loss(
    student_score:  torch.Tensor,
    teacher_score:  torch.Tensor,
    hard_label:     torch.Tensor,
    error_logits:   torch.Tensor,
    error_labels:   torch.Tensor,
    lambda_error:   float = 0.1,
    class_weights:  Optional[torch.Tensor] = None,
    focal_gamma:    float = 2.0,
) -> dict:
    """
    Combined training loss.

    Returns a dict with individual components for logging.
    """
    l_score = adaptive_score_loss(student_score, teacher_score, hard_label)
    l_error = error_type_focal_loss(
        error_logits, error_labels,
        class_weights=class_weights,
        gamma=focal_gamma,
    )
    l_total = l_score + lambda_error * l_error
    return {
        "loss":       l_total,
        "loss_score": l_score,
        "loss_error": l_error,
    }


# ─── Tokenization helper ─────────────────────────────────────────────────────

def build_input_text(question: str, context: str, current_step: str) -> str:
    """Format a step record into the input string for DistillPRM."""
    parts = [f"Question: {question}"]
    if context:
        parts.append(f"Context:\n{context}")
    parts.append(f"Current step:\n{current_step}")
    return "\n\n".join(parts)


# ─── Forward pass test ───────────────────────────────────────────────────────

def test_forward_pass():
    """
    Verify model loads and forward pass runs correctly.
    Tests both real model (if available) and dummy fallback.
    """
    print("=" * 55)
    print("DistillPRM Forward Pass Test")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Try loading real model ──
    model_available = Path(STUDENT_MODEL_PATH).exists() and (
        Path(STUDENT_MODEL_PATH) / "config.json"
    ).exists()

    if model_available:
        print(f"\nLoading backbone from {STUDENT_MODEL_PATH} ...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_PATH, trust_remote_code=True)
        model     = DistillPRM(model_name_or_path=STUDENT_MODEL_PATH)
        model.to(device)
        model.eval()

        # Test input
        texts = [
            build_input_text(
                question="If $a \\ge b > 1$, what is the largest possible value of "
                         "$\\log_a (a/b) + \\log_b (b/a)$?",
                context="",
                current_step="We are given the expression $\\log_a (a/b) + \\log_b (b/a)$.",
            ),
            build_input_text(
                question="Find the sum of all positive integers from 1 to 100.",
                context="First, we apply the formula n(n+1)/2.",
                current_step="For n=100, the sum is 100*101/2 = 5050.",
            ),
        ]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
    else:
        print(f"\nModel not found at {STUDENT_MODEL_PATH}.")
        print("Running dummy forward pass to verify architecture...\n")

        # Dummy model config matching Qwen2.5-Math-1.5B dimensions
        from transformers import AutoConfig, AutoModel
        from unittest.mock import MagicMock, patch

        class DummyConfig:
            hidden_size   = 1536
            architectures = ["Qwen2ForCausalLM"]

        class DummyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = DummyConfig()
                self.embed  = nn.Embedding(1000, 1536)

            def forward(self, input_ids, attention_mask):
                from types import SimpleNamespace
                h = self.embed(input_ids)   # [B, T, 1536]
                return SimpleNamespace(last_hidden_state=h)

        # Monkey-patch to use dummy backbone
        class DummyDistillPRM(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone  = DummyBackbone()
                hidden_dim     = 1536
                self.score_head = nn.Sequential(
                    nn.Linear(hidden_dim, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1)
                )
                self.error_head = nn.Sequential(
                    nn.Linear(hidden_dim, 256), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(256, NUM_ERROR_TYPES)
                )

            def get_last_token_hidden(self, input_ids, attention_mask):
                out         = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
                hidden      = out.last_hidden_state
                seq_lengths = attention_mask.sum(dim=1) - 1
                return hidden[torch.arange(input_ids.size(0)), seq_lengths]

            def forward(self, input_ids, attention_mask):
                last_h       = self.get_last_token_hidden(input_ids, attention_mask).float()
                score        = torch.sigmoid(self.score_head(last_h)).squeeze(-1)
                error_logits = self.error_head(last_h)
                return score, error_logits

        model = DummyDistillPRM().to(device)
        model.eval()

        # Dummy tokenized input
        batch_size, seq_len = 2, 128
        input_ids      = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long).to(device)
        # Simulate padding on the second sample
        attention_mask[1, 100:] = 0

    # ── Run forward pass ──
    print("Running forward pass...")
    with torch.no_grad():
        score, error_logits = model(input_ids, attention_mask)

    print(f"\nForward pass outputs:")
    print(f"  score shape        : {score.shape}   (expected: [batch])")
    print(f"  error_logits shape : {error_logits.shape}   (expected: [batch, 7])")
    print(f"  score values       : {score.tolist()}")
    print(f"  error probs        : {F.softmax(error_logits, dim=-1).tolist()}")

    assert score.shape == (input_ids.size(0),),         f"score shape mismatch: {score.shape}"
    assert error_logits.shape == (input_ids.size(0), NUM_ERROR_TYPES), \
        f"error_logits shape mismatch: {error_logits.shape}"
    assert (score >= 0).all() and (score <= 1).all(),    "scores out of [0,1] range"
    print("\n✓ Shape assertions passed.")

    # ── Test loss functions ──
    print("\nTesting loss functions...")
    batch = input_ids.size(0)

    student_score  = score
    teacher_score  = torch.tensor([0.92, 0.31], device=device)   # high / low confidence
    hard_label     = torch.tensor([1.0, 0.0],   device=device)
    error_labels   = torch.tensor([0, 1],        device=device, dtype=torch.long)

    loss_out = total_loss(
        student_score  = student_score,
        teacher_score  = teacher_score,
        hard_label     = hard_label,
        error_logits   = error_logits,
        error_labels   = error_labels,
        lambda_error   = 0.1,
    )

    print(f"  total_loss  : {loss_out['loss'].item():.4f}")
    print(f"  loss_score  : {loss_out['loss_score'].item():.4f}")
    print(f"  loss_error  : {loss_out['loss_error'].item():.4f}")
    assert loss_out["loss"].item() >= 0,       "negative total loss"
    assert loss_out["loss_score"].item() >= 0, "negative score loss"
    assert loss_out["loss_error"].item() >= 0, "negative error loss"
    print("✓ Loss function assertions passed.")

    # ── Difficulty sanity check ──
    print("\nDifficulty check:")
    test_scores   = torch.tensor([0.95, 0.70, 0.50, 0.30, 0.05])
    difficulties  = compute_difficulty(test_scores)
    for s, d in zip(test_scores.tolist(), difficulties.tolist()):
        label = "hard" if d > 0.5 else "easy"
        print(f"  teacher_score={s:.2f}  difficulty={d:.2f}  ({label})")

    # ── Parameter count ──
    total_params    = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters()) \
                      if hasattr(model, "backbone") else 0
    head_params     = total_params - backbone_params
    print(f"\nParameter count:")
    print(f"  Backbone  : {backbone_params / 1e6:.1f}M")
    print(f"  Heads     : {head_params / 1e3:.1f}K")
    print(f"  Total     : {total_params / 1e6:.1f}M")

    print("\n" + "=" * 55)
    print("All checks passed — DistillPRM architecture is ready.")
    print("=" * 55)


if __name__ == "__main__":
    test_forward_pass()
