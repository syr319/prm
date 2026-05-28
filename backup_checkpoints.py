"""
Upload self-trained DistillPRM checkpoints to HuggingFace Hub (private repo).

Usage:
    python3 backup_checkpoints.py --token hf_xxxx [--repo syr319/DistillPRM-checkpoints]

Get your token at: https://huggingface.co/settings/tokens  (write permission)
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Checkpoints to upload ────────────────────────────────────────────────────
# Format: (local_path_relative_to_ROOT, path_in_repo, priority)
CHECKPOINTS = [
    # ★★★ Core results — upload first
    ("models/DistillPRM-1.5B/adaptive_t3/best_model.pt",
     "DistillPRM-1.5B/adaptive_t3/best_model.pt", "core"),

    ("models/DistillPRM-7B/adaptive_t3/best_model.pt",
     "DistillPRM-7B/adaptive_t3/best_model.pt", "core"),

    ("outputs/distillprm-7b-instruct-adaptive-t3/adaptive_t3/best_model.pt",
     "DistillPRM-7B-Instruct/adaptive_t3/best_model.pt", "core"),

    ("outputs/distillprm-7b-iter2/best_model.pt",
     "DistillPRM-7B-iter2/best_model.pt", "core"),

    ("outputs/distillprm-7b-iter2-combined/best_model.pt",
     "DistillPRM-7B-iter2-combined/best_model.pt", "core"),

    # ★★☆ Ablation checkpoints — upload if you have time
    ("models/DistillPRM-1.5B/ce/best_model.pt",
     "DistillPRM-1.5B/ce/best_model.pt", "ablation"),

    ("models/DistillPRM-1.5B/kl/best_model.pt",
     "DistillPRM-1.5B/kl/best_model.pt", "ablation"),

    ("models/DistillPRM-1.5B/ablation_no_error_head/best_model.pt",
     "DistillPRM-1.5B/ablation_no_error_head/best_model.pt", "ablation"),

    ("models/DistillPRM-7B/adaptive_multidim/best_model.pt",
     "DistillPRM-7B/adaptive_multidim/best_model.pt", "ablation"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token",  required=True,
                        help="HuggingFace write token (hf_xxx)")
    parser.add_argument("--repo",   default="syr319/DistillPRM-checkpoints",
                        help="HF repo id (default: syr319/DistillPRM-checkpoints)")
    parser.add_argument("--only",   choices=["core", "ablation", "all"],
                        default="core",
                        help="Which checkpoints to upload (default: core only)")
    args = parser.parse_args()

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=args.token)

    # Create private repo if it doesn't exist
    try:
        create_repo(args.repo, repo_type="model", private=True, token=args.token,
                    exist_ok=True)
        print(f"Repo ready: https://huggingface.co/{args.repo}")
    except Exception as e:
        print(f"Warning creating repo: {e}")

    targets = [c for c in CHECKPOINTS
               if args.only == "all" or c[2] == args.only]

    total = len(targets)
    for idx, (local_rel, repo_path, priority) in enumerate(targets, 1):
        local = ROOT / local_rel
        if not local.exists():
            print(f"[{idx}/{total}] SKIP (not found): {local_rel}")
            continue

        size_gb = local.stat().st_size / 1e9
        print(f"\n[{idx}/{total}] Uploading {local_rel}  ({size_gb:.1f} GB)")
        print(f"         → {args.repo}/{repo_path}")

        try:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=repo_path,
                repo_id=args.repo,
                repo_type="model",
            )
            print(f"         ✓ Done")
        except Exception as e:
            print(f"         ✗ Failed: {e}", file=sys.stderr)

    print(f"\nAll done. View at: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
