"""Compute similarity metrics between target and generated embeddings.

This script reads two embedding JSON files (target and disturbed-caption generations),
computes cosine similarities per target across its matched generations, and writes
`attack_results.json` next to the disturbed embeddings (or `--output_dir`).
"""

import os
import numpy as np
import argparse
import json



def parse_args():
    """Parse CLI arguments for similarity evaluation."""
    parser = argparse.ArgumentParser(description="Compute similarities between target and generated embeddings")
    parser.add_argument("--target_embeddings", type=str, default=None, help="JSON file with target embeddings")
    parser.add_argument("--disturbed_caption_gen_embeddings", type=str, default=None, help="JSON file with disturbed caption generated embeddings")
    parser.add_argument("--method",type=str,default="threshold", help="Unused legacy flag; kept for CLI compatibility")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results (default: alongside disturbed embeddings)")
    args = parser.parse_args()

    return args

"""
Compute similarity metrics between target and generated embeddings.
Reads two embedding JSON files (target and disturbed-caption generations),
computes cosine similarities per target across its matched generations, and writes
`attack_results.json` next to the disturbed embeddings (or `--output_dir`).
"""

import os
import json
import argparse
import numpy as np


def parse_args():
    """Parse CLI arguments for similarity evaluation."""
    parser = argparse.ArgumentParser(
        description="Compute similarities between target and generated embeddings"
    )
    parser.add_argument(
        "--target_embeddings", type=str, required=True, help="JSON file with target embeddings"
    )
    parser.add_argument(
        "--disturbed_caption_gen_embeddings",
        type=str,
        required=True,
        help="JSON file with disturbed caption generated embeddings",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results (default: alongside disturbed embeddings)",
    )
    # Legacy flag kept for CLI compatibility; not used
    parser.add_argument(
        "--method", type=str, default="threshold", help="Unused legacy flag"
    )
    return parser.parse_args()


def cosine_similarity(a, b, eps: float = 1e-8) -> float:
    """Compute cosine similarity between two 1D vectors (lists/arrays)."""
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))


def main() -> None:
    args = parse_args()

    if args.output_dir is None:
        sample_root = os.path.dirname(args.disturbed_caption_gen_embeddings)
    else:
        sample_root = args.output_dir

    with open(args.target_embeddings, "r", encoding="utf-8") as f:
        targets = json.load(f)
    with open(args.disturbed_caption_gen_embeddings, "r", encoding="utf-8") as f:
        disturbed_caption_gen = json.load(f)

    results = []
    for t in targets:
        t_emb = t.get("embedding")
        if t_emb is None:
            continue
        t_label = t.get("label")
        t_path = t.get("path")

        # Match generated entries by original path
        matched = [s for s in disturbed_caption_gen if s.get("orig_path") == t_path]
        if not matched:
            print(f"[warning] No matching sample embeddings for target: {t_path}")
            continue

        disturbed_similarity = [cosine_similarity(t_emb, s.get("embedding")) for s in matched]

        # If embeddings are concatenated [img|text], split in half for modality-specific similarity
        half = len(t_emb) // 2
        disturbed_img_similarity = [
            cosine_similarity(t_emb[:half], s.get("embedding", [])[:half]) for s in matched
        ]
        disturbed_text_similarity = [
            cosine_similarity(t_emb[half:], s.get("embedding", [])[half:]) for s in matched
        ]

        out_record = {
            "path": t_path,
            "caption": t.get("caption"),
            "label": t_label,
            "disturbed_similarity": disturbed_similarity,
            "disturbed_img_similarity": disturbed_img_similarity,
            "disturbed_text_similarity": disturbed_text_similarity,
        }
        results.append(out_record)

    out_path = os.path.join(sample_root, "attack_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
