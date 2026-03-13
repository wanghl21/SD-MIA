#!/usr/bin/env python3
"""
API-based image generation (no local diffusion model).

This script mirrors the behavior of src/inference.py but calls a text-to-image API
instead of loading a local Stable Diffusion model. It supports:
 - Generating multiple images per caption
 - Optional perturbed-caption second pass
 - Multithreaded requests to speed up generation
 - Saving args, jobs.json, and metafile.json for traceability

Providers: any OpenAI-compatible Images API endpoint. For example:
 - Model: "dall-e-3" (OpenAI Images API)

Set OPENAI_API_KEY and (optionally) OPENAI_BASE_URL in the environment, or pass
--api_key and --api_base_url explicitly.

Note: Some features in local inference.py do not translate 1:1 to hosted APIs.
 - Seed reproducibility is typically not supported by official OpenAI endpoints.
 - add_original_image_for_generation (img2img) may not be supported by all providers;
   in this script, we currently fall back to text-only if the provider doesn't support edits/variations.
"""

from src.api_paint import generate_image as api_generate_image
import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from tqdm.auto import tqdm

# Reuse the unified API helper that wraps an OpenAI-compatible Images.generate.
# Ensure repository root is on sys.path when running as `python src/inference_api.py`.
import sys
CURRENT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("API-based image generation (no local model)")
    # Model/API
    p.add_argument("--model", type=str, required=True,
                   help="API model name, e.g. dall-e-3")
    p.add_argument("--api_base_url", type=str,
                   default=os.getenv("OPENAI_BASE_URL"), help="OpenAI-compatible base URL")
    p.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"),
                   help="API key; reads OPENAI_API_KEY by default")
    p.add_argument("--api_keys", type=str, default=None,
                   help="Comma/space-separated list of API keys for parallel usage")
    p.add_argument("--api_keys_file", type=str, default=None,
                   help="File with one API key per line for parallel usage")

    # Generation behavior (kept similar to src/inference.py)
    p.add_argument("--num_validation_images", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Number of concurrent API requests")
    p.add_argument("--inference", type=int, default=40,
                   help="Ignored for API; kept for CLI parity")
    p.add_argument("--width", type=int, default=1024,
                   help="Output width; mapped to API size string WxH")
    p.add_argument("--height", type=int, default=1024,
                   help="Output height; mapped to API size string WxH")

    # Data and output
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to JSON list of {path, caption, label, ...}")
    # Keep default path style consistent with src/inference.py
    import time
    p.add_argument(
        "--save_dir",
        type=str,
        help="Output directory (images + metadata)",
    )

    # Repro and extras (limited API parity)
    p.add_argument("--seed", type=int, default=None,
                   help="Most APIs ignore this; kept for parity")
    p.add_argument("--add_original_image_for_generation", action="store_true",
                   help="Attempt img2img if provider supports; else fallback to text-only")
    p.add_argument("--img2img_strength", type=float, default=0.8,
                   help="Currently not passed to API; kept for parity")

    # Perturbation controls
    p.add_argument("--use_perturbed_caption", action="store_true",
                   help="Also generate with perturbed captions if available")
    p.add_argument(
        "--perturb_type",
        type=str,
        choices=["synonym_replacement", "random_deletion", "style_insertion"],
        help="Type of perturbation to pick when --use_perturbed_caption is set",
    )

    return p.parse_args()


def _snap_dim(x: int) -> int:
    x = int(x)
    if x < 64:
        x = 64
    return max(64, int(round(x / 8)) * 8)


def _size_str(w: int, h: int) -> str:
    return f"{w}x{h}"


def _load_dataset(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d


def _save_json(path: str, obj) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ------------- Error logging utilities (thread-safe) -------------
_ERROR_LOG_PATH: Optional[str] = None
_ERROR_LOCK = threading.Lock()


def _init_error_logger(save_dir: str) -> None:
    global _ERROR_LOG_PATH
    _ensure_dir(save_dir)
    _ERROR_LOG_PATH = os.path.join(save_dir, "errors.jsonl")


def _append_error_log(record: Dict) -> None:
    if not _ERROR_LOG_PATH:
        return
    try:
        with _ERROR_LOCK:
            with open(_ERROR_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Silent guard: never break generation because of logging
        pass


def _load_api_keys(args: argparse.Namespace) -> List[str]:
    keys: List[str] = []
    # File wins
    if args.api_keys_file:
        try:
            with open(args.api_keys_file, "r", encoding="utf-8") as f:
                for ln in f:
                    k = ln.strip()
                    if k:
                        keys.append(k)
        except Exception as e:
            raise RuntimeError(f"Failed to read --api_keys_file: {e}")
    # Inline list
    if not keys and args.api_keys:
        parts = re.split(r"[,;\s]+", str(args.api_keys))
        keys = [p.strip() for p in parts if p.strip()]
    # Single key
    if not keys and args.api_key:
        keys = [args.api_key]
    # Env fallback (support both OpenAI and Gemini)
    if not keys:
        env_list = os.getenv("GEMINI_API_KEYS")
        if env_list:
            parts = re.split(r"[,;\s]+", env_list)
            keys = [p.strip() for p in parts if p.strip()]
    if not keys:
        envk = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if envk:
            keys = [envk]
    if not keys:
        raise RuntimeError(
            "No API keys provided. Use --api_keys_file, --api_keys, --api_key or set OPENAI_API_KEY.")
    return keys


def _build_jobs(dataset: List[Dict], out_dir: str, n_per_caption: int, use_perturbed: bool, perturb_type: Optional[str]) -> Tuple[List[Dict], List[Dict]]:
    jobs: List[Dict] = []
    meta: List[Dict] = []

    for data in dataset:
        original_path = data['path']
        if not original_path:
            continue
        original_filename = Path(original_path).name
        parts = original_filename.split(".")
        if len(parts) < 2:
            base, ext = original_filename, "jpg"
        else:
            base, ext = parts[0], parts[1]

        for j in range(n_per_caption):
            filename = f"{data.get('label', 0)}_{base}_{j+1:02}.{ext}"

            save_path = os.path.join(out_dir, "images", filename)
            _ensure_dir(os.path.dirname(save_path))

            perturbed_caption = None
            perturb_save_path = None
            if use_perturbed:
                if perturb_type is None:
                    raise ValueError(
                        "If --use_perturbed_caption is set, --perturb_type must be provided.")
                perturb_save_path = os.path.join(
                    out_dir, "images_perturbed_caption", filename)
                _ensure_dir(os.path.dirname(perturb_save_path))

                # Locate perturbed caption
                if isinstance(data.get("perturbations"), list):
                    for p in data["perturbations"]:
                        if p.get("method") == perturb_type:
                            perturbed_caption = p.get("perturbed_caption")
                            break
                if perturbed_caption is None and isinstance(data.get(perturb_type), str):
                    perturbed_caption = data[perturb_type]

                if perturbed_caption is None:
                    print(
                        f"Warning: no perturbed caption found for method '{perturb_type}' in item {original_filename}. Skipping its perturbed image."
                    )

            jobs.append({
                "caption": data["caption"],
                "perturbed_caption": perturbed_caption,
                "save_path": save_path,
                "perturb_save_path": perturb_save_path,
                "image_path": original_path,
                "label": data["label"],
            })

            meta.append({
                "caption": data["caption"],
                "label": data["label"],
                "path": save_path,
                "repeat": j + 1,
                "orig_path": original_path,
            })
            if use_perturbed and perturbed_caption and perturb_save_path:
                meta.append({
                    "caption": perturbed_caption,
                    "label": data['label'],
                    "path": perturb_save_path,
                    "repeat": j + 1,
                    "orig_path": original_path,
                })

    return jobs, meta


def _generate_single(
    prompt: str,
    out_path: str,
    model: str,
    size_str: str,
    api_base_url: Optional[str],
    api_key: Optional[str],
    provider: Optional[str] = None,
) -> Optional[str]:
    """Generate one image and save to out_path. Returns the saved path or None on error."""
    # Add retry for rate limit (429). For other errors, log and stop.
    max_retries = 8
    attempt = 0
    while True:
        attempt += 1
        try:
            img: Image.Image = api_generate_image(
                prompt=prompt,
                model=model,
                size=size_str,
                quality="standard",
                base_url=api_base_url,
                api_key=api_key,
                provider=provider,
            )
            img.save(out_path)
            return out_path
        except Exception as e:
            err_text = str(e)
            status_code = None
            # Try to extract an HTTP status code if present in the string, e.g., "Error code: 429 - { ... }"
            m = re.search(
                r"(Error code:|status code[:\s])\s*(\d{3})", err_text, flags=re.IGNORECASE)
            if m:
                try:
                    status_code = int(m.group(2))
                except Exception:
                    status_code = None

            # 429 handling: wait then retry (do not return)
            is_rate_limit = (status_code == 429) or (
                "429" in err_text and "Error code" in err_text) or ("rate limit" in err_text.lower())
            if is_rate_limit:
                # Parse suggested wait from message if available (e.g., "retry after 28 seconds")
                wait_s = 30
                m2 = re.search(r"retry\s+after\s+(\d+)\s*seconds?",
                               err_text, flags=re.IGNORECASE)
                if m2:
                    try:
                        wait_s = max(1, int(m2.group(1)))
                    except Exception:
                        wait_s = 30
                else:
                    # Exponential backoff with cap
                    wait_s = min(60, 2 ** min(attempt, 5))

                _append_error_log({
                    "time": datetime.utcnow().isoformat() + "Z",
                    "event": "rate_limit",
                    "attempt": attempt,
                    "wait_seconds": wait_s,
                    "status_code": status_code,
                    "model": model,
                    "out_path": out_path,
                    "prompt": prompt,
                    "error": err_text,
                })
                tqdm.write(
                    f"Rate limited. Waiting {wait_s}s then retrying (attempt {attempt}).")
                time.sleep(wait_s)
                if attempt < max_retries:
                    continue
                else:
                    tqdm.write(
                        "Max retries reached for rate limit; skipping this image.")
                    return None

            # 500 content policy violation: log to file and stop for this item
            is_content_policy = (status_code == 500 and (
                "content_policy_violation" in err_text or "safety system" in err_text.lower() or "rejected" in err_text.lower()))
            if is_content_policy:
                _append_error_log({
                    "time": datetime.utcnow().isoformat() + "Z",
                    "event": "content_policy_violation",
                    "status_code": status_code,
                    "model": model,
                    "out_path": out_path,
                    "prompt": prompt,
                    "error": err_text,
                })
                tqdm.write(f"Content policy violation, logged: {err_text}")
                return None

            # Other errors: log and stop
            _append_error_log({
                "time": datetime.utcnow().isoformat() + "Z",
                "event": "error",
                "status_code": status_code,
                "model": model,
                "out_path": out_path,
                "prompt": prompt,
                "error": err_text,
            })
            tqdm.write(f"Error generating image: {err_text}")
            return None


def main() -> None:
    args = parse_args()

    # Normalize resolution and map to API size string (WxH)
    out_w = _snap_dim(args.width)
    out_h = _snap_dim(args.height)
    size_str = _size_str(out_w, out_h)

    # Prepare save directory and persist args
    final_save_dir = args.save_dir
    _ensure_dir(final_save_dir)
    _init_error_logger(final_save_dir)
    try:
        # Mask sensitive fields when saving
        safe_args = dict(vars(args))
        for k in ("api_key", "api_keys"):
            if safe_args.get(k):
                safe_args[k] = "***"
        _save_json(os.path.join(final_save_dir,
                   "inference_args.json"), safe_args)
    except Exception as e:
        print(f"Warning: failed to save inference_args.json: {e}")

    # Warn about img2img flag, which is not supported uniformly by API providers here
    if args.add_original_image_for_generation:
        print(
            "[note] --add_original_image_for_generation requested, but this API-based script currently runs text-only generation. "
            "Some providers support image edits/variations via different endpoints; integrate as needed."
        )

    # Load dataset and build jobs
    dataset = _load_dataset(args.data_dir)
    jobs, meta_file = _build_jobs(
        dataset=dataset,
        out_dir=final_save_dir,
        n_per_caption=args.num_validation_images,
        use_perturbed=args.use_perturbed_caption,
        perturb_type=args.perturb_type,
    )

    # Load API keys (possibly multiple) for parallel usage
    api_keys = _load_api_keys(args)
    key_count = max(1, len(api_keys))

    # Provider detection (force Gemini path when requested)
    provider: Optional[str] = None
    if (args.model or "").lower().startswith("gemini"):
        provider = "gemini"
    elif args.api_base_url and "generativelanguage" in str(args.api_base_url).lower():
        provider = "gemini"

    # Summary counts
    perturbed_count = sum(1 for j in jobs if args.use_perturbed_caption and j.get(
        "perturbed_caption") and j.get("perturb_save_path"))
    total_images = len(jobs) + \
        (perturbed_count if args.use_perturbed_caption else 0)

    # Concurrency control
    threads = max(1, int(args.batch_size))

    successful_paths: set[str] = set()

    with tqdm(total=total_images, desc="Generating images", unit="img") as pbar:
        # Pass 1: original captions
        tasks1 = []
        for it in jobs:
            tasks1.append((it["caption"], it["save_path"]))
        with ThreadPoolExecutor(max_workers=threads) as ex:
            futures = []
            for idx, (prompt, out_path) in enumerate(tasks1):
                key = api_keys[idx % key_count]
                futures.append(
                    ex.submit(
                        _generate_single,
                        prompt, out_path,
                        args.model, size_str, args.api_base_url, key, provider,
                    )
                )
            for f in as_completed(futures):
                outp = f.result()
                if outp:
                    successful_paths.add(outp)
                pbar.update(1)

        # Pass 2: perturbed captions (optional)
        if args.use_perturbed_caption:
            tasks2 = []
            for it in jobs:
                pc = it.get("perturbed_caption")
                ps = it.get("perturb_save_path")
                if pc and ps:
                    tasks2.append((pc, ps))
            if tasks2:
                with ThreadPoolExecutor(max_workers=threads) as ex:
                    futures2 = []
                    for idx, (prompt, out_path) in enumerate(tasks2):
                        key = api_keys[idx % key_count]
                        futures2.append(
                            ex.submit(
                                _generate_single,
                                prompt, out_path,
                                args.model, size_str, args.api_base_url, key, provider,
                            )
                        )
                    for f in as_completed(futures2):
                        outp = f.result()
                        if outp:
                            successful_paths.add(outp)
                        pbar.update(1)

    # Save metadata
    # Filter meta to only successful items
    filtered_meta = []
    for rec in meta_file:
        p = rec.get("path")
        if not p:
            continue
        if p in successful_paths or os.path.exists(p):
            filtered_meta.append(rec)

    try:
        _save_json(os.path.join(final_save_dir, "jobs.json"), jobs)
        _save_json(os.path.join(final_save_dir,
                   "metafile.json"), filtered_meta)
    except Exception as e:
        print(f"Warning: failed to save metadata: {e}")


if __name__ == "__main__":
    main()
