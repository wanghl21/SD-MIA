#!/usr/bin/env python3
"""Simple caption perturbation using OpenAI-compatible chat API.

Input: JSON file (list of objects containing at least 'caption').
Output: JSON file where each item includes original fields plus one or more perturbed captions.

Features:
- Accepts three prompt templates (placeholders: {caption}) provided via CLI.
- Can run in parallel and accept multiple API keys (--api_keys or --api_keys_file).
- Minimal, readable code.
"""
import json
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Union, Optional
from tqdm import tqdm

from openai import OpenAI

def build_prompt(caption: str, perturbation_type: str) -> List[Dict[str, str]]:
    """Build a prompt for the requested perturbation type.

    Args:
        caption: Original caption text.
        perturbation_type: One of 'style_view_perturbation', 'semantic_view_perturbation', 'token_view_perturbation'.

    Returns:
        A list of role/content messages suitable for a chat completion API.
    """
    
    if perturbation_type == "style_view_perturbation":
        # Keep semantics the same, change the artistic style
        system = (
            "Rewrite the given image caption so that the CONTENT/SUBJECT remains EXACTLY the same, but change the ARTISTIC STYLE of the image.\n"
            "Add ONLY 1-2 style modifiers like 'photorealistic', 'cinematic', 'oil painting', 'cartoon style', etc. before, after, or within the caption.\n\n"
            "Examples:\n"
            "- 'a cat on a chair' → 'photorealistic, a cat on a chair'\n"
            "- 'UK Active logo' → 'UK Active logo, in the style of oil painting'\n"
            "- 'person smiling' → 'a watercolor painting of person smiling'\n"
            "- 'sunset over mountains' → 'cinematic, sunset over mountains'\n"
            "- 'Salad with chestnuts' → 'Salad with chestnuts, digital art'\n\n"
            "Common style modifiers (choose 1-2 only):\n"
            "- photorealistic, cinematic, highly detailed, 4k\n"
            "- oil painting, watercolor painting, acrylic painting\n"
            "- pencil sketch, ink drawing, charcoal drawing  \n"
            "- cartoon style, anime style, manga\n"
            "- digital art, 3D render, vector art\n"
            "- in the style of [artist/movement]\n\n"
            "Rules: "
            "1) Keep the EXACT same content/subject. "
            "2) Add ONLY 1-2 style modifiers (NOT more). "
            "3) Output only the new caption, no quotes or extra text. "
            "4) Ensure that the output caption conforms to objective facts."
        )
    
    elif perturbation_type == "semantic_view_perturbation":
        # Change semantics/content, keep the artistic style
        system = (
            "Rewrite the given image caption so that the CONTENT/SUBJECT is CHANGED, but keep the SAME ARTISTIC STYLE.\n"
            "Keep the same style modifiers (if any) but change the main subject/content.\n\n"
            "Examples:\n"
            "- 'photorealistic, a cat on a chair' → 'photorealistic, a dog on a sofa'\n"
            "- 'UK Active logo' → 'Nike logo' (both simple descriptions without style modifiers)\n"
            "- 'oil painting of mountains' → 'oil painting of an ocean'\n"
            "- 'sunset over mountains, digital art' → 'sunrise over cityscape, digital art'\n"
            "- 'person smiling' → 'person running' (both simple, no style modifiers)\n\n"
            "Rules: "
            "1) Change the subject/content to something DIFFERENT. "
            "2) Keep the SAME style modifiers if present in the original. "
            "3) If no style modifiers in original, keep the same simple format. "
            "4) Output only the new caption, no quotes or extra text. "
            "5) Ensure that the output caption conforms to objective facts."
        )
    
    elif perturbation_type == "token_view_perturbation":
        # Rephrase caption while preserving content/subject and style
        system = (
            "Rewrite the given image caption by REPHRASING the text while PRESERVING BOTH the ORIGINAL CONTENT/SUBJECT and the ARTISTIC STYLE exactly.\n"
            "Do NOT change the main subject or any style modifiers (e.g., 'photorealistic', 'oil painting', 'cartoon style'). Only modify wording, word order, or small descriptive phrasing.\n\n"
            "Examples:\n"
            "- 'photorealistic, a cat on a chair' → 'photorealistic, a cat sitting on a chair'\n"
            "- 'oil painting of mountains at sunset' → 'oil painting of mountain peaks at sunset'\n"
            "- 'cartoon style, child playing' → 'cartoon style, a child at play'\n"
            "- 'digital art, futuristic cityscape' → 'digital art, a futuristic city skyline'\n\n"
            "Rules: "
            "1) Preserve the EXACT subject/content and any style modifiers. Do NOT introduce new subjects or styles. "
            "2) Only rephrase or slightly rearrange words; avoid adding new objects or changing factual content. "
            "3) Output only the new caption, no quotes or extra text. "
            "4) Ensure the output remains truthful and consistent with the original caption."
        )
    
    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")

    user = (
        f"Original caption: {caption}\n"
        "New caption:"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]


def _load_api_keys(args):
    """Load one or more API keys from CLI flags or environment."""
    keys = []
    if getattr(args, "api_keys_file", None):
        with open(args.api_keys_file, "r", encoding="utf-8") as f:
            keys = [ln.strip() for ln in f if ln.strip()]
    if not keys and getattr(args, "api_keys", None):
        keys = [p for p in re.split(r"[,;\s]+", args.api_keys) if p]
    if not keys and getattr(args, "api_key", None):
        keys = [args.api_key]
    if not keys:
        env = os.getenv("OPENAI_API_KEY")
        if env:
            keys = [env]
    if not keys:
        raise RuntimeError("No API keys provided")
    return keys


def _call_gpt(prompt: Union[str, List[Dict[str, str]]], api_key: str, base_url: str | None, model: str, attempts: int = 4) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    for i in range(attempts):
        try:
            # Accept either a raw string or a list of message dicts
            if isinstance(prompt, list):
                messages = prompt
            else:
                messages = [{"role": "user", "content": prompt}]
            resp = client.chat.completions.create(model=model, messages=messages, max_tokens=256)
            # resp.choices may be list-like depending on client
            text = None
            if getattr(resp, "choices", None):
                text = resp.choices[0].message.content
            elif getattr(resp, "output", None):
                text = resp.output[0].content[0].text
            else:
                text = str(resp)
            return text.strip()
        except Exception as e:
            # simple retry on transient errors
            if i + 1 == attempts:
                raise
            backoff = min(60, 2 ** i)
            time.sleep(backoff)
    raise RuntimeError("unreachable")


def worker(item, keys: List[str], base_url: str, model: str, perturbation_type: str, all_styles: bool, debug: bool):
    """Generate one or more perturbed captions for a single item.

    Returns a copy of the item with 'perturbed_caption' (or 'perturbed_1..3' when all_styles).
    """
    caption = item['caption']
    out = dict(item)
    if not caption:
        # nothing to do
        if all_styles:
            out.update({f"perturbed_{i+1}": None for i in range(3)})
        else:
            out["perturbed_caption"] = None
        return out

    # Default retries: if model returns same caption as original, retry a few times
    max_retries = item.get("_max_retries", None) if isinstance(item, dict) else None
    # We'll override via closure in main by passing a numeric value in item['_max_retries'] if needed
    messages = build_prompt(caption, perturbation_type)
    if debug:
        print("=== Prompt ===")
        for msg in messages:
            print(f"{msg['role'].upper()}: {msg['content']}")
        print("==============")
    # messages is a single message sequence (list of role/content dicts)
    sequences = [messages]
    if all_styles:
        # produce multiple variants by calling the same sequence multiple times
        sequences = [messages for _ in range(3)]

    results = []
    for idx, seq in enumerate(sequences):
        key = keys[idx % len(keys)]  # simple distribution; caller uses concurrency
        text: Optional[str] = None
        # Outer retry loop: regenerate if result equals original caption
        retries = 0
        max_r = 3
        # allow overriding per-item via special key
        try:
            if isinstance(item, dict) and item.get("_max_retries") is not None:
                max_r = int(item.get("_max_retries"))
        except Exception:
            pass

        while retries < max_r:
            try:
                text = _call_gpt(seq, key, base_url, model)
            except Exception:
                text = None
            if not text:
                # if the API failed to produce text, break and treat as failure
                break
            # normalize and compare
            tnorm = text.strip()
            onorm = caption.strip()
            if tnorm == onorm:
                # identical, retry
                retries += 1
                if debug:
                    print(f"Retrying because output equals input (attempt {retries}/{max_r})")
                # small backoff
                time.sleep(min(2 ** retries, 8))
                continue
            # different -> accept
            break
        results.append(text)

    if all_styles:
        # If any generated caption is None or identical to original even after retries,
        # treat the whole item as failed (do not save) and let the caller record an error.
        for i, r in enumerate(results):
            out[f"perturbed_{i+1}"] = r
        return out
    else:
        out["perturbed_caption"] = results[0]
        return out


def parse_args():
    import argparse

    p = argparse.ArgumentParser("Perturb captions via GPT")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--N", type=int, default=None, help="Number of captions to process (default: all)")
    p.add_argument("--model", help="Chat model to call")
    p.add_argument("--base_url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    p.add_argument("--api_keys", default=None)
    p.add_argument("--api_keys_file", default=None)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--all_styles", action="store_true", help="Produce all three perturbed variants")
    p.add_argument("--perturbation_type", type=str, required=True,
                   choices=["style_view_perturbation", "semantic_view_perturbation", "token_view_perturbation"],
                   help="Type of perturbation to apply")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--max_retries", type=int, default=3,
                   help="Number of times to retry if generated caption equals original (default: 3)")
    return p.parse_args()


def main():
    """CLI entry to perturb captions in a JSON list using a chat API."""
    args = parse_args()
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.N is not None:
        data = data[: args.N]
    if not isinstance(data, list):
        raise RuntimeError("Input JSON must be a list of objects containing 'caption'")

    keys = _load_api_keys(args)

    # We'll collect errors in-memory (thread-safe) and write them out at the end to error.json
    errors: List[Dict] = []
    errors_lock = threading.Lock()

    def worker_wrapper(idx, item):
        # attach max_retries to item for worker to read if desired
        if args.max_retries is not None:
            try:
                item["_max_retries"] = int(args.max_retries)
            except Exception:
                item["_max_retries"] = 3
        out = worker(item, keys, args.base_url, args.model, args.perturbation_type, args.all_styles, args.debug)

        # Determine failure conditions:
        if out is None:
            # worker signalled to drop (not used currently)
            with errors_lock:
                errors.append({
                    "index": idx,
                    "caption": item.get("caption"),
                    "perturbation_type": args.perturbation_type,
                    "reason": "worker_returned_none"
                })
            return None

        # If not all_styles, drop the item if perturbed_caption is None or equals original
        if not args.all_styles:
            p = out.get("perturbed_caption")
            if p is None or p.strip() == item.get("caption", "").strip():
                with errors_lock:
                    errors.append({
                        "index": idx,
                        "caption": item.get("caption"),
                        "perturbation_type": args.perturbation_type,
                        "reason": "no_change_after_retries",
                        "perturbed": p,
                    })
                return None
        else:
            # all_styles: if any of perturbed_1..3 is None or equals original, drop entire item
            bad = False
            for i in range(1, 4):
                p = out.get(f"perturbed_{i}")
                if p is None or p.strip() == item.get("caption", "").strip():
                    bad = True
                    break
            if bad:
                with errors_lock:
                    errors.append({
                        "index": idx,
                        "caption": item.get("caption"),
                        "perturbation_type": args.perturbation_type,
                        "reason": "no_change_after_retries",
                        "perturbed": {f"perturbed_{i}": out.get(f"perturbed_{i}") for i in range(1,4)}
                    })
                return None

        return out

    results = []
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = [ex.submit(worker_wrapper, idx, item) for idx, item in enumerate(data)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Perturbing captions"):
            results.append(f.result())

    # Filter out None results (failed/discarded)
    final_results = [r for r in results if r is not None]

    # ensure output dir exists
    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    if errors:
        err_path = os.path.join(out_dir, "error.json")
        # write the errors as a JSON array
        with open(err_path, "w", encoding="utf-8") as ef:
            json.dump(errors, ef, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
