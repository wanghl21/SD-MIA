import os
import base64
import re
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from openai import OpenAI
from PIL import Image
from io import BytesIO
from tqdm import tqdm


def _get_client(base_url: Optional[str] = None, api_key: Optional[str] = None) -> OpenAI:
    """Create an OpenAI-compatible client.

    Priority order for base_url:
      1) Explicit argument
      2) Env OPENAI_BASE_URL
      3) Default https://api.openai.com/v1

    Priority order for api_key:
      1) Explicit argument
      2) Env OPENAI_API_KEY
    """
    resolved_base_url = base_url or os.getenv(
        "OPENAI_BASE_URL") or "https://api.openai.com/v1"
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError(
            "Missing API key. Set OPENAI_API_KEY env or pass --api-key explicitly."
        )
    return OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)


def _image_from_response_item(item) -> Image.Image:
    """Build a PIL Image from either URL or base64 response formats."""
    if getattr(item, "url", None):
        resp = requests.get(item.url, timeout=60)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content))
    if getattr(item, "b64_json", None):
        img_bytes = base64.b64decode(item.b64_json)
        return Image.open(BytesIO(img_bytes))
    raise ValueError(
        "Unsupported image response format (expected url or b64_json)")


def _gemini_generate_image(
    prompt: str,
    model: str,
    size: str = "1024x1024",
    quality: str = "standard",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Image.Image:
    """Generate image via Google Gemini API (native), returning a PIL Image.

    Notes:
      - Requires a Gemini API key (env GEMINI_API_KEY or provided api_key).
      - Endpoint: {base_url}/models/{model}:generateContent
      - Expects image bytes in candidates[0].content.parts[*].inline_data.data (base64)
    """
    gemini_base = base_url or os.getenv(
        "GEMINI_API_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta"
    gemini_key = api_key or os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        # Allow fallback to OPENAI_API_KEY only if explicitly provided via arg
        gemini_key = api_key
    if not gemini_key:
        raise RuntimeError(
            "Missing Gemini API key. Set GEMINI_API_KEY or pass --api_key for Gemini.")

    url = f"{gemini_base.rstrip('/')}/models/{model}:generateContent?key={gemini_key}"

    # Map size "WxH" to optional config. The public beta API may not support explicit size.
    # We'll set response_mime_type and let server choose size; advanced control can be added later.
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": str(prompt)}],
            }
        ],
        "generationConfig": {
            "response_mime_type": "image/png",
        },
    }

    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        # Raise with a clear message so callers can parse status codes for retry/policy handling
        raise RuntimeError(f"Error code: {resp.status_code} - {resp.text}")

    data = resp.json()
    # Navigate to inline_data
    try:
        candidates = data.get("candidates", [])
        if not candidates:
            raise KeyError("candidates")
        parts = candidates[0].get("content", {}).get("parts", [])
        # Find first part with inline_data image
        b64 = None
        for part in parts:
            inline = part.get("inline_data") or part.get("inlineData")
            if inline and inline.get("data"):
                b64 = inline["data"]
                break
        if not b64:
            # If text-only response, surface a structured error that upstream can treat as content policy / refusal
            text_snippet = None
            for part in parts:
                if part.get("text"):
                    text_snippet = part.get("text")
                    break
            if text_snippet:
                text_snippet = text_snippet.strip().replace("\n", " ")
                if len(text_snippet) > 180:
                    text_snippet = text_snippet[:180] + "..."
            # Raise as a 500-like content policy to match existing retry/log logic
            raise RuntimeError(
                f"Error code: 500 - {{'error': {{'message': 'Gemini no image returned: {text_snippet}', 'code': 'content_policy_violation'}}}}"
            )
        img_bytes = base64.b64decode(b64)
        return Image.open(BytesIO(img_bytes))
    except RuntimeError:
        # Re-raise structured runtime errors
        raise
    except Exception as e:
        raise RuntimeError(f"Gemini response parse error: {e}; raw={data}")


def generate_image(
    prompt: str,
    model: str,
    size: str = "1024x1024",
    quality: str = "standard",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
):
    """Text-to-image via an OpenAI-compatible API or native Gemini.

    Args:
        prompt: The text prompt.
        model: Model name (e.g., gpt-4o, dall-e-3).
        size: Image size in "WxH" format (e.g., 1024x1024).
        quality: Image quality flag, typically "standard" or "hd" depending on provider.
        base_url: OpenAI-compatible base URL (official or third-party aggregator).
        api_key: API key; defaults to env var OPENAI_API_KEY if not provided.
        provider: Optional explicit provider hint (e.g., "gemini").

    Returns:
        PIL.Image.Image: The generated image.
    """
    # Dispatch: Gemini native vs OpenAI-compatible
    use_gemini = False
    if provider and provider.lower() == "gemini":
        use_gemini = True
    elif (model or "").lower().startswith("gemini"):
        # If a clear OpenAI-compatible base_url is provided, honor that path; else use native Gemini
        if not base_url or ("generativelanguage" in str(base_url)):
            use_gemini = True

    if use_gemini:
        prompt = "Please generate an image based on the following description: " + prompt
        return _gemini_generate_image(
            prompt=prompt,
            model=model,
            size=size,
            quality=quality,
            base_url=base_url,
            api_key=api_key,
        )
    else:
        client = _get_client(base_url, api_key)
        resp = client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=1,
        )
        return _image_from_response_item(resp.data[0])


def _safe_filename(text: str, max_len: int = 60) -> str:
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^a-zA-Z0-9._-]", "", text)
    if len(text) > max_len:
        text = text[:max_len]
    return text or "image"


def batch_generate_images(
    prompts: List[str],
    model: str,
    size: str = "1024x1024",
    quality: str = "standard",
    n_per_prompt: int = 1,
    threads: int = 1,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    output_dir: Optional[str] = None,
    show: bool = False,
) -> List[Tuple[str, Image.Image]]:
    """Batch-generate images with optional multithreading and progress bar.

    Returns: List of tuples (filepath_or_name, PIL.Image)
    """
    os.makedirs(output_dir, exist_ok=True) if output_dir else None

    tasks = []
    for idx, p in enumerate(prompts):
        for k in range(n_per_prompt):
            tasks.append((idx, k, p))

    results: List[Tuple[str, Image.Image]] = []

    def _worker(idx: int, k: int, p: str) -> Tuple[str, Image.Image]:
        img = generate_image(
            prompt=p,
            model=model,
            size=size,
            quality=quality,
            base_url=base_url,
            api_key=api_key,
        )
        name = f"{idx:04d}_{k:02d}_" + _safe_filename(p[:40]) + ".png"
        if output_dir:
            path = os.path.join(output_dir, name)
            img.save(path)
            return path, img
        return name, img

    max_workers = max(1, int(threads or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_worker, i, k, p) for (i, k, p) in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            try:
                results.append(f.result())
                if show and results[-1][1] is not None:
                    # show() is blocking in some environments; call non-blocking where possible
                    results[-1][1].show()
            except Exception as e:
                # Keep going on individual failures
                tqdm.write(f"Error: {e}")

    return results


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified text-to-image generator")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model name",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", type=str, help="Single prompt text")
    group.add_argument(
        "--prompts-file",
        type=str,
        help="Path to a text file with one prompt per line",
    )
    parser.add_argument("--n", type=int, default=1, help="Images per prompt")
    parser.add_argument("--size", type=str,
                        default="1024x1024", help="Image size")
    parser.add_argument("--quality", type=str,
                        default="standard", help="Image quality: standard|hd")
    parser.add_argument("--threads", type=int, default=1,
                        help="Number of concurrent threads")
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible base URL (env OPENAI_BASE_URL)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="API key (env OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save images (created if missing)",
    )
    parser.add_argument("--show", action="store_true",
                        help="Show images as they complete")
    return parser.parse_args()


def _load_prompts(args) -> List[str]:
    if args.prompt is not None:
        return [args.prompt]
    # prompts-file
    with open(args.prompts_file, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]


if __name__ == "__main__":
    args = _parse_args()
    prompts = _load_prompts(args)

    # Run generation
    batch_generate_images(
        prompts=prompts,
        model=args.model,
        size=args.size,
        quality=args.quality,
        n_per_prompt=args.n,
        threads=args.threads,
        base_url=args.base_url,
        api_key=args.api_key,
        output_dir=args.output_dir,
        show=args.show,
    )
