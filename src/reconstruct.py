import argparse
import os
import torch
import torch.utils.checkpoint
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline
from PIL import Image
import json
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm

from diffusers.pipelines.stable_diffusion import safety_checker
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple
import time

def sc(self, clip_input, images) : return images, [False for i in images]

safety_checker.StableDiffusionSafetyChecker.forward = sc
def parse_args():
    """Parse CLI args for local Stable Diffusion image generation."""
    parser = argparse.ArgumentParser(description="Local Stable Diffusion image generation script")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    ) 
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=3
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of prompts to process simultaneously. Uses single-image inference when set to 1.",
    )
    parser.add_argument("--inference", type=int, default=100)
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Output image width (will be snapped to a multiple of 8).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output image height (will be snapped to a multiple of 8).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducible generation. If provided, each sample uses seed+index.",
    )
    parser.add_argument(
        "--add_original_image_for_generation",
        action="store_true",
        help="Whether to add the original image as a condition for generation.",
    )
    parser.add_argument(
        "--img2img_strength",
        type=float,
        default=0.8,
        help="Strength for img2img when using original images (0.0 preserve, 1.0 ignore).",
    )
    parser.add_argument(
        "--use_perturbed_caption",
        action="store_true",
        help="Whether to use the perturbed caption for generation.",
    )
    parser.add_argument(
        "--perturb_type",
        type=str,
        choices=["synonym_replacement", "random_deletion", "style_insertion"],
        help="Type of perturbation to apply to captions if --use_perturbed_caption is set.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume generation by skipping jobs whose output images already exist and pass integrity checks.",
    )
    args = parser.parse_args()

    return args


def main():
    """Run batched text/image-to-image generation with optional resume and perturbations."""
    args = parse_args()

    # Validate and normalize output resolution (SD requires multiples of 8)
    def _snap_dim(x: int) -> int:
        x = int(x)
        if x < 64:
            x = 64
        # round to nearest multiple of 8
        snapped = max(64, int(round(x / 8)) * 8)
        return snapped

    out_w = _snap_dim(args.width)
    out_h = _snap_dim(args.height)

    # Create timestamped save directory and persist args
    if args.save_dir is None:
        raise ValueError("--save_dir must be provided")
    final_save_dir = args.save_dir
    os.makedirs(final_save_dir, exist_ok=True)

    # Save args for reproducibility
    try:
        with open(os.path.join(final_save_dir, "inference_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    except Exception as e:
        print(f"Warning: failed to save args.json due to: {e}")

    # Choose pipeline(s). If multiple GPUs are available, instantiate one pipeline per GPU
    # and split each batch across devices. CLI and outputs remain unchanged.
    n_gpus = torch.cuda.device_count()
    use_multi_gpu = n_gpus > 1

    def create_pipeline() :
        if args.add_original_image_for_generation:
            if "3" in args.pretrained_model_name_or_path:
                return StableDiffusion3Img2ImgPipeline.from_pretrained(
                    args.pretrained_model_name_or_path, torch_dtype=torch.float16, safety_checker=None,
                )
            else:
                return StableDiffusionImg2ImgPipeline.from_pretrained(
                    args.pretrained_model_name_or_path, torch_dtype=torch.float16, safety_checker=None,
                )
        else:
            if "3" in args.pretrained_model_name_or_path:
                return StableDiffusion3Pipeline.from_pretrained(
                    args.pretrained_model_name_or_path, torch_dtype=torch.float16, safety_checker=None,
                )
            else:
                return StableDiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path, torch_dtype=torch.float16, safety_checker=None,
                )

    if use_multi_gpu:
        pipelines = []
        for i in range(n_gpus):
            p = create_pipeline()
            p.to(f"cuda:{i}")
            pipelines.append(p)
    else:
        pipeline = create_pipeline()
        pipeline.to("cuda:0")


    if args.data_dir.endswith(".json"):
        with open(args.data_dir, "r") as f:
            dataset = json.load(f)
    elif args.data_dir.endswith(".jsonl"):
        with open(args.data_dir, "r") as f:
            dataset = [json.loads(line) for line in f]

    # Build a flat job list to preserve exact filename semantics while enabling batching.
    # Each job must carry both original and perturbed info for paired generation when enabled.
    # Job keys: caption, perturbed_caption, save_path, perturb_save_path, image_path
    jobs = []
    meta_file = []
    job_index_counter = 0  # global ordering for reproducible seeding across resume runs
    for data in dataset:
        original_filename = Path(data["path"]).name
        name_parts = original_filename.split(".")
        base, ext = name_parts[0], name_parts[1]
        for j in range(args.num_validation_images):
            filename = f"{data['label']}_{base}_{j+1:02}.{ext}"

            save_path = os.path.join(final_save_dir, "images", filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            perturb_save_path = None
            perturbed_caption = None

            if args.use_perturbed_caption:
                if args.perturb_type is None:
                    raise ValueError("If --use_perturbed_caption is set, --perturb_type must be provided.")
                perturb_save_path = os.path.join(final_save_dir, "images_perturbed_caption", filename)
                os.makedirs(os.path.dirname(perturb_save_path), exist_ok=True)

                # Try to locate perturbed caption from a list under 'perturbations'
                if isinstance(data.get("perturbations"), list):
                    for perturbation in data["perturbations"]:
                        if perturbation.get("method") == args.perturb_type:
                            perturbed_caption = perturbation.get("perturbed_caption")
                            break
                # Fallback: allow direct key (e.g. data['synonym_replacement']) if present
                if perturbed_caption is None and isinstance(data.get(args.perturb_type), str):
                    perturbed_caption = data[args.perturb_type]

                if perturbed_caption is None:
                    print(
                        f"Warning: no perturbed caption found for method '{args.perturb_type}' in item {original_filename}. "
                        "Perturbed image generation for this item will be skipped."
                    )

            jobs.append({
                "caption": data["caption"],
                "perturbed_caption": perturbed_caption,
                "save_path": save_path,
                "perturb_save_path": perturb_save_path,
                "image_path": data.get("path", None),
                "index": job_index_counter,
            })
            job_index_counter += 1
            meta_file.append({
                "caption": data["caption"],
                "label": data["label"],
                "path": save_path,
                "repeat": j+1,
                "orig_path": data['path'],
            })
            if args.use_perturbed_caption:
                meta_file.append({
                    "caption": perturbed_caption,
                    "label": data["label"],
                    "path": perturb_save_path,
                    "repeat": j+1,
                    "orig_path": data['path'],
                })

    # Integrity / resume logic -------------------------------------------------
    def _check_image_integrity(path: str, expected_w: int, expected_h: int) -> bool:
        """Return True if existing image at path can be opened & matches expected size.
        Conservative: any exception or size mismatch => False (needs regeneration).
        """
        if not os.path.isfile(path):
            return False
        try:
            with Image.open(path) as im:
                im.verify()  # verify header
            # reopen to access size after verify (which invalidates the core image object)
            with Image.open(path) as im2:
                if im2.size != (expected_w, expected_h):
                    return False
        except Exception:
            return False
        return True

    if args.resume:
        skipped_original = 0
        skipped_perturbed = 0
        for job in jobs:
            # Original image check
            if _check_image_integrity(job["save_path"], out_w, out_h):
                job["skip_original"] = True
                skipped_original += 1
            else:
                job["skip_original"] = False
            # Perturbed image check (only if feature enabled and caption exists)
            if args.use_perturbed_caption and job.get("perturbed_caption") and job.get("perturb_save_path"):
                if _check_image_integrity(job["perturb_save_path"], out_w, out_h):
                    job["skip_perturbed"] = True
                    skipped_perturbed += 1
                else:
                    job["skip_perturbed"] = False
            else:
                job["skip_perturbed"] = True  # treat as non-existent perturbed generation
        print(f"Resume mode: {skipped_original} original images and {skipped_perturbed} perturbed images already complete and will be skipped.")
    else:
        for job in jobs:
            job["skip_original"] = False
            job["skip_perturbed"] = False if (args.use_perturbed_caption and job.get("perturbed_caption") and job.get("perturb_save_path")) else True

    # Optional: set up per-sample generators for reproducibility if seed is provided
    def make_generators(n, base_seed):
        if base_seed is None:
            return None
        gens = []
        for idx in range(n):
            g = torch.Generator(device="cuda")
            g.manual_seed(int(base_seed) + idx)
            gens.append(g)
        return gens

    bs = max(1, int(args.batch_size))

    # Helpers for multi-GPU batching
    def split_even(lst: List, k: int) -> List[List]:
        """Split list into k nearly-even chunks (dropping empty tails if k>len(lst))."""
        k = max(1, min(k, len(lst)))
        base = len(lst) // k
        rem = len(lst) % k
        chunks = []
        start = 0
        for i in range(k):
            size = base + (1 if i < rem else 0)
            if size == 0:
                continue
            chunks.append(lst[start:start+size])
            start += size
        return chunks

    def make_generators_for_items(items_len: int, base_seed: Optional[int], device: str, start_index: int) -> Optional[List[torch.Generator]]:
        if base_seed is None:
            return None
        gens: List[torch.Generator] = []
        for local_idx in range(items_len):
            g = torch.Generator(device=device)
            # Global index = start_index + local_idx
            g.manual_seed(int(base_seed) + start_index + local_idx)
            gens.append(g)
        return gens

    def run_on_device(p, device: str, prompts: List[str], init_images: Optional[List[Image.Image]],
                      num_steps: int, guidance_scale: float, strength: Optional[float],
                      generators: Optional[List[torch.Generator]], width: int, height: int):
        with torch.inference_mode():
            if init_images is not None:
                out = p(
                    prompts,
                    image=init_images,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    strength=float(strength) if strength is not None else None,
                    generator=generators,
                    width=width,
                    height=height,
                )
            else:
                out = p(
                    prompts,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    generator=generators,
                    width=width,
                    height=height,
                )
        return out.images
    # Recompute total images to generate (exclude those skipped in resume mode)
    perturbed_count = sum(1 for j in jobs if (args.use_perturbed_caption and j.get("perturbed_caption") and j.get("perturb_save_path") and not j.get("skip_perturbed")))
    original_count = sum(1 for j in jobs if not j.get("skip_original"))
    total_images = original_count + perturbed_count
    with tqdm(total=total_images, desc="Generating images", unit="img") as pbar:
        for job_start in range(0, len(jobs), bs):
            job_end = min(job_start + bs, len(jobs))
            batch = jobs[job_start:job_end]

            # Pass 1: original caption -> save_path (filter skipped)
            original_items = [item for item in batch if not item.get("skip_original")]
            prompts = [item["caption"] for item in original_items]
            if use_multi_gpu:
                # Split only the items we need to process across devices
                sub_batches = split_even(original_items, n_gpus)
                futures = []
                results_order: List[Tuple[List[Image.Image], List[dict]]] = []  # (images, items)

                with ThreadPoolExecutor(max_workers=len(sub_batches) if len(sub_batches) > 0 else 1) as ex:
                    if len(sub_batches) == 0:
                        pass  # nothing to do
                    for dev_idx, sub in enumerate(sub_batches):
                        device = f"cuda:{dev_idx}"
                        p = pipelines[dev_idx]
                        sub_prompts = [it["caption"] for it in sub]
                        if args.add_original_image_for_generation:
                            sub_inits = []
                            for it in sub:
                                if not it.get("image_path"):
                                    raise ValueError("Dataset item missing 'path' required for image+text generation.")
                                img = Image.open(it["image_path"]).convert("RGB")
                                img = img.resize((out_w, out_h))
                                sub_inits.append(img)
                        else:
                            sub_inits = None

                        # Base seed for the whole run is args.seed; we incorporate absolute index for each item
                        # Generators seeded by global job index for reproducibility across resume runs
                        if args.seed is not None:
                            gens = []
                            for it in sub:
                                g = torch.Generator(device=device)
                                g.manual_seed(int(args.seed) + int(it["index"]))
                                gens.append(g)
                        else:
                            gens = None

                        futures.append(ex.submit(
                            run_on_device, p, device, sub_prompts, sub_inits,
                            args.inference, 7.5, args.img2img_strength if args.add_original_image_for_generation else None,
                            gens, out_w, out_h
                        ))

                    # Collect results preserving sub-batch order
                    for sub, fut in zip(sub_batches, futures):
                        imgs = fut.result()
                        results_order.append((imgs, sub))

                # Save in the original batch order
                for imgs, items in results_order:
                    for img, item in zip(imgs, items):
                        img.save(item["save_path"])
                        pbar.update(1)
            else:
                if len(original_items) > 0:
                    generators = None
                    if args.seed is not None:
                        generators = []
                        for it in original_items:
                            g = torch.Generator(device="cuda:0")
                            g.manual_seed(int(args.seed) + int(it["index"]))
                            generators.append(g)

                    if args.add_original_image_for_generation:
                        init_images = []
                        for item in original_items:
                            if not item.get("image_path"):
                                raise ValueError("Dataset item missing 'path' required for image+text generation.")
                            img = Image.open(item["image_path"]).convert("RGB")
                            img = img.resize((out_w, out_h))
                            init_images.append(img)
                        images = run_on_device(
                            pipeline, "cuda:0", prompts, init_images,
                            args.inference, 7.5, args.img2img_strength, generators, out_w, out_h
                        )
                    else:
                        images = run_on_device(
                            pipeline, "cuda:0", prompts, None,
                            args.inference, 7.5, None, generators, out_w, out_h
                        )

                    for img, item in zip(images, original_items):
                        img.save(item["save_path"])
                        pbar.update(1)

            # Pass 2: perturbed caption -> perturb_save_path (only if enabled)
            if args.use_perturbed_caption:
                # Filter out items missing a perturbed caption or save path and those already skipped in resume
                perturbed_batch = [it for it in batch if it.get("perturbed_caption") and it.get("perturb_save_path") and not it.get("skip_perturbed")]
                if len(perturbed_batch) > 0:
                    if use_multi_gpu:
                        sub_batches = split_even(perturbed_batch, n_gpus)
                        futures = []
                        results_order: List[Tuple[List[Image.Image], List[dict]]] = []
                        with ThreadPoolExecutor(max_workers=len(sub_batches)) as ex:
                            for dev_idx, sub in enumerate(sub_batches):
                                device = f"cuda:{dev_idx}"
                                p = pipelines[dev_idx]
                                sub_prompts = [it["perturbed_caption"] for it in sub]
                                if args.add_original_image_for_generation:
                                    sub_inits = []
                                    for it in sub:
                                        if not it.get("image_path"):
                                            raise ValueError("Dataset item missing 'path' required for image+text generation.")
                                        img = Image.open(it["image_path"]).convert("RGB")
                                        img = img.resize((out_w, out_h))
                                        sub_inits.append(img)
                                else:
                                    sub_inits = None

                                if args.seed is not None:
                                    gens = []
                                    for it in sub:
                                        g = torch.Generator(device=device)
                                        g.manual_seed(int(args.seed) + int(it["index"]))
                                        gens.append(g)
                                else:
                                    gens = None

                                futures.append(ex.submit(
                                    run_on_device, p, device, sub_prompts, sub_inits,
                                    args.inference, 7.5, args.img2img_strength if args.add_original_image_for_generation else None,
                                    gens, out_w, out_h
                                ))

                            for sub, fut in zip(sub_batches, futures):
                                imgs = fut.result()
                                results_order.append((imgs, sub))

                        for imgs, items in results_order:
                            for img, it in zip(imgs, items):
                                img.save(it["perturb_save_path"])
                                pbar.update(1)
                    else:
                        perturbed_prompts = [it["perturbed_caption"] for it in perturbed_batch]
                        generators2 = None
                        if args.seed is not None:
                            generators2 = []
                            for it in perturbed_batch:
                                g = torch.Generator(device="cuda:0")
                                g.manual_seed(int(args.seed) + int(it["index"]))
                                generators2.append(g)

                        if args.add_original_image_for_generation:
                            init_images2 = []
                            for it in perturbed_batch:
                                if not it.get("image_path"):
                                    raise ValueError("Dataset item missing 'path' required for image+text generation.")
                                img = Image.open(it["image_path"]).convert("RGB")
                                img = img.resize((out_w, out_h))
                                init_images2.append(img)
                            images2 = run_on_device(
                                pipeline, "cuda:0", perturbed_prompts, init_images2,
                                args.inference, 7.5, args.img2img_strength, generators2, out_w, out_h
                            )
                        else:
                            images2 = run_on_device(
                                pipeline, "cuda:0", perturbed_prompts, None,
                                args.inference, 7.5, None, generators2, out_w, out_h
                            )

                        for img, it in zip(images2, perturbed_batch):
                            img.save(it["perturb_save_path"])
                            pbar.update(1)

    # Save jobs at the end for traceability
    try:
        with open(os.path.join(final_save_dir, "jobs.json"), "w", encoding="utf-8") as f:
            json.dump(jobs, f, ensure_ascii=False, indent=2)
        with open(os.path.join(final_save_dir, "metafile.json"), "w", encoding="utf-8") as f:
            json.dump(meta_file, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: failed to save jobs.json due to: {e}")

if __name__ == "__main__":
    main()
