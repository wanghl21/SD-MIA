import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import argparse
import os
import json
from tqdm import tqdm
from typing import List

def parse_args():
    """Parse CLI arguments for BLIP-2 caption generation."""
    parser = argparse.ArgumentParser(description="Generate caption for an image using BLIP-2 model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input json or jsonl.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output captions (json or jsonl).")
    parser.add_argument("--blip_model_name", type=str, default="Salesforce/blip2-opt-6.7b", help="Pretrained BLIP-2 model name.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing images.")
    return parser.parse_args()

def main():
    """Load images from a JSON/JSONL list and write BLIP-2 captions."""
    args = parse_args()

    processor = Blip2Processor.from_pretrained(args.blip_model_name)
    # Load model (keep CLI unchanged). If running on CUDA, prefer float16; otherwise fall back to default dtype.

    model = Blip2ForConditionalGeneration.from_pretrained(
        args.blip_model_name, torch_dtype=torch.float16,
    )
    model.to(args.device)


    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    save_data = []
    if args.data_path.endswith('.jsonl'):
        with open(args.data_path, 'r') as f_in:
            input_data = [json.loads(line) for line in f_in]
    elif args.data_path.endswith('.json'):
        with open(args.data_path, 'r') as f_in:
            input_data = json.load(f_in)
    
    # Determine batch size without changing CLI (env var override allowed)
    batch_size = args.batch_size

    def chunk_indices(n: int, bs: int) -> List[range]:
        return [range(i, min(i + bs, n)) for i in range(0, n, bs)]

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    results = [None] * len(input_data)
    for batch_idx in tqdm(chunk_indices(len(input_data), batch_size)):
        # Load a batch of images
        images = []
        for i in batch_idx:
            img_url = input_data[i]['path']
            img = Image.open(img_url).convert('RGB')
            images.append(img)

        # Preprocess as a batch
        inputs = processor(images, return_tensors="pt").to(model.device)
        # Use float16 on CUDA if possible
        if device.startswith("cuda"):
            inputs = {k: v.to(dtype=torch.float16) if isinstance(v, torch.Tensor) and v.dtype.is_floating_point else v for k, v in inputs.items()}

        with torch.inference_mode():
            out = model.generate(**inputs)

        # Decode captions per item in batch
        for offset, seq in enumerate(out):
            caption = processor.decode(seq, skip_special_tokens=True).strip()
            idx = batch_idx.start + offset
            item = dict(input_data[idx])  # copy to avoid side-effects
            item['caption'] = caption
            results[idx] = item

        # Proactively close PIL images
        for img in images:
            try:
                img.close()
            except Exception:
                pass

    # Preserve original write logic and ordering
    save_data = results
    with open(args.output_path, 'w') as f_out:
        if args.output_path.endswith('.jsonl'):
            for item in save_data:
                f_out.write(json.dumps(item) + '\n')
        elif args.output_path.endswith('.json'):
            json.dump(save_data, f_out, indent=4)

if __name__ == "__main__":
    main()
