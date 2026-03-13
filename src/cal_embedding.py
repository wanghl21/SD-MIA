from transformers import (
    DeiTFeatureExtractor,
    DeiTModel,
    DeformableDetrModel,
    AutoImageProcessor,
    BeitModel,
    EfficientFormerModel,
    ViTModel,
)
try:
    # Optional import; only needed when using caption conditioning
    from transformers import CLIPProcessor, CLIPModel
    _HAS_CLIP = True
except Exception:
    _HAS_CLIP = False
import torch
from PIL import Image
import os
import argparse
from tqdm import tqdm
import json
from PIL import Image, UnidentifiedImageError
import time

def parse_args():
    """Parse CLI arguments for embedding extraction.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Extract image embeddings with optional caption conditioning")
    parser.add_argument("--data_dir",type=str,default=None, help="Input JSON file listing images (path, caption, label)")
    parser.add_argument("--output_dir", type=str, help="Directory to write embeddings JSON and meta file")
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--image_encoder",type=str,default="clip")
    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP model id to use",
    )
    args = parser.parse_args()

    return args

def main():
    """Read dataset JSON, encode each image, and write embeddings.

    For CLIP-based conditioning, combines image and text features according. Outputs a single `embeddings.json` and a `meta.json`.
    """
    with open(args.data_dir, "r") as f:
        dataset = json.load(f)

    # Initialize encoders
    clip_processor = None
    if not _HAS_CLIP:
        raise ImportError("transformers.CLIPModel is required for image_encoder=clip")
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
    model = CLIPModel.from_pretrained(args.clip_model)
    feature_extractor = None  # handled by CLIPProcessor
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    embeddings_out = []
    # Helpers
    def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8):
        """L2-normalize tensor along a dimension with numerical stability."""
        return x / (x.norm(dim=dim, keepdim=True) + eps)

    def _clip_image_features(img: Image.Image, caption_text: str | None):
        """Compute (optionally) caption-conditioned CLIP image embedding.
        Returns a (1, D) tensor on device.
        """
        assert clip_processor is not None and isinstance(model, CLIPModel)
        # Prepare image batch
        img_inputs = clip_processor(images=img, return_tensors="pt")
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}

        # Prepare text
        text_inputs = clip_processor(text=[caption_text], return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        # Concatenate normalized image and text embeddings in the same joint space
        with torch.no_grad():
            img_feats = model.get_image_features(**img_inputs)  # (1, D)
            txt_feats = model.get_text_features(**text_inputs)  # (1, D)
        img_feats = _l2norm(img_feats)
        txt_feats = _l2norm(txt_feats)
        cat = torch.cat([img_feats, txt_feats], dim=-1)  # (1, 2D)
        return _l2norm(cat)

    for item in tqdm(dataset):
        image_path = item.get("path")
        caption_text = item.get("caption", None)
        label = item.get("label", None)
        if image_path is None:
            print("[skip] item missing 'path'")
            continue
        try:
            img = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, PermissionError, UnidentifiedImageError, OSError) as e:
            print(f"[skip] Cannot open image: {image_path} -> {e}")
            continue
        img = img.resize((512, 512))

        if args.image_encoder == "clip":
            emb_t = _clip_image_features(img, caption_text)
        else:
            inputs = feature_extractor(img, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            # outputs.last_hidden_state shape: (1, S, H) or pooled; pool to vector
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb_t = outputs.pooler_output
            else:
                emb_t = outputs.last_hidden_state.mean(dim=1)

        # ensure 1D numpy list
        emb_np = emb_t.cpu().numpy().reshape(-1).tolist()

        embeddings_out.append(item)
        embeddings_out[-1]["embedding"] = emb_np
        embeddings_out[-1]["image_encoder"] = args.image_encoder
    # write embeddings JSON
    os.makedirs(args.output_dir, exist_ok=True)
    output_embeddings_path = os.path.join(args.output_dir, "embeddings.json")
    with open(output_embeddings_path, 'w') as f:
        json.dump(embeddings_out, f, indent=4)

    # write meta file with args and embedding path
    meta = {
        "args": vars(args),
        "embeddings_path": output_embeddings_path,
    }
    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    main()
