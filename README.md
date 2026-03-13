# SD-MIA
Official repo for Black-box Membership Inference Attacks on thePre-training Data of Image-generation Models

<img src="img/method.pdf" alt="SD-MIA Method"/>

## Method Overview
- Apply three perturbation views to the original caption.
- Reconstruct the images.
- Evaluate cross-modal relevance with original images.
- Pool top K% of relevance scores.

## Environment
- Python 3.10.16 recommended.
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Step 1: Apply three perturbation views to the original caption
Use `src/perturb_captions_gpt.py` to produce the three perturbation versions. 

```bash
# Examples
python src/perturb_captions_gpt.py \
  --input <path-to-your-original-caption> \
  --output <output-dir> \
  --perturbation_type <token_view_perturbation/style_view_perturbation/semantic_view_perturbation>
```

## Step 2: Reconstruct the images, and evaluate cross-modal relevance with original images
Edit `scripts/pipeline.sh` with your paths and run it.


## Step 3: Pool top K% of relevance scores
Use `process.ipynb`.