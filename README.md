# DRAPE AI

AI-powered fashion model generator. Takes a flat-lay garment image and produces a photorealistic fashion photograph of a model wearing it — no GPU required.

## How It Works

Three-stage pipeline, all running on free HuggingFace APIs:

1. **Base Model Generation** — generates a full-body fashion model image from text prompts (FLUX.1-dev → FLUX.1-schnell → SDXL, with fallbacks)
2. **Image Preparation** — pads garment to portrait format; crops/resizes person image for IDM-VTON compatibility
3. **Virtual Try-On** — dresses the model in your garment via [IDM-VTON](https://huggingface.co/yisol/IDM-VTON) on HuggingFace Spaces

Output: `drape_output/base_model.png` (generated model) + `drape_output/result.png` (final try-on)

## Setup

```bash
git clone https://github.com/your-username/drape-ai.git
cd drape-ai
pip install -r requirements.txt
```

Set your [HuggingFace token](https://huggingface.co/settings/tokens):

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
```

Or pass it at runtime with `--token`.

## Usage

```bash
# Basic
python shirt_to_model.py --shirt myshirt.jpg

# Female model, editorial style
python shirt_to_model.py --shirt shirt.png --gender female --style editorial

# Use a pre-existing person image (skips generation)
python shirt_to_model.py --shirt shirt.jpg --person my_model.jpg

# Fine-tune quality and reproducibility
python shirt_to_model.py --shirt shirt.jpg --steps 40 --seed 123
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--shirt` | *(required)* | Path to garment image (JPG, PNG, WEBP, BMP) |
| `--person` | — | Use existing person image instead of generating one |
| `--gender` | `male` | Model gender: `male`, `female`, `any` |
| `--style` | `commercial` | Photography style: `commercial`, `editorial`, `streetwear`, `casual` |
| `--pose` | `front` | Pose: `front`, `dynamic`, `walking` |
| `--steps` | `30` | Denoising steps (higher = better quality, slower) |
| `--seed` | `42` | Random seed for reproducibility |
| `--output` | `./drape_output` | Output directory |
| `--no-open` | — | Skip auto-opening the result image |
| `--token` | — | HuggingFace API token (or use `HF_TOKEN` env var) |

## Requirements

- Python 3.8+
- Free [HuggingFace account](https://huggingface.co/join) + API token
- No GPU needed — uses HuggingFace Inference API and Spaces

Dependencies: `gradio_client`, `Pillow`, `requests`, `huggingface_hub`

## Notes

- **Try-on takes 60–120 seconds** on HuggingFace's free CPU tier
- **Garment images should be flat-lay / product shots** on a clean background
- **Person image must show full body** (head to toe) for a realistic result
- IDM-VTON expects portrait-oriented (768×1024) inputs — the pipeline handles this automatically
