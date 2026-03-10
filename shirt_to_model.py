#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════╗
║         DRAPE — AI Fashion Model Generator           ║
║   Uses free Hugging Face Spaces (no GPU required)    ║
╚══════════════════════════════════════════════════════╝

PIPELINE:
  1. You provide a shirt image
  2. Script generates a base fashion model image (HF Inference API)
  3. Virtual try-on dresses the model in your shirt (IDM-VTON Space)
  4. Result saved to output file

FREE MODELS USED:
  • Base model generation : HF Inference API — stabilityai/stable-diffusion-xl-base-1.0
  • Virtual try-on        : HF Space       — yisol/IDM-VTON (Gradio API)
  • Fallback try-on       : HF Space       — Nymbo/Virtual-Try-On

REQUIREMENTS:
  • Python 3.8+
  • Free HuggingFace account + token (https://huggingface.co/settings/tokens)

USAGE:
  python shirt_to_model.py --shirt path/to/shirt.jpg
  python shirt_to_model.py --shirt shirt.jpg --gender female --style editorial
  python shirt_to_model.py --shirt shirt.jpg --person my_photo.jpg   # skip generation
  python shirt_to_model.py --shirt shirt.jpg --steps 40 --seed 123
"""

import os
import sys
import time
import shutil
import argparse
import subprocess
import textwrap


# ─────────────────────────────────────────────
#  Auto-install dependencies
# ─────────────────────────────────────────────

REQUIRED = ["gradio_client", "Pillow", "requests", "huggingface_hub"]

def install_packages():
    print("📦 Checking / installing dependencies...")
    for pkg in REQUIRED:
        try:
            __import__(pkg.replace("-", "_").split("[")[0])
        except ImportError:
            print(f"   Installing {pkg}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "-q"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    print("   ✅ All dependencies ready.\n")

install_packages()

# ─────────────────────────────────────────────
#  Imports (after install)
# ─────────────────────────────────────────────

import requests
from pathlib import Path
from PIL import Image
from gradio_client import Client, handle_file
from huggingface_hub import InferenceClient


# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

VTON_SPACES = [
    "yisol/IDM-VTON",           # Primary — best quality
    "freddyaboulton/IDM-VTON",  # Mirror of IDM-VTON
    "Nymbo/Virtual-Try-On",     # Fallback
]

SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

GENDER_PROMPTS = {
    "male":   "young male fashion model, athletic build, neutral expression",
    "female": "young female fashion model, slim build, neutral expression",
    "any":    "androgynous fashion model, slim build, neutral expression",
}

STYLE_PROMPTS = {
    "editorial":  "editorial high fashion, studio lighting, Vogue style",
    "commercial": "commercial catalog photography, clean white background",
    "streetwear": "streetwear editorial, urban background, natural light",
    "casual":     "lifestyle photography, natural daylight, relaxed pose",
}

POSE_PROMPTS = {
    "front":    "standing upright, front facing, arms at sides",
    "dynamic":  "confident pose, slight angle, hands in pockets",
    "walking":  "walking pose, candid, movement, dynamic",
}


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def banner():
    print("""
╔══════════════════════════════════════════════════════╗
║         DRAPE — AI Fashion Model Generator           ║
║         Free HuggingFace Spaces Pipeline             ║
╚══════════════════════════════════════════════════════╝
""")

def step(n, text):
    print(f"\n{'─'*50}")
    print(f"  STEP {n} │ {text}")
    print(f"{'─'*50}")

def info(msg):
    print(f"  ℹ  {msg}")

def ok(msg):
    print(f"  ✅ {msg}")

def warn(msg):
    print(f"  ⚠️  {msg}")

def err(msg):
    print(f"  ❌ {msg}")

def get_hf_token():
    """Get HF token from env, file, or prompt user."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        info("Found HF_TOKEN in environment.")
        return token

    # Check HF cache
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        token = token_file.read_text().strip()
        if token:
            info("Found saved HuggingFace token.")
            return token

    print("""
  🔑 HuggingFace token required (free account).
     Get yours at: https://huggingface.co/settings/tokens
     (Read-only token is enough)
""")
    token = input("  Paste your HF token: ").strip()
    if not token:
        err("No token provided. Exiting.")
        sys.exit(1)

    # Save for next run
    save = input("  Save token for future runs? [y/N]: ").strip().lower()
    if save == "y":
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(token)
        ok("Token saved to ~/.cache/huggingface/token")

    return token


def validate_image(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        err(f"File not found: {path}")
        sys.exit(1)
    if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        err(f"Unsupported image format: {p.suffix}")
        sys.exit(1)
    # Check size
    size = p.stat().st_size / (1024 * 1024)
    info(f"Shirt image: {p.name} ({size:.1f} MB)")
    return p


def resize_for_api(image_path: Path, max_size: int = 768) -> Path:
    """Resize image if too large, return path to (possibly new) file."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        out = image_path.parent / f"_resized_{image_path.name}"
        img.save(out)
        info(f"Resized image to {img.size[0]}x{img.size[1]} for API.")
        return out
    return image_path


def prepare_garment_image(image_path: Path, target_w: int = 768, target_h: int = 1024) -> Path:
    """
    Pad garment flat-lay to portrait 768x1024 with white background.
    IDM-VTON requires portrait-oriented garment images — landscape inputs cause broken pixels.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # Scale to fit inside target while preserving aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Paste onto white canvas
    canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas.paste(img, (offset_x, offset_y))

    out = image_path.parent / f"_garment_prepared_{image_path.stem}.png"
    canvas.save(out)
    info(f"Garment padded to {target_w}x{target_h} portrait.")
    return out


def prepare_person_image(image_path: Path, target_w: int = 768, target_h: int = 1024) -> Path:
    """
    Resize + center-crop person image to 768x1024 portrait.
    IDM-VTON needs a full-body portrait-oriented person image.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    if w == target_w and h == target_h:
        return image_path

    # Scale so the shorter side fits, then center crop
    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    img = img.crop((left, top, left + target_w, top + target_h))

    out = image_path.parent / f"_person_prepared_{image_path.stem}.png"
    img.save(out)
    info(f"Person image prepared to {target_w}x{target_h}.")
    return out


# ─────────────────────────────────────────────
#  Step 1 — Generate base model person
# ─────────────────────────────────────────────

def generate_base_model(
    hf_token: str,
    gender: str,
    style: str,
    pose: str,
    seed: int,
    output_dir: Path,
) -> Path:
    """Generate a fashion model person image using HF Inference API (free tier)."""

    # Photorealistic full-body prompt — must show feet to head for IDM-VTON
    prompt = (
        f"Full body photograph of a {GENDER_PROMPTS[gender]}, "
        f"wearing a plain fitted white t-shirt and slim chinos, {POSE_PROMPTS[pose]}, "
        f"{STYLE_PROMPTS[style]}, "
        "full body visible from head to toe, feet on ground, "
        "real human skin texture, natural lighting, DSLR photograph, "
        "shot on Canon EOS R5, 85mm lens, shallow depth of field, "
        "hyperrealistic, ultra detailed, photographic quality"
    )

    negative_prompt = (
        "illustration, painting, drawing, anime, cartoon, 3d render, cgi, "
        "deformed, distorted, disfigured, bad anatomy, extra limbs, missing limbs, "
        "cropped body, cut off legs, cut off feet, half body, upper body only, "
        "blurry, low quality, text, watermark, logo, plastic skin, fake, airbrushed"
    )

    info(f"Prompt: {textwrap.shorten(prompt, 90)}")

    # Try FLUX.1-dev first — far more photorealistic than SDXL
    MODELS = [
        ("black-forest-labs/FLUX.1-dev",     {"width": 768, "height": 1024, "num_inference_steps": 25}),
        ("black-forest-labs/FLUX.1-schnell", {"width": 768, "height": 1024, "num_inference_steps": 4}),
        (SDXL_MODEL,                         {"width": 768, "height": 1024, "num_inference_steps": 30,
                                               "guidance_scale": 7.5, "negative_prompt": negative_prompt}),
    ]

    for model_id, kwargs in MODELS:
        info(f"Trying model: {model_id} @ 768×1024...")
        try:
            client = InferenceClient(model=model_id, token=hf_token)
            image = client.text_to_image(prompt=prompt, **kwargs)
            out_path = output_dir / "base_model.png"
            image.save(out_path)
            ok(f"Base model saved → {out_path}  (via {model_id})")
            return out_path
        except Exception as e:
            warn(f"{model_id} failed: {e}")
            continue

    err("All generation models failed. Use --person with your own full-body model photo.")
    sys.exit(1)


# ─────────────────────────────────────────────
#  Step 2 — Virtual Try-On (IDM-VTON)
# ─────────────────────────────────────────────

def virtual_tryon(
    person_path: Path,
    garment_path: Path,
    hf_token: str,
    denoise_steps: int,
    seed: int,
    output_dir: Path,
) -> Path:
    """Dress the person in the garment using IDM-VTON via Gradio API."""

    last_error = None

    for space_id in VTON_SPACES:
        info(f"Trying Space: {space_id}")
        try:
            client = Client(space_id, token=hf_token)

            info(f"Calling /tryon endpoint (steps={denoise_steps}, seed={seed})...")
            info("This usually takes 60–120 seconds on free CPU tier...")

            result = client.predict(
                dict={
                    "background": handle_file(str(person_path)),
                    "layers": [],
                    "composite": None,
                },
                garm_img=handle_file(str(garment_path)),
                garment_des="",        # garment description (optional)
                is_checked=True,       # auto-masking
                is_checked_crop=False, # auto-cropping
                denoise_steps=denoise_steps,
                seed=seed,
                api_name="/tryon",
            )

            # result[0] is the output image path (local temp file)
            result_img_path = result[0] if isinstance(result, (list, tuple)) else result
            out_path = output_dir / "result.png"
            shutil.copy(result_img_path, out_path)
            ok(f"Try-on result saved → {out_path}")
            return out_path

        except Exception as e:
            warn(f"Space {space_id} failed: {e}")
            last_error = e
            continue

    err(f"All try-on spaces failed. Last error: {last_error}")
    err(
        "Tips:\n"
        "     • The Space may be sleeping — visit it in browser first:\n"
        "       https://huggingface.co/spaces/yisol/IDM-VTON\n"
        "     • Free tier has queue limits; try again in a few minutes.\n"
        "     • Make sure your HF token is valid."
    )
    sys.exit(1)


# ─────────────────────────────────────────────
#  Step 3 — Show result
# ─────────────────────────────────────────────

def show_result(result_path: Path, open_image: bool):
    print(f"""
╔══════════════════════════════════════════════════════╗
║  ✅ DONE! Your fashion look is ready.                ║
╚══════════════════════════════════════════════════════╝

  Output: {result_path.resolve()}
""")
    if open_image:
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(result_path)])
            elif sys.platform == "win32":
                os.startfile(str(result_path))
            else:
                subprocess.run(["xdg-open", str(result_path)])
        except Exception:
            pass


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        prog="shirt_to_model.py",
        description="Generate AI fashion model wearing your shirt using free HuggingFace Spaces.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python shirt_to_model.py --shirt myshirt.jpg
          python shirt_to_model.py --shirt shirt.png --gender female --style editorial
          python shirt_to_model.py --shirt shirt.jpg --person model.jpg
          python shirt_to_model.py --shirt shirt.jpg --steps 40 --seed 42 --no-open

        Set HF_TOKEN env variable to skip the token prompt:
          export HF_TOKEN=hf_xxxxxxxxxxxx
          python shirt_to_model.py --shirt shirt.jpg
        """),
    )

    parser.add_argument(
        "--shirt", "-s",
        required=True,
        help="Path to your shirt/garment image (JPG, PNG, WEBP)"
    )
    parser.add_argument(
        "--person", "-p",
        default=None,
        help="Path to a person/model image (skip AI generation step)"
    )
    parser.add_argument(
        "--gender", "-g",
        choices=["male", "female", "any"],
        default="male",
        help="Gender of the generated model (default: male)"
    )
    parser.add_argument(
        "--style",
        choices=list(STYLE_PROMPTS.keys()),
        default="commercial",
        help="Photography style (default: commercial)"
    )
    parser.add_argument(
        "--pose",
        choices=list(POSE_PROMPTS.keys()),
        default="front",
        help="Model pose (default: front)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Denoising steps for try-on model (higher = better quality, slower) (default: 30)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output", "-o",
        default="./drape_output",
        help="Output directory (default: ./drape_output)"
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Don't auto-open the result image"
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)"
    )

    return parser.parse_args()


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    banner()
    args = parse_args()

    # Output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # HF Token
    if args.token:
        hf_token = args.token
    else:
        hf_token = get_hf_token()

    # ── Step 1: Validate & prepare shirt image ──
    step(1, "Preparing garment image")
    shirt_path = validate_image(args.shirt)
    shirt_path = prepare_garment_image(shirt_path)   # → 768×1024 portrait

    # ── Step 2: Get person/model image ─────────
    if args.person:
        step(2, "Using provided person image (skipping generation)")
        person_path = validate_image(args.person)
        person_path = prepare_person_image(person_path)  # → 768×1024 portrait
        ok(f"Person image: {person_path}")
    else:
        step(2, f"Generating base model [{args.gender} / {args.style} / {args.pose}]")
        person_path = generate_base_model(
            hf_token=hf_token,
            gender=args.gender,
            style=args.style,
            pose=args.pose,
            seed=args.seed,
            output_dir=output_dir,
        )
        # Ensure exact 768×1024 for IDM-VTON
        person_path = prepare_person_image(person_path)

    # ── Step 3: Virtual try-on ─────────────────
    step(3, "Running virtual try-on (IDM-VTON)")
    info("Free tier may queue — be patient ☕")
    t0 = time.time()
    result_path = virtual_tryon(
        person_path=person_path,
        garment_path=shirt_path,
        hf_token=hf_token,
        denoise_steps=args.steps,
        seed=args.seed,
        output_dir=output_dir,
    )
    elapsed = time.time() - t0
    info(f"Completed in {elapsed:.0f}s")

    # ── Done ───────────────────────────────────
    show_result(result_path, open_image=not args.no_open)


if __name__ == "__main__":
    main()
