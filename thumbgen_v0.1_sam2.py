#!/usr/bin/env python3
"""
thumbgen_v0.1_sam2.py - Thumbnail-style image generator.

Takes a photo, removes the background (keeping the person/persons) using SAM 2
automatic mask generation, draws a white outline around them, and places them
back over the original background - which is blurred, darkened and faded.

Saves 3 images next to the output path:
    <output>.jpg            the final composite
    <output>_outline.png    the person cutout with the outline border (transparent PNG)
    <output>_background.jpg just the faded/blurred/darkened background

Usage:
    python thumbgen_v0.1_sam2.py input.jpg -o output.jpg
    python thumbgen_v0.1_sam2.py input.jpg -o output.jpg --outline 18 --blur 30 \
        --darken 0.5 --fade 0.5
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2


def load_sam2_mask_generator(
    config_path: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    checkpoint_path: str = "sam2.1_hiera_large.pt",
) -> SAM2AutomaticMaskGenerator:
    """Initialize SAM 2 on available hardware (CUDA / MPS / CPU)."""
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    sam2_model = build_sam2(config_path, checkpoint_path, device=device)
    return SAM2AutomaticMaskGenerator(sam2_model)


def cut_out_subject(
    img: Image.Image,
    mask_generator: SAM2AutomaticMaskGenerator,
) -> Image.Image:
    """Segment the subject(s) with SAM 2 automatic mask generation.

    Keeps masks overlapping the image-center region (falls back to the
    largest mask if none do) and returns an RGBA cutout.
    """
    np_img = np.array(img)
    masks = mask_generator.generate(np_img)

    # Ignore masks that cover (almost) the whole image - they are background-like.
    max_area = 0.9 * img.width * img.height
    masks = [m for m in masks if m["area"] < max_area]

    if not masks:
        print("Warning: No subject detected by SAM 2.")
        # Return transparent image
        return Image.new("RGBA", img.size, (0, 0, 0, 0))

    # Keep masks overlapping the central window (middle 50% in both axes).
    x0, x1 = img.width // 4, 3 * img.width // 4
    y0, y1 = img.height // 4, 3 * img.height // 4
    selected = [m for m in masks if m["segmentation"][y0:y1, x0:x1].any()]

    # Fallback: nothing near the center -> take the single largest mask.
    if not selected:
        selected = [max(masks, key=lambda m: m["area"])]

    # Combine all selected masks into a single boolean array
    combined_mask = np.zeros((img.height, img.width), dtype=bool)
    for m in selected:
        combined_mask = np.logical_or(combined_mask, m["segmentation"])

    # Construct RGBA image cutout
    mask_pil = Image.fromarray((combined_mask * 255).astype(np.uint8), mode="L")
    fg = img.convert("RGBA")
    fg.putalpha(mask_pil)
    return fg


def make_background(
    img: Image.Image,
    blur: float,
    darken: float,
    fade: float,
) -> Image.Image:
    """Original image, blurred + darkened + faded (desaturated/contrast-softened)."""
    bg = img.convert("RGB")
    if blur > 0:
        bg = bg.filter(ImageFilter.GaussianBlur(blur))
    bg = ImageEnhance.Color(bg).enhance(max(0.0, 1.0 - fade))
    bg = ImageEnhance.Contrast(bg).enhance(max(0.0, 1.0 - fade / 2))
    bg = ImageEnhance.Brightness(bg).enhance(max(0.0, darken))
    return bg


def prepare_mask(fg: Image.Image) -> Image.Image:
    """Refine mask edges.

    Since SAM 2 produces sharp boundaries without halo artifacts,
    we use a soft, non-destructive edge pass.
    """
    mask = fg.getchannel("A").point(lambda p: 255 if p > 128 else 0)
    mask = mask.filter(ImageFilter.MinFilter(3))  # Gentle 1px edge trim
    mask = mask.filter(ImageFilter.GaussianBlur(1.5))
    mask = mask.point(lambda p: 255 if p > 127 else 0)
    return mask.filter(ImageFilter.GaussianBlur(0.6))


def make_outline_mask(mask: Image.Image, thickness: int) -> Image.Image | None:
    """Dilate the subject mask iteratively to create a smooth outline region."""
    if thickness <= 0:
        return None

    dilated = mask.copy()
    # Iterative 3x3 MaxFilter expansion avoids kernel overflow issues on large thickness
    for _ in range(thickness):
        dilated = dilated.filter(ImageFilter.MaxFilter(3))

    dilated = dilated.filter(ImageFilter.GaussianBlur(1.5))
    return dilated.point(lambda p: 255 if p > 127 else 0)


def process(
    input_path: str | Path,
    output_path: str | Path,
    mask_generator: SAM2AutomaticMaskGenerator,
    outline: int = 14,
    outline_color: tuple[int, int, int] = (255, 255, 255),
    blur: float = 25,
    darken: float = 0.55,
    fade: float = 0.4,
) -> None:
    img = Image.open(input_path).convert("RGB")

    # 1. Keep the person(s), drop the background using SAM 2
    fg = cut_out_subject(img, mask_generator)
    mask = prepare_mask(fg)
    if mask.getbbox() is None:
        print("Warning: no subject detected - output will be background only.")
    fg.putalpha(mask)

    # 2. Background: original image, blurred / darkened / faded.
    background = make_background(img, blur, darken, fade)
    result = background.convert("RGBA")

    # 3. White outline behind the subject.
    outline_mask = make_outline_mask(mask, outline)
    silhouette = Image.new("RGBA", img.size, (*outline_color, 255))

    if outline_mask is not None:
        result.paste(silhouette, (0, 0), outline_mask)

    # 4. Subject on top.
    result.paste(fg, (0, 0), fg)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.with_suffix("")

    # Extra output 1: subject cutout with the outline border (transparent PNG sticker).
    if outline_mask is not None:
        sticker = Image.new("RGBA", img.size, (0, 0, 0, 0))
        sticker.paste(silhouette, (0, 0), outline_mask)
        sticker.paste(fg, (0, 0), fg)
        outline_path = stem.parent / f"{stem.name}_outline.png"
        sticker.save(outline_path)
        print(f"Saved: {outline_path}")

    # Extra output 2: just the faded / blurred / darkened background.
    background_path = stem.parent / f"{stem.name}_background.jpg"
    background.save(background_path, quality=100)
    print(f"Saved: {background_path}")

    # Main output: full composite.
    result.convert("RGB").save(output_path, quality=100)
    print(f"Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cut out the person(s), add a white outline, and put them "
        "over a blurred, darkened, faded version of the original background."
    )
    p.add_argument("input", help="Input image path")
    p.add_argument("-o", "--output", help="Output image path (default: <input>_thumb.jpg)")
    p.add_argument(
        "--outline",
        type=int,
        default=14,
        help="Outline thickness in pixels (0 = no outline, default: 14)",
    )
    p.add_argument(
        "--outline-color",
        default="255,255,255",
        help="Outline color as R,G,B (default: 255,255,255)",
    )
    p.add_argument(
        "--blur",
        type=float,
        default=25,
        help="Background Gaussian blur radius (default: 25)",
    )
    p.add_argument(
        "--darken",
        type=float,
        default=0.55,
        help="Background brightness factor, 0=black 1=original (default: 0.55)",
    )
    p.add_argument(
        "--fade",
        type=float,
        default=0.4,
        help="Background fade amount (desaturation), 0-1 (default: 0.4)",
    )
    p.add_argument(
        "--sam2-config",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM 2 config path",
    )
    p.add_argument(
        "--sam2-ckpt",
        default="sam2.1_hiera_large.pt",
        help="SAM 2 checkpoint path",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    color = tuple(int(c) for c in args.outline_color.split(","))
    if len(color) != 3:
        raise SystemExit("--outline-color must be R,G,B (e.g. 255,255,255)")

    output = args.output
    if output is None:
        stem = Path(args.input)
        output = str(stem.with_name(stem.stem + "_thumb.jpg"))

    # Load SAM 2 mask generator
    mask_generator = load_sam2_mask_generator(args.sam2_config, args.sam2_ckpt)

    process(
        args.input,
        output,
        mask_generator,
        outline=args.outline,
        outline_color=color,
        blur=args.blur,
        darken=args.darken,
        fade=args.fade,
    )


if __name__ == "__main__":
    main()
