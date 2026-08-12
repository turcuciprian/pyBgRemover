#!/uscf/bin/env python3

import argparse
import re
from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image, ImageColor, ImageEnhance, ImageFilter
from transparent_background import Remover


def parse_color(value: str) -> tuple[int, int, int]:
    """Parse a hex (with/without '#') or named color string into an RGB tuple."""
    value = value.strip()
    if re.fullmatch(r"#?[0-9a-fA-F]{3}([0-9a-fA-F]{3})?", value):
        value = value if value.startswith("#") else f"#{value}"
    rgb = ImageColor.getrgb(value)
    return rgb[0], rgb[1], rgb[2]


def get_device() -> str:
    """Return the best available torch device: cuda, then mps, else cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def stylize_background(
    image: Image.Image, blur: float, fade: float, darken: float
) -> Image.Image:
    """Apply blur, fade, and brightness adjustments to the background."""
    background = image
    if blur > 0:
        background = background.filter(ImageFilter.GaussianBlur(blur))
    if fade > 0:
        background = ImageEnhance.Color(background).enhance(max(0.0, 1.0 - fade))
        background = ImageEnhance.Contrast(background).enhance(max(0.0, 1.0 - fade / 2))
    return ImageEnhance.Brightness(background).enhance(max(0.0, darken))


def build_outline_mask(alpha_mask: Image.Image, thickness: int) -> Image.Image:
    """Expand the binarized alpha mask into a solid stroke of the given thickness."""
    binary_mask = alpha_mask.point([255 if p > 50 else 0 for p in range(256)])
    return (
        binary_mask.filter(ImageFilter.MaxFilter(thickness * 2 + 1))
        .filter(ImageFilter.GaussianBlur(1.5))
        .point([255 if p > 127 else 0 for p in range(256)])
    )


def process(args: argparse.Namespace) -> None:
    in_path = Path(args.input)
    out_path = (
        Path(args.output)
        if args.output
        else in_path.with_name(f"{in_path.stem}_thumb.jpg")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.with_suffix("")

    def save(image: Image.Image, path: Path, **kwargs) -> None:
        image.save(path, **kwargs)
        print(f"Saved: {path}")

    image = Image.open(in_path).convert("RGB")

    # 1. InSPyReNet Background Removal
    device = get_device()
    print(f"Extracting subject with InSPyReNet (mode: {args.mode}) on {device}...")
    # type="rgba" (default) guarantees a PIL Image with fine alpha matting
    foreground = cast(
        Image.Image, Remover(mode=args.mode, device=device).process(image)
    )
    alpha_mask = foreground.getchannel("A")
    save(alpha_mask, stem.parent / f"{stem.name}_alpha_mask.png")

    # 2. Generate Stylized Background
    background = stylize_background(image, args.blur, args.fade, args.darken)
    save(background, stem.parent / f"{stem.name}_background.jpg", quality=100)

    # 3. Generate Outline Layers (skipped entirely when --outline 0)
    outline_mask = silhouette = None
    if args.outline > 0:
        outline_mask = build_outline_mask(alpha_mask, args.outline)
        silhouette = Image.new(
            "RGBA", image.size, (*parse_color(args.outlinecolor), 255)
        )

    def paste_layers(base: Image.Image) -> Image.Image:
        if outline_mask and silhouette:
            base.paste(silhouette, (0, 0), outline_mask)
        # Paste foreground using its native alpha channel to preserve hair transparency
        base.paste(foreground, (0, 0), foreground)
        return base

    # 4. Save Standalone Cutout (always saved, with or without outline)
    save(
        paste_layers(Image.new("RGBA", image.size, (0, 0, 0, 0))),
        stem.parent / f"{stem.name}_cutout.png",
    )

    # 5. Save Final Composite
    save(paste_layers(background.convert("RGBA")).convert("RGB"), out_path, quality=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Thumbnail Generator: InSPyReNet Salient Object Matting"
    )
    parser.add_argument("input", help="Input image path")

    arguments: tuple[tuple[tuple[str, ...], dict[str, Any]], ...] = (
        (("-o", "--output"), {"help": "Output image path"}),
        (
            ("--outline",),
            {"type": int, "default": 14, "help": "Outline thickness (0=none)"},
        ),
        (
            ("--outlinecolor",),
            {
                "default": "#fff",
                "help": (
                    "Outline color as hex string (e.g. #fff or #ffffff) or named color"
                ),
            },
        ),
        (
            ("--blur",),
            {"type": float, "default": 5.0, "help": "Background blur radius"},
        ),
        (
            ("--darken",),
            {"type": float, "default": 0.55, "help": "Background brightness (0-1)"},
        ),
        (
            ("--fade",),
            {
                "type": float,
                "default": 0.4,
                "help": "Background fade/desaturation (0-1)",
            },
        ),
        (
            ("--mode",),
            {
                "default": "base",
                "choices": ["base", "fast", "nightly"],
                "help": (
                    "InSPyReNet model mode: 'base' (highest quality),"
                    " 'fast' (lightweight), or 'nightly' (latest experimental)"
                ),
            },
        ),
    )
    for flags, kwargs in arguments:
        parser.add_argument(*flags, **kwargs)

    process(parser.parse_args())
