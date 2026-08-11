#!/usr/bin/env python3

import argparse
from pathlib import Path
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageColor
from transparent_background import Remover

def process(args: argparse.Namespace) -> None:
    color_str = args.outlinecolor.lstrip("#")
    color = ImageColor.getrgb(f"#{color_str}" if len(color_str) in (3,6) else args.outlinecolor)
    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_name(f"{in_path.stem}_thumb.jpg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.with_suffix("")

    img = Image.open(in_path).convert("RGB")

    # 1. InSPyReNet Background Removal
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Extracting subject with InSPyReNet (mode: {args.mode}) on {device}...")
    
    # Initialize InSPyReNet remover
    remover = Remover(mode=args.mode, device=device)
    
    # Process image: Returns an RGBA PIL Image with fine alpha matting
    fg = remover.process(img)
    alpha_mask = fg.getchannel("A")

    # 2. Generate Stylized Background
    bg = img.convert("RGB")
    if args.blur > 0: 
        bg = bg.filter(ImageFilter.GaussianBlur(args.blur))
    if args.fade > 0:
        bg = ImageEnhance.Color(bg).enhance(max(0.0, 1.0 - args.fade))
        bg = ImageEnhance.Contrast(bg).enhance(max(0.0, 1.0 - args.fade / 2))
    bg = ImageEnhance.Brightness(bg).enhance(max(0.0, args.darken))
    
    bg_path = stem.parent / f"{stem.name}_background.jpg"
    bg.save(bg_path, quality=100)
    print(f"Saved: {bg_path}")

    # 3. Generate Outline & Composite Layers
    outline_mask = None
    if args.outline >= 0:
        # Binarize alpha mask so fine hair edges create a solid stroke
        binary_mask = alpha_mask.point(lambda p: 255 if p > 50 else 0)
        outline_mask = (binary_mask.filter(ImageFilter.MaxFilter(args.outline * 2 + 1))
                        .filter(ImageFilter.GaussianBlur(1.5))
                        .point(lambda p: 255 if p > 127 else 0))
    # saving the alpha mask of the person cutout
    alpha_mask.save(f"{stem}_alpha_mask.png")
    
    
    silhouette = Image.new("RGBA", img.size, (*color, 255)) 
    
    # 

    def paste_layers(base: Image.Image) -> Image.Image:
        if outline_mask: 
            base.paste(silhouette, (0, 0), outline_mask)
        # Paste foreground using its native alpha channel to preserve hair transparency
        base.paste(fg, (0, 0), fg)
        return base

    # 4. Save Standalone Cutout (Always saved, with or without outline)
    cutout_path = stem.parent / f"{stem.name}_cutout.png"
    paste_layers(Image.new("RGBA", img.size, (0, 0, 0, 0))).save(cutout_path)
    print(f"Saved: {cutout_path}")

    # 5. Save Final Composite
    paste_layers(bg.convert("RGBA")).convert("RGB").save(out_path, quality=100)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Thumbnail Generator: InSPyReNet Salient Object Matting")
    p.add_argument("input", help="Input image path")
    
    for flags, kwargs in (
        (("-o", "--output"), {"help": "Output image path"}),
        (("--outline",), {"type": int, "default": 14, "help": "Outline thickness (0=none)"}),
        (("--outlinecolor",), {"default": "#fff", "help": "Hex white color. the string here should be a hex value with the # before it"}),
        (("--blur",), {"type": float, "default": 5.0, "help": "Background blur radius"}),
        (("--darken",), {"type": float, "default": 0.55, "help": "Background brightness (0-1)"}),
        (("--fade",), {"type": float, "default": 0.4, "help": "Background fade/desaturation (0-1)"}),
        (("--mode",), {
            "default": "base", 
            "choices": ["base", "fast", "nightly"], 
            "help": "InSPyReNet model mode: 'base' (highest quality), 'fast' (lightweight), or 'nightly' (latest experimental)"
        }),
    ):
        p.add_argument(*flags, **kwargs)
        
    process(p.parse_args())