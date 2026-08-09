#!/usr/bin/env python3

import argparse
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from rembg import new_session, remove

def process(args: argparse.Namespace) -> None:
    color = tuple(int(c) for c in args.outline_color.split(","))
    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_name(f"{in_path.stem}_thumb.jpg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.with_suffix("")

    img = Image.open(in_path).convert("RGB")

    # 1. Hair Matting & Background Removal (Single Pass)
    print(f"Extracting subject with fine matting using rembg ({args.model})...")
    session = new_session(args.model)
    
    # Returns an RGBA image where the alpha channel holds fractional transparency (gradients for hair)
    fg = remove(img, session=session)
    
    # Extract the soft alpha channel map
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
    if args.outline > 0:
        # Binarize the alpha mask so fine hair edges create a clean solid stroke
        binary_mask = alpha_mask.point(lambda p: 255 if p > 50 else 0)
        outline_mask = (binary_mask.filter(ImageFilter.MaxFilter(args.outline * 2 + 1))
                        .filter(ImageFilter.GaussianBlur(1.5))
                        .point(lambda p: 255 if p > 127 else 0))
        
    silhouette = Image.new("RGBA", img.size, (*color, 255))

    def paste_layers(base: Image.Image) -> Image.Image:
        if outline_mask: 
            base.paste(silhouette, (0, 0), outline_mask)
        # Pasting 'fg' with itself as mask retains soft transparency gradients for hair strands
        base.paste(fg, (0, 0), fg)
        return base

    # Save Cutout Sticker (if outline is active)
    if outline_mask:
        sticker_path = stem.parent / f"{stem.name}_outline.png"
        paste_layers(Image.new("RGBA", img.size, (0, 0, 0, 0))).save(sticker_path)
        print(f"Saved: {sticker_path}")

    # Save Final Composite
    paste_layers(bg.convert("RGBA")).convert("RGB").save(out_path, quality=100)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Thumbnail Generator: High-Precision Hair Matting")
    p.add_argument("input", help="Input image path")
    
    for flags, kwargs in (
        (("-o", "--output"), {"help": "Output image path"}),
        (("--outline",), {"type": int, "default": 14, "help": "Outline thickness (0=none)"}),
        (("--outline-color",), {"default": "255,255,255", "help": "Outline RGB color"}),
        (("--blur",), {"type": float, "default": 25.0, "help": "Background blur radius"}),
        (("--darken",), {"type": float, "default": 0.55, "help": "Background brightness (0-1)"}),
        (("--fade",), {"type": float, "default": 0.4, "help": "Background fade/desaturation (0-1)"}),
        (("--model",), {
            "default": "birefnet-general", 
            "help": "rembg model choice: 'birefnet-general' (best for hair), 'u2net_human_seg', or 'isnet-general-use'"
        }),
    ):
        p.add_argument(*flags, **kwargs)
        
    process(p.parse_args())