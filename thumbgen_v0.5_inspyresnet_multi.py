#!/usr/bin/env python3

import argparse
from pathlib import Path
import torch
from PIL import Image, ImageEnhance, ImageFilter
from ultralytics import YOLO
from transparent_background import Remover

def process(args: argparse.Namespace) -> None:
    color = tuple(int(c) for c in args.outline_color.split(","))
    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_name(f"{in_path.stem}_thumb.jpg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.with_suffix("")

    img = Image.open(in_path).convert("RGB")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    # 1. Detect All People via YOLO
    print(f"Detecting all humans with YOLO ({args.yolo_model})...")
    yolo = YOLO(args.yolo_model)
    results = yolo(img, verbose=False)[0].boxes
    person_boxes = results.xyxy.cpu().numpy()[results.cls.cpu().numpy() == 0] if results is not None and len(results) else []

    # 2. Extract Each Person with InSPyReNet & Composite
    remover = Remover(mode=args.mode, device=device)
    fg = Image.new("RGBA", img.size, (0, 0, 0, 0))

    if len(person_boxes) == 0:
        print("Warning: No human detected. Extracting entire frame with InSPyReNet as fallback...")
        fg = remover.process(img)
    else:
        print(f"Found {len(person_boxes)} person(s). Matting each subject with InSPyReNet...")
        for i, box in enumerate(person_boxes, start=1):
            x1, y1, x2, y2 = map(int, box)
            
            # Add padding around the box so hair/edges aren't clipped at crop boundaries
            pad = args.crop_padding
            crop_x1 = max(0, x1 - pad)
            crop_y1 = max(0, y1 - pad)
            crop_x2 = min(img.width, x2 + pad)
            crop_y2 = min(img.height, y2 + pad)

            # Crop person, run InSPyReNet matting, and paste onto master foreground
            crop = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            crop_fg = remover.process(crop)
            fg.paste(crop_fg, (crop_x1, crop_y1), crop_fg)

    alpha_mask = fg.getchannel("A")

    # 3. Generate Stylized Background
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

    # 4. Generate Outline & Composite Layers
    outline_mask = None
    if args.outline > 0:
        binary_mask = alpha_mask.point(lambda p: 255 if p > 50 else 0)
        outline_mask = (binary_mask.filter(ImageFilter.MaxFilter(args.outline * 2 + 1))
                        .filter(ImageFilter.GaussianBlur(1.5))
                        .point(lambda p: 255 if p > 127 else 0))
        
    silhouette = Image.new("RGBA", img.size, (*color, 255))

    def paste_layers(base: Image.Image) -> Image.Image:
        if outline_mask: 
            base.paste(silhouette, (0, 0), outline_mask)
        base.paste(fg, (0, 0), fg)
        return base

    # 5. Save Standalone Cutout
    cutout_path = stem.parent / f"{stem.name}_cutout.png"
    paste_layers(Image.new("RGBA", img.size, (0, 0, 0, 0))).save(cutout_path)
    print(f"Saved: {cutout_path}")

    # 6. Save Final Composite
    paste_layers(bg.convert("RGBA")).convert("RGB").save(out_path, quality=100)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Multi-Person Thumbnail Generator: YOLO + InSPyReNet")
    p.add_argument("input", help="Input image path")
    
    for flags, kwargs in (
        (("-o", "--output"), {"help": "Output image path"}),
        (("--outline",), {"type": int, "default": 14, "help": "Outline thickness (0=none)"}),
        (("--outline-color",), {"default": "255,255,255", "help": "Outline RGB color"}),
        (("--blur",), {"type": float, "default": 25.0, "help": "Background blur radius"}),
        (("--darken",), {"type": float, "default": 0.55, "help": "Background brightness (0-1)"}),
        (("--fade",), {"type": float, "default": 0.4, "help": "Background fade/desaturation (0-1)"}),
        (("--crop-padding",), {"type": int, "default": 30, "help": "Pixel padding around person crops"}),
        (("--yolo-model",), {"default": "yolov8x.pt", "help": "YOLO model for multi-person detection"}),
        (("--mode",), {
            "default": "base", 
            "choices": ["base", "fast", "nightly"], 
            "help": "InSPyReNet mode: 'base', 'fast', or 'nightly'"
        }),
    ):
        p.add_argument(*flags, **kwargs)
        
    process(p.parse_args())