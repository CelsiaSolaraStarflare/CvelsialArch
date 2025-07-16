import cv2
import torch
import os
import numpy as np
from pathlib import Path

from depth_anything_v2.dpt import DepthAnythingV2
import torch
print(torch.cuda.is_available())

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vits'  # Using small model as requested

    print("Loading Depth Anything V2 model...")
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()
    print("Model loaded successfully!")

    # Input and output directories
    input_dir = Path('data/images')
    output_dir = Path('data/depths')

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in input_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in image_extensions]

    print(f"Found {len(image_files)} images to process")

    for i, image_path in enumerate(image_files, 1):
        print(f"Processing {i}/{len(image_files)}: {image_path.name}")

        # Read image
        raw_img = cv2.imread(str(image_path))
        if raw_img is None:
            print(f"Error: Could not read image {image_path}")
            continue

        # Generate depth map
        depth = model.infer_image(raw_img)  # HxW raw depth map in numpy

        # Normalize depth for visualization (0-255)
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Save depth map
        output_path = output_dir / f"{image_path.stem}_depth.png"
        cv2.imwrite(str(output_path), depth_normalized)

        # Also save raw depth data as numpy array
        raw_output_path = output_dir / f"{image_path.stem}_depth_raw.npy"
        np.save(str(raw_output_path), depth)

        print(f"Saved depth map to: {output_path}")

    print("Depth map generation completed!")

if __name__ == "__main__":
    main()