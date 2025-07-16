import os
import csv
import json
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime
import torch  # <-- Added for CUDA & model support


def convert_gps_coordinates(coord, ref):
    """Convert GPS coordinates from degrees/minutes/seconds to decimal degrees"""
    if coord is None or ref is None:
        return None

    degrees = float(coord[0])
    minutes = float(coord[1])
    seconds = float(coord[2])

    decimal_degrees = degrees + (minutes / 60.0) + (seconds / 3600.0)

    if ref in ['S', 'W']:
        decimal_degrees = -decimal_degrees

    return decimal_degrees


def extract_exif_data(image_path):
    """Extract EXIF data from an image file"""
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()

        if exif_data is None:
            return None

        result = {
            'filename': image_path.name,
            'datetime': None,
            'gps_latitude': None,
            'gps_longitude': None,
            'gps_altitude': None,
            'orientation': None,
            'camera_make': None,
            'camera_model': None,
            'image_width': None,
            'image_height': None
        }

        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)

            if tag in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']:
                if value and result['datetime'] is None:
                    try:
                        result['datetime'] = datetime.strptime(value, '%Y:%m:%d %H:%M:%S').isoformat()
                    except ValueError:
                        result['datetime'] = value

            elif tag == 'Orientation':
                orientation_map = {
                    1: 'Normal',
                    2: 'Mirrored horizontal',
                    3: 'Rotated 180°',
                    4: 'Mirrored vertical',
                    5: 'Mirrored horizontal, rotated 270°',
                    6: 'Rotated 90°',
                    7: 'Mirrored horizontal, rotated 90°',
                    8: 'Rotated 270°'
                }
                result['orientation'] = orientation_map.get(value, f'Unknown ({value})')

            elif tag == 'Make':
                result['camera_make'] = value
            elif tag == 'Model':
                result['camera_model'] = value
            elif tag == 'ExifImageWidth':
                result['image_width'] = value
            elif tag == 'ExifImageHeight':
                result['image_height'] = value

            elif tag == 'GPSInfo':
                gps_data = {}
                for gps_tag_id, gps_value in value.items():
                    gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_data[gps_tag] = gps_value

                if 'GPSLatitude' in gps_data and 'GPSLatitudeRef' in gps_data:
                    result['gps_latitude'] = convert_gps_coordinates(
                        gps_data['GPSLatitude'],
                        gps_data['GPSLatitudeRef']
                    )

                if 'GPSLongitude' in gps_data and 'GPSLongitudeRef' in gps_data:
                    result['gps_longitude'] = convert_gps_coordinates(
                        gps_data['GPSLongitude'],
                        gps_data['GPSLongitudeRef']
                    )

                if 'GPSAltitude' in gps_data:
                    altitude = float(gps_data['GPSAltitude'])
                    if 'GPSAltitudeRef' in gps_data and gps_data['GPSAltitudeRef'] == 1:
                        altitude = -altitude
                    result['gps_altitude'] = altitude

        return result

    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")
        return None


def load_model(model_path):
    """Load a PyTorch model with CUDA support if available"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        model = torch.load(model_path, map_location=device)
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
        return model, device
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None, device


def main():
    input_dir = Path('data/images')
    output_dir = Path('data')
    model_path = Path('data/model.pt')  # Modify as needed

    # Load model
    if model_path.exists():
        model, device = load_model(model_path)
    else:
        print(f"Model not found at {model_path}, skipping model-related processing.")
        model, device = None, torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Collect image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    print(f"Found {len(image_files)} images to process")

    exif_results = []

    for i, image_path in enumerate(image_files, 1):
        print(f"Processing {i}/{len(image_files)}: {image_path.name}")
        exif_data = extract_exif_data(image_path)
        if exif_data:
            exif_results.append(exif_data)
        else:
            print(f"  No EXIF data found for {image_path.name}")

    print(f"\nSuccessfully extracted EXIF data from {len(exif_results)} images")

    # Save to CSV
    csv_output = output_dir / 'exif_data.csv'
    if exif_results:
        with open(csv_output, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = exif_results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(exif_results)
        print(f"EXIF data saved to: {csv_output}")

    # Save to JSON
    json_output = output_dir / 'exif_data.json'
    with open(json_output, 'w', encoding='utf-8') as jsonfile:
        json.dump(exif_results, jsonfile, indent=2, ensure_ascii=False)
    print(f"EXIF data saved to: {json_output}")

    # Summary
    print("\n=== EXIF Data Summary ===")
    gps_count = sum(1 for r in exif_results if r['gps_latitude'] is not None)
    datetime_count = sum(1 for r in exif_results if r['datetime'] is not None)
    orientation_count = sum(1 for r in exif_results if r['orientation'] is not None)

    print(f"Images with GPS data: {gps_count}/{len(exif_results)}")
    print(f"Images with datetime: {datetime_count}/{len(exif_results)}")
    print(f"Images with orientation: {orientation_count}/{len(exif_results)}")

    if gps_count > 0:
        print("\nSample GPS coordinates:")
        for r in exif_results:
            if r['gps_latitude'] is not None:
                print(f"  {r['filename']}: {r['gps_latitude']:.6f}, {r['gps_longitude']:.6f}")
                break  # Only show first


if __name__ == "__main__":
    main()
