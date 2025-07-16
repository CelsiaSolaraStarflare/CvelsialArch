import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import os
from gps_orientation_trainer import (
    GPSPredictionNet, GPSOrientationDataset, create_data_loaders, HaversineDistanceLoss
)
import torchvision.transforms as transforms
from datetime import datetime
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def load_model(checkpoint_path):
    """Load trained model from checkpoint"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Add safe globals for numpy
    import numpy
    torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)


    input_type = checkpoint.get('input_type', 'rgb')
    model = GPSPredictionNet(input_type=input_type).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataset_stats = checkpoint.get('dataset_stats', {})

    return model, input_type, dataset_stats


def evaluate_model(model, val_loader, dataset):
    """Comprehensive model evaluation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model.eval()

    all_predictions = {
        'gps': [],
        'time': [],
        'orientation': [],
        'altitude': []
    }

    all_targets = {
        'gps_lat': [],
        'gps_lon': [],
        'hour': [],
        'minute': [],
        'orientation': [],
        'altitude': []
    }

    gps_distances = []
    time_errors = []
    orientation_correct = []
    altitude_errors = []

    haversine_loss = HaversineDistanceLoss()

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input'].to(device)
            targets = {k: v.to(device) for k, v in batch['targets'].items()}

            # Forward pass with mixed precision if CUDA
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    predictions = model(inputs)
            else:
                predictions = model(inputs)

            # Store predictions and targets
            all_predictions['gps'].append(predictions['gps'].cpu())
            all_predictions['time'].append(predictions['time'].cpu())
            all_predictions['orientation'].append(predictions['orientation'].cpu())
            all_predictions['altitude'].append(predictions['altitude'].cpu())

            all_targets['gps_lat'].append(targets['gps_lat'].cpu())
            all_targets['gps_lon'].append(targets['gps_lon'].cpu())
            all_targets['hour'].append(targets['hour'].cpu())
            all_targets['minute'].append(targets['minute'].cpu())
            all_targets['orientation'].append(targets['orientation'].cpu())
            all_targets['altitude'].append(targets['altitude'].cpu())

            # Calculate GPS distances
            for i in range(predictions['gps'].shape[0]):
                pred_lat, pred_lon = dataset.denormalize_gps(
                    predictions['gps'][i, 0].item(), predictions['gps'][i, 1].item()
                )
                true_lat, true_lon = dataset.denormalize_gps(
                    targets['gps_lat'][i].item(), targets['gps_lon'][i].item()
                )

                pred_coords = torch.tensor([[pred_lat, pred_lon]], device=device)
                true_coords = torch.tensor([[true_lat, true_lon]], device=device)
                distance = haversine_loss(pred_coords, true_coords).item()
                gps_distances.append(distance)

            # Calculate time errors (in minutes)
            pred_time = predictions['time'].cpu()
            true_time = torch.stack([targets['hour'].cpu(), targets['minute'].cpu()], dim=1)

            for i in range(pred_time.shape[0]):
                pred_hour = pred_time[i, 0].item() * 23
                pred_minute = pred_time[i, 1].item() * 59
                true_hour = true_time[i, 0].item() * 23
                true_minute = true_time[i, 1].item() * 59

                pred_total_minutes = pred_hour * 60 + pred_minute
                true_total_minutes = true_hour * 60 + true_minute
                time_error = abs(pred_total_minutes - true_total_minutes)
                time_errors.append(time_error)

            # Calculate orientation accuracy
            pred_orientation = torch.argmax(predictions['orientation'], dim=1).cpu()
            true_orientation = targets['orientation'].cpu()
            orientation_correct.extend((pred_orientation == true_orientation).tolist())

            # Calculate altitude errors
            pred_altitude = predictions['altitude'].cpu().squeeze()
            true_altitude = targets['altitude'].cpu()
            altitude_errors.extend(torch.abs(pred_altitude - true_altitude).tolist())

    # Concatenate all predictions and targets
    all_pred_gps = torch.cat(all_predictions['gps'], dim=0)
    all_pred_orientation = torch.cat(all_predictions['orientation'], dim=0)
    all_true_orientation = torch.cat(all_targets['orientation'], dim=0)

    # Calculate metrics
    metrics = {
        'gps_distance_mean_km': np.mean(gps_distances),
        'gps_distance_std_km': np.std(gps_distances),
        'gps_distance_median_km': np.median(gps_distances),
        'time_error_mean_minutes': np.mean(time_errors),
        'time_error_std_minutes': np.std(time_errors),
        'orientation_accuracy': np.mean(orientation_correct),
        'altitude_error_mean_m': np.mean(altitude_errors),
        'altitude_error_std_m': np.std(altitude_errors)
    }

    return metrics, {
        'gps_distances': gps_distances,
        'time_errors': time_errors,
        'orientation_correct': orientation_correct,
        'altitude_errors': altitude_errors,
        'pred_orientation': torch.argmax(all_pred_orientation, dim=1),
        'true_orientation': all_true_orientation
    }


def visualize_results(metrics, detailed_results, save_dir='evaluation_plots'):
    """Create visualization plots for evaluation results"""
    os.makedirs(save_dir, exist_ok=True)

    # GPS distance distribution
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.hist(detailed_results['gps_distances'], bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('GPS Distance Error (km)')
    plt.ylabel('Frequency')
    plt.title(
        f'GPS Distance Error Distribution\nMean: {metrics["gps_distance_mean_km"]:.2f}±{metrics["gps_distance_std_km"]:.2f} km')
    plt.grid(True, alpha=0.3)

    # Time error distribution
    plt.subplot(2, 2, 2)
    plt.hist(detailed_results['time_errors'], bins=30, alpha=0.7, edgecolor='black', color='orange')
    plt.xlabel('Time Error (minutes)')
    plt.ylabel('Frequency')
    plt.title(
        f'Time Prediction Error\nMean: {metrics["time_error_mean_minutes"]:.1f}±{metrics["time_error_std_minutes"]:.1f} min')
    plt.grid(True, alpha=0.3)

    # Orientation confusion matrix
    plt.subplot(2, 2, 3)
    orientation_labels = ['Normal', 'Rotate 90°', 'Rotate 180°', 'Rotate 270°', 'Flip H', 'Flip V']
    cm = confusion_matrix(detailed_results['true_orientation'], detailed_results['pred_orientation'])
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=orientation_labels,
                yticklabels=orientation_labels, cmap='Blues')
    plt.title(f'Orientation Confusion Matrix\nAccuracy: {metrics["orientation_accuracy"]:.2f}')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Altitude error distribution
    plt.subplot(2, 2, 4)
    plt.hist(detailed_results['altitude_errors'], bins=30, alpha=0.7, edgecolor='black', color='green')
    plt.xlabel('Altitude Error (m)')
    plt.ylabel('Frequency')
    plt.title(
        f'Altitude Prediction Error\nMean: {metrics["altitude_error_mean_m"]:.1f}±{metrics["altitude_error_std_m"]:.1f} m')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_metrics.png'), dpi=300, bbox_inches='tight')
    plt.show()


def inference_single_image(model, image_path, depth_path, dataset, input_type='rgb'):
    """Run inference on a single image"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if input_type == 'rgb':
        # Load and preprocess RGB image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

    elif input_type == 'depth':
        # Load and preprocess depth data
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        if depth_path and os.path.exists(depth_path):
            depth_data = np.load(depth_path)
            # Normalize depth values to [0, 1] range
            depth_normalized = (depth_data - depth_data.min()) / (depth_data.max() - depth_data.min())
            # Convert to PIL Image for transforms
            depth_image = Image.fromarray((depth_normalized * 255).astype(np.uint8), mode='L')
            input_tensor = transform(depth_image).unsqueeze(0).to(device)
        else:
            # Fallback: create dummy depth data
            dummy_depth = Image.new('L', (224, 224), 0)
            input_tensor = transform(dummy_depth).unsqueeze(0).to(device)

    # Run inference with mixed precision if CUDA
    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.amp.autocast('cuda'):
                predictions = model(input_tensor)
        else:
            predictions = model(input_tensor)

    # Decode predictions
    pred_gps = predictions['gps'][0].cpu()
    pred_time = predictions['time'][0].cpu()
    pred_orientation = torch.argmax(predictions['orientation'][0]).item()
    pred_altitude = predictions['altitude'][0].item()

    # Denormalize GPS
    pred_lat, pred_lon = dataset.denormalize_gps(pred_gps[0].item(), pred_gps[1].item())

    # Denormalize time
    pred_hour = pred_time[0].item() * 23
    pred_minute = pred_time[1].item() * 59

    # Decode orientation
    orientation_map = {
        0: 'Normal',
        1: 'Rotate 90 CW',
        2: 'Rotate 180',
        3: 'Rotate 270 CW',
        4: 'Flip Horizontal',
        5: 'Flip Vertical'
    }

    results = {
        'gps_latitude': pred_lat,
        'gps_longitude': pred_lon,
        'hour': int(pred_hour),
        'minute': int(pred_minute),
        'orientation': orientation_map[pred_orientation],
        'altitude': pred_altitude
    }

    return results


def compare_models(rgb_model_path, depth_model_path, data_dir, exif_file):
    """Compare RGB vs Depth models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Comparing RGB vs Depth models...")

    # Load models
    rgb_model, _, rgb_stats = load_model(rgb_model_path)
    depth_model, _, depth_stats = load_model(depth_model_path)

    # Create data loaders
    _, val_loader_rgb, dataset_rgb = create_data_loaders(
        data_dir, exif_file, batch_size=8, input_type='rgb'
    )
    _, val_loader_depth, dataset_depth = create_data_loaders(
        data_dir, exif_file, batch_size=8, input_type='depth'
    )

    # Evaluate both models
    print("Evaluating RGB model...")
    metrics_rgb, _ = evaluate_model(rgb_model, val_loader_rgb, dataset_rgb)

    print("Evaluating Depth model...")
    metrics_depth, _ = evaluate_model(depth_model, val_loader_depth, dataset_depth)

    # Print comparison
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)

    print(f"GPS Distance Error (km):")
    print(f"  RGB Model:   {metrics_rgb['gps_distance_mean_km']:.3f} ± {metrics_rgb['gps_distance_std_km']:.3f}")
    print(f"  Depth Model: {metrics_depth['gps_distance_mean_km']:.3f} ± {metrics_depth['gps_distance_std_km']:.3f}")

    print(f"\nTime Error (minutes):")
    print(f"  RGB Model:   {metrics_rgb['time_error_mean_minutes']:.1f} ± {metrics_rgb['time_error_std_minutes']:.1f}")
    print(
        f"  Depth Model: {metrics_depth['time_error_mean_minutes']:.1f} ± {metrics_depth['time_error_std_minutes']:.1f}")

    print(f"\nOrientation Accuracy:")
    print(f"  RGB Model:   {metrics_rgb['orientation_accuracy']:.3f}")
    print(f"  Depth Model: {metrics_depth['orientation_accuracy']:.3f}")

    print(f"\nAltitude Error (m):")
    print(f"  RGB Model:   {metrics_rgb['altitude_error_mean_m']:.1f} ± {metrics_rgb['altitude_error_std_m']:.1f}")
    print(f"  Depth Model: {metrics_depth['altitude_error_mean_m']:.1f} ± {metrics_depth['altitude_error_std_m']:.1f}")


def main():
    """Main evaluation function"""
    DATA_DIR = "data"
    EXIF_FILE = "data/exif_data.json"

    RGB_MODEL_PATH = "best_gps_model_rgb.pth"
    DEPTH_MODEL_PATH = "best_gps_model_depth.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Check if models exist
    models_exist = {
        'rgb': os.path.exists(RGB_MODEL_PATH),
        'depth': os.path.exists(DEPTH_MODEL_PATH)
    }

    if not any(models_exist.values()):
        print("No model checkpoints found!")
        print("Please train models first using gps_orientation_trainer.py")
        return

    # Evaluate available models
    results = {}

    if models_exist['rgb']:
        print("📊 Evaluating RGB Model...")
        print("=" * 40)

        # Load RGB model
        rgb_model, input_type, dataset_stats = load_model(RGB_MODEL_PATH)

        # Create data loader
        _, val_loader_rgb, dataset_rgb = create_data_loaders(
            DATA_DIR, EXIF_FILE, batch_size=8, input_type='rgb'
        )

        # Evaluate model
        metrics_rgb, detailed_results_rgb = evaluate_model(rgb_model, val_loader_rgb, dataset_rgb)
        results['rgb'] = (metrics_rgb, detailed_results_rgb)

        # Print results
        print(f"RGB Model Results:")
        print(
            f"  GPS Distance Error: {metrics_rgb['gps_distance_mean_km']:.3f} ± {metrics_rgb['gps_distance_std_km']:.3f} km")
        print(
            f"  Time Error: {metrics_rgb['time_error_mean_minutes']:.1f} ± {metrics_rgb['time_error_std_minutes']:.1f} minutes")
        print(f"  Orientation Accuracy: {metrics_rgb['orientation_accuracy']:.3f}")
        print(
            f"  Altitude Error: {metrics_rgb['altitude_error_mean_m']:.1f} ± {metrics_rgb['altitude_error_std_m']:.1f} meters")

    if models_exist['depth']:
        print("\n📊 Evaluating Depth Model...")
        print("=" * 40)

        # Load Depth model
        depth_model, input_type, dataset_stats = load_model(DEPTH_MODEL_PATH)

        # Create data loader
        _, val_loader_depth, dataset_depth = create_data_loaders(
            DATA_DIR, EXIF_FILE, batch_size=8, input_type='depth'
        )

        # Evaluate model
        metrics_depth, detailed_results_depth = evaluate_model(depth_model, val_loader_depth, dataset_depth)
        results['depth'] = (metrics_depth, detailed_results_depth)

        # Print results
        print(f"Depth Model Results:")
        print(
            f"  GPS Distance Error: {metrics_depth['gps_distance_mean_km']:.3f} ± {metrics_depth['gps_distance_std_km']:.3f} km")
        print(
            f"  Time Error: {metrics_depth['time_error_mean_minutes']:.1f} ± {metrics_depth['time_error_std_minutes']:.1f} minutes")
        print(f"  Orientation Accuracy: {metrics_depth['orientation_accuracy']:.3f}")
        print(
            f"  Altitude Error: {metrics_depth['altitude_error_mean_m']:.1f} ± {metrics_depth['altitude_error_std_m']:.1f} meters")

    # Compare models if both exist
    if all(models_exist.values()):
        print("\n📈 Model Comparison:")
        print("=" * 40)

        gps_improvement = ((metrics_rgb['gps_distance_mean_km'] - metrics_depth['gps_distance_mean_km']) / metrics_rgb[
            'gps_distance_mean_km']) * 100
        time_improvement = ((metrics_rgb['time_error_mean_minutes'] - metrics_depth['time_error_mean_minutes']) /
                            metrics_rgb['time_error_mean_minutes']) * 100
        orientation_improvement = ((metrics_depth['orientation_accuracy'] - metrics_rgb['orientation_accuracy']) /
                                   metrics_rgb['orientation_accuracy']) * 100

        print(f"Depth vs RGB Improvements:")
        print(f"  GPS Distance: {gps_improvement:.1f}% {'improvement' if gps_improvement > 0 else 'degradation'}")
        print(f"  Time Prediction: {time_improvement:.1f}% {'improvement' if time_improvement > 0 else 'degradation'}")
        print(
            f"  Orientation: {orientation_improvement:.1f}% {'improvement' if orientation_improvement > 0 else 'degradation'}")

        # Visualize results for both models
        print("\nGenerating comparison plots...")
        for model_type, (metrics, detailed_results) in results.items():
            print(f"Generating plots for {model_type.upper()} model...")
            visualize_results(metrics, detailed_results, save_dir=f'evaluation_plots_{model_type}')

    # Test single image inference
    print("\n🔍 Testing Single Image Inference...")
    test_image = os.path.join(DATA_DIR, "images", "IMG_1141.JPG")
    test_depth = os.path.join(DATA_DIR, "depths", "IMG_1141_depth_raw.npy")

    if os.path.exists(test_image):
        for model_type in ['rgb', 'depth']:
            if models_exist[model_type]:
                model_path = RGB_MODEL_PATH if model_type == 'rgb' else DEPTH_MODEL_PATH
                model, input_type, _ = load_model(model_path)
                _, _, dataset = create_data_loaders(DATA_DIR, EXIF_FILE, batch_size=8, input_type=input_type)

                results = inference_single_image(model, test_image, test_depth, dataset, input_type)
                print(f"\n{model_type.upper()} Model prediction for {os.path.basename(test_image)}:")
                print(f"  GPS: {results['gps_latitude']:.6f}, {results['gps_longitude']:.6f}")
                print(f"  Time: {results['hour']:02d}:{results['minute']:02d}")
                print(f"  Orientation: {results['orientation']}")
                print(f"  Altitude: {results['altitude']:.1f}m")


if __name__ == "__main__":
    main()