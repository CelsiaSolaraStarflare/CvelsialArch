import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from scipy.cluster.vq import kmeans, vq
import json
import numpy as np
from PIL import Image
import os
import math
import random
from typing import Tuple
import torch.serialization
import folium  # Added for map plotting

# GPSDataset (unchanged)
class GPSDataset:
    def __init__(self, data_dir: str, exif_file: str, transform=None, input_type='rgb'):
        self.data_dir = data_dir
        self.input_type = input_type
        self.transform = transform
        with open(exif_file, 'r') as f:
            self.exif_data = json.load(f)
        self.image_dir = os.path.join(data_dir, 'images')
        self.depth_dir = os.path.join(data_dir, 'depths')
        self.valid_data = [item for item in self.exif_data
                           if item['gps_latitude'] is not None and item['gps_longitude'] is not None]
        lats = [item['gps_latitude'] for item in self.valid_data]
        lons = [item['gps_longitude'] for item in self.valid_data]
        self.lat_mean, self.lat_std = np.mean(lats), np.std(lats)
        self.lon_mean, self.lon_std = np.mean(lons), np.std(lons)
        gps_points = np.array([[item['gps_latitude'], item['gps_longitude']] for item in self.valid_data])
        num_samples = len(gps_points)
        self.num_clusters = min(num_samples, 100)
        if self.num_clusters > 1:
            centroids, _ = kmeans(gps_points.astype(np.float32), self.num_clusters)
            cluster_labels, _ = vq(gps_points.astype(np.float32), centroids)
        else:
            cluster_labels = np.zeros(num_samples, dtype=int)
        self.cluster_labels = cluster_labels
        print(f"Dataset initialized with {len(self.valid_data)} valid samples")
        print(f"Input type: {self.input_type.upper()}")
        print(f"GPS stats - Lat: {self.lat_mean:.6f}±{self.lat_std:.6f}, Lon: {self.lon_mean:.6f}±{self.lon_std:.6f}")
        print(f"Number of clusters: {self.num_clusters}")

    def __len__(self):
        return len(self.valid_data)

    def normalize_gps(self, lat: float, lon: float) -> Tuple[float, float]:
        norm_lat = (lat - self.lat_mean) / self.lat_std
        norm_lon = (lon - self.lon_mean) / self.lon_std
        return norm_lat, norm_lon

    def denormalize_gps(self, norm_lat: float, norm_lon: float) -> Tuple[float, float]:
        lat = norm_lat * self.lat_std + self.lat_mean
        lon = norm_lon * self.lon_std + self.lon_mean
        return lat, lon

    def __getitem__(self, idx):
        item = self.valid_data[idx]
        if self.input_type == 'rgb':
            img_path = os.path.join(self.image_dir, item['filename'])
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            input_data = image
        elif self.input_type == 'depth':
            depth_filename = item['filename'].replace('.JPG', '_depth_raw.npy')
            depth_path = os.path.join(self.depth_dir, depth_filename)
            if os.path.exists(depth_path):
                depth_raw = np.load(depth_path)
                valid_mask = (depth_raw > 0) & (depth_raw < np.inf)
                if valid_mask.any():
                    valid_depths = depth_raw[valid_mask]
                    p1, p99 = np.percentile(valid_depths, [1, 99])
                    depth_clipped = np.clip(depth_raw, p1, p99)
                    depth_normalized = (depth_clipped - p1) / (p99 - p1)
                else:
                    depth_normalized = np.zeros_like(depth_raw)
                depth_image = Image.fromarray((depth_normalized * 255).astype(np.uint8), mode='L')
                if self.transform:
                    input_data = self.transform(depth_image)
                else:
                    input_data = torch.from_numpy(depth_normalized).float().unsqueeze(0)
            else:
                if self.transform:
                    dummy_depth = Image.new('L', (224, 224), 0)
                    input_data = self.transform(dummy_depth)
                else:
                    input_data = torch.zeros(1, 224, 224)
        norm_lat, norm_lon = self.normalize_gps(item['gps_latitude'], item['gps_longitude'])
        targets = {
            'gps_lat': torch.tensor(norm_lat, dtype=torch.float32),
            'gps_lon': torch.tensor(norm_lon, dtype=torch.float32),
            'cluster': torch.tensor(self.cluster_labels[idx], dtype=torch.long)
        }
        return {
            'input': input_data,
            'targets': targets,
            'filename': item['filename']
        }

# GPSPredictionNet (unchanged)
class GPSPredictionNet(nn.Module):
    def __init__(self, input_type='rgb', num_clusters=100):
        super(GPSPredictionNet, self).__init__()
        self.input_type = input_type
        self.num_clusters = num_clusters
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if input_type == 'rgb' else None)
        if self.input_type == 'depth':
            original_conv1 = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if hasattr(original_conv1, 'weight'):
                with torch.no_grad():
                    self.backbone.conv1.weight = nn.Parameter(
                        original_conv1.weight.mean(dim=1, keepdim=True)
                    )
        backbone_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.feature_processor = nn.Sequential(
            nn.Linear(backbone_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.gps_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )
        with torch.no_grad():
            self.gps_head[-1].weight.data.mul_(0.01)
            self.gps_head[-1].bias.data.zero_()
        self.cluster_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_clusters)
        )

    def forward(self, input_tensor):
        features = self.backbone(input_tensor)
        processed_features = self.feature_processor(features)
        gps_pred = self.gps_head(processed_features)
        cluster_pred = self.cluster_head(processed_features)
        return {
            'gps': gps_pred,
            'cluster': cluster_pred,
        }

# Haversine distance function (unchanged)
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float, earth_radius_km=6371.0) -> float:
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    distance_km = earth_radius_km * c
    return distance_km

# Create validation dataset (unchanged)
def create_val_dataset(data_dir: str, exif_file: str, val_split=0.2, input_type='rgb'):
    if input_type == 'rgb':
        val_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif input_type == 'depth':
        val_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    temp_dataset = GPSDataset(data_dir, exif_file, transform=None, input_type=input_type)
    dataset_size = len(temp_dataset)
    indices = list(range(dataset_size))
    random.seed(42)
    random.shuffle(indices)
    split = int(val_split * dataset_size)
    val_idx = indices[:split]
    val_valid_data = [temp_dataset.valid_data[i] for i in val_idx]
    val_cluster_labels = temp_dataset.cluster_labels[val_idx]
    val_dataset = GPSDataset(data_dir, exif_file, transform=val_transform, input_type=input_type)
    val_dataset.valid_data = val_valid_data
    val_dataset.cluster_labels = val_cluster_labels
    val_dataset.num_clusters = len(set(val_cluster_labels))
    return val_dataset, temp_dataset

# Placeholder for visualize_results (unchanged)
def visualize_results(metrics, detailed_results, save_dir, dataset=None):
    print(f"Visualizing results for {save_dir}...")
    print(f"Metrics: {metrics}")
    print(f"Number of clusters: {dataset.num_clusters if dataset else 'unknown'}")
    pass

# Function to create Folium map
def create_folium_map(detailed_results, save_path, center_lat, center_lon):
    # Create map centered on mean ground truth coordinates
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
    # Add markers and lines
    for result in detailed_results:
        filename = result['filename']
        true_lat, true_lon = result['true_lat'], result['true_lon']
        pred_lat, pred_lon = result['pred_lat'], result['pred_lon']
        distance = result['distance']
        
        # Ground truth marker (green)
        folium.Marker(
            location=[true_lat, true_lon],
            popup=f"{filename}\nGround Truth\nLat: {true_lat:.6f}\nLon: {true_lon:.6f}",
            icon=folium.Icon(color='green')
        ).add_to(m)
        
        # Predicted marker (red)
        folium.Marker(
            location=[pred_lat, pred_lon],
            popup=f"{filename}\nPredicted\nLat: {pred_lat:.6f}\nLon: {pred_lon:.6f}\nError: {distance:.2f} km",
            icon=folium.Icon(color='red')
        ).add_to(m)
        
        # Line between ground truth and predicted
        folium.PolyLine(
            locations=[[true_lat, true_lon], [pred_lat, pred_lon]],
            color='blue',
            weight=2,
            popup=f"Distance: {distance:.2f} km"
        ).add_to(m)
    
    # Save map
    m.save(save_path)
    print(f"Folium map saved to {save_path}")

# Updated evaluate_model function
def evaluate_model(input_type='rgb', threshold_km=10.0, sample_fraction=0.08):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating {input_type.upper()} model on device: {device}")

    # Load checkpoint with allowlisted NumPy globals
    checkpoint_path = f'best_gps_model_{input_type}.pth'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    torch.serialization.add_safe_globals(['numpy._core.multiarray.scalar'])
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"Failed to load with weights_only=True: {e}")
        print("Falling back to weights_only=False (ensure checkpoint is trusted)")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    stats = checkpoint['dataset_stats']
    model = GPSPredictionNet(input_type=input_type, num_clusters=stats['num_clusters']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']} with best val loss {checkpoint['best_val_loss']:.4f}")

    # Create validation dataset
    DATA_DIR = "data"
    EXIF_FILE = "data/exif_data.json"
    val_dataset, temp_dataset = create_val_dataset(DATA_DIR, EXIF_FILE, val_split=0.2, input_type=input_type)

    if len(val_dataset) < 2:
        raise ValueError("Validation dataset has fewer than 2 samples.")

    # Select 8% of validation dataset
    sample_size = max(2, int(len(val_dataset) * sample_fraction))
    indices = random.sample(range(len(val_dataset)), sample_size)
    samples = [val_dataset[i] for i in indices]

    print(f"Evaluating {sample_size} samples ({sample_fraction*100:.1f}% of validation set)")

    # Collect metrics for visualization
    metrics = {'haversine_distances': [], 'correct_predictions': []}
    detailed_results = []

    # Process samples
    true_lats, true_lons = [], []
    for sample in samples:
        filename = sample['filename']
        input_data = sample['input'].unsqueeze(0).to(device)
        targets = sample['targets']

        with torch.no_grad():
            predictions = model(input_data)
            pred_gps_norm = predictions['gps'][0]

        pred_lat, pred_lon = val_dataset.denormalize_gps(pred_gps_norm[0].item(), pred_gps_norm[1].item())
        true_lat, true_lon = val_dataset.denormalize_gps(targets['gps_lat'].item(), targets['gps_lon'].item())
        distance = haversine_distance(pred_lat, pred_lon, true_lat, true_lon)
        is_correct = distance <= threshold_km

        print(f"\nFilename: {filename}")
        print(f"Predicted GPS: ({pred_lat:.6f}, {pred_lon:.6f})")
        print(f"Ground Truth GPS: ({true_lat:.6f}, {true_lon:.6f})")
        print(f"Distance Error: {distance:.2f} km")
        print(f"Correct (within {threshold_km} km): {'Yes' if is_correct else 'No'}")

        metrics['haversine_distances'].append(distance)
        metrics['correct_predictions'].append(is_correct)
        detailed_results.append({
            'filename': filename,
            'pred_lat': pred_lat,
            'pred_lon': pred_lon,
            'true_lat': true_lat,
            'true_lon': true_lon,
            'distance': distance,
            'is_correct': is_correct
        })
        true_lats.append(true_lat)
        true_lons.append(true_lon)

    # Create Folium map
    center_lat = np.mean(true_lats) if true_lats else 0.0
    center_lon = np.mean(true_lons) if true_lons else 0.0
    map_save_path = f'evaluation_map_{input_type}.html'
    create_folium_map(detailed_results, map_save_path, center_lat, center_lon)

    # Call existing visualize_results
    visualize_results(metrics, detailed_results, save_dir=f'evaluation_plots_{input_type}', dataset=temp_dataset)

# Main function
def main():
    print("Evaluating RGB Model...")
    evaluate_model(input_type='rgb', threshold_km=10.0, sample_fraction=0.08)
    print("\nEvaluating Depth Model...")
    evaluate_model(input_type='depth', threshold_km=10.0, sample_fraction=0.08)
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()