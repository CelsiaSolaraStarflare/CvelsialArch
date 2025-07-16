import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import json
import numpy as np
from PIL import Image
import os
from datetime import datetime
import math
from typing import Tuple, Dict, Any
import torch.nn.functional as F
from scipy.cluster.vq import kmeans, vq


class GPSDataset(Dataset):
    def __init__(self, data_dir: str, exif_file: str, transform=None, input_type='rgb'):
        self.data_dir = data_dir
        self.input_type = input_type  # 'rgb' or 'depth'
        self.transform = transform

        with open(exif_file, 'r') as f:
            self.exif_data = json.load(f)

        self.image_dir = os.path.join(data_dir, 'images')
        self.depth_dir = os.path.join(data_dir, 'depths')

        # Filter out entries without valid GPS data
        self.valid_data = [item for item in self.exif_data
                           if item['gps_latitude'] is not None and item['gps_longitude'] is not None]

        # Calculate normalization stats for GPS coordinates
        lats = [item['gps_latitude'] for item in self.valid_data]
        lons = [item['gps_longitude'] for item in self.valid_data]
        self.lat_mean, self.lat_std = np.mean(lats), np.std(lats)
        self.lon_mean, self.lon_std = np.mean(lons), np.std(lons)

        # Cluster GPS locations
        gps_points = np.array([[item['gps_latitude'], item['gps_longitude']] for item in self.valid_data])
        num_samples = len(gps_points)
        self.num_clusters = min(num_samples, 100)  # Reduced from 1000
        if self.num_clusters > 1:
            centroids, _ = kmeans(gps_points.astype(np.float32), self.num_clusters)
            cluster_labels, _ = vq(gps_points.astype(np.float32), centroids)
        else:
            cluster_labels = np.zeros(num_samples, dtype=int)
        self.cluster_labels = cluster_labels

        print(f"Dataset initialized with {len(self.valid_data)} valid samples")
        print(f"Input type: {input_type.upper()}")
        print(f"GPS stats - Lat: {self.lat_mean:.6f}±{self.lat_std:.6f}, Lon: {self.lon_mean:.6f}±{self.lon_std:.6f}")
        print(f"Number of clusters: {self.num_clusters}")

    def __len__(self):
        return len(self.valid_data)

    def normalize_gps(self, lat: float, lon: float) -> Tuple[float, float]:
        """Normalize GPS coordinates to have mean 0 and std 1"""
        norm_lat = (lat - self.lat_mean) / self.lat_std
        norm_lon = (lon - self.lon_mean) / self.lon_std
        return norm_lat, norm_lon

    def denormalize_gps(self, norm_lat: float, norm_lon: float) -> Tuple[float, float]:
        """Convert normalized GPS back to actual coordinates"""
        lat = norm_lat * self.lat_std + self.lat_mean
        lon = norm_lon * self.lon_std + self.lon_mean
        return lat, lon

    def __getitem__(self, idx):
        item = self.valid_data[idx]

        if self.input_type == 'rgb':
            # Load RGB image
            img_path = os.path.join(self.image_dir, item['filename'])
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            input_data = image

        elif self.input_type == 'depth':
            # Load depth data as image
            depth_filename = item['filename'].replace('.JPG', '_depth_raw.npy')
            depth_path = os.path.join(self.depth_dir, depth_filename)

            if os.path.exists(depth_path):
                depth_raw = np.load(depth_path)
                # Robust depth normalization
                valid_mask = (depth_raw > 0) & (depth_raw < np.inf)
                if valid_mask.any():
                    valid_depths = depth_raw[valid_mask]
                    # Use percentiles for robustness
                    p1, p99 = np.percentile(valid_depths, [1, 99])
                    depth_clipped = np.clip(depth_raw, p1, p99)
                    depth_normalized = (depth_clipped - p1) / (p99 - p1)
                else:
                    depth_normalized = np.zeros_like(depth_raw)
                
                # Convert to PIL Image for transforms
                depth_image = Image.fromarray((depth_normalized * 255).astype(np.uint8), mode='L')

                if self.transform:
                    # Apply transforms (this will convert to tensor)
                    input_data = self.transform(depth_image)
                else:
                    # Convert to tensor manually if no transforms
                    input_data = torch.from_numpy(depth_normalized).float().unsqueeze(0)
            else:
                # Fallback: create dummy depth data
                if self.transform:
                    dummy_depth = Image.new('L', (224, 224), 0)
                    input_data = self.transform(dummy_depth)
                else:
                    input_data = torch.zeros(1, 224, 224)

        # Prepare targets - GPS only
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


class GPSPredictionNet(nn.Module):
    def __init__(self, input_type='rgb', num_clusters=100):
        super(GPSPredictionNet, self).__init__()
        self.input_type = input_type
        self.num_clusters = num_clusters

        # Single backbone - ResNet50
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if input_type == 'rgb' else None)

        # Modify first conv layer for depth input
        if input_type == 'depth':
            original_conv1 = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if hasattr(original_conv1, 'weight'):
                with torch.no_grad():
                    self.backbone.conv1.weight = nn.Parameter(
                        original_conv1.weight.mean(dim=1, keepdim=True)
                    )

        backbone_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove final classification layer

        # Shared feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(backbone_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # GPS head with better initialization
        self.gps_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # lat, lon
        )
        
        # Initialize GPS head with small values
        with torch.no_grad():
            self.gps_head[-1].weight.data.mul_(0.01)
            self.gps_head[-1].bias.data.zero_()

        # Cluster head (optional auxiliary task)
        self.cluster_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_clusters)
        )

    def forward(self, input_tensor):
        # Extract features from input (RGB or depth)
        features = self.backbone(input_tensor)

        # Process features
        processed_features = self.feature_processor(features)

        # Generate predictions
        gps_pred = self.gps_head(processed_features)
        cluster_pred = self.cluster_head(processed_features)

        return {
            'gps': gps_pred,  # [lat, lon] normalized
            'cluster': cluster_pred,  # logits for cluster classification
        }


class HaversineDistanceLoss(nn.Module):
    """Custom loss for GPS coordinates using Haversine distance"""

    def __init__(self, earth_radius_km=6371.0):
        super(HaversineDistanceLoss, self).__init__()
        self.earth_radius = earth_radius_km

    def forward(self, pred_coords, true_coords):
        """
        pred_coords: [batch, 2] - predicted [lat, lon] in degrees
        true_coords: [batch, 2] - true [lat, lon] in degrees
        """
        # Convert to radians
        pred_lat, pred_lon = pred_coords[:, 0] * math.pi / 180, pred_coords[:, 1] * math.pi / 180
        true_lat, true_lon = true_coords[:, 0] * math.pi / 180, true_coords[:, 1] * math.pi / 180

        # Haversine formula
        dlat = pred_lat - true_lat
        dlon = pred_lon - true_lon

        a = torch.sin(dlat / 2) ** 2 + torch.cos(true_lat) * torch.cos(pred_lat) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))
        distance_km = self.earth_radius * c

        return torch.mean(distance_km)


class GPSLoss(nn.Module):
    def __init__(self, lat_mean, lat_std, lon_mean, lon_std, weights=None):
        super(GPSLoss, self).__init__()
        # Store normalization parameters as buffers
        self.register_buffer('lat_mean', torch.tensor(lat_mean))
        self.register_buffer('lat_std', torch.tensor(lat_std))
        self.register_buffer('lon_mean', torch.tensor(lon_mean))
        self.register_buffer('lon_std', torch.tensor(lon_std))
        
        self.haversine_loss = HaversineDistanceLoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        # Weights for different loss components
        if weights is None:
            self.weights = {
                'gps_mse': 0.3,      # MSE on normalized coordinates
                'gps_haversine': 0.5, # Haversine distance
                'cluster': 0.2        # Cluster classification
            }
        else:
            self.weights = weights

    def denormalize_gps_batch(self, norm_coords):
        """Vectorized denormalization"""
        lat = norm_coords[:, 0] * self.lat_std + self.lat_mean
        lon = norm_coords[:, 1] * self.lon_std + self.lon_mean
        return torch.stack([lat, lon], dim=1)

    def forward(self, predictions, targets):
        # GPS predictions and targets
        pred_gps = predictions['gps']
        true_gps = torch.stack([targets['gps_lat'], targets['gps_lon']], dim=1)

        # MSE loss on normalized coordinates
        gps_mse_loss = self.mse_loss(pred_gps, true_gps)

        # Denormalize for Haversine calculation
        pred_gps_denorm = self.denormalize_gps_batch(pred_gps)
        true_gps_denorm = self.denormalize_gps_batch(true_gps)

        # Haversine distance loss
        gps_haversine_loss = self.haversine_loss(pred_gps_denorm, true_gps_denorm)

        # Cluster loss
        cluster_loss = self.ce_loss(predictions['cluster'], targets['cluster'])

        # Combine losses
        total_loss = (
            self.weights['gps_mse'] * gps_mse_loss +
            self.weights['gps_haversine'] * gps_haversine_loss +
            self.weights['cluster'] * cluster_loss
        )

        return {
            'total_loss': total_loss,
            'gps_mse_loss': gps_mse_loss,
            'gps_haversine_loss': gps_haversine_loss,
            'cluster_loss': cluster_loss
        }


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def create_data_loaders(data_dir: str, exif_file: str, batch_size=8,
                        val_split=0.2, input_type='rgb', distributed=False, rank=0, world_size=1):
    """Create training and validation data loaders"""

    # Data transforms for RGB
    if input_type == 'rgb':
        train_transform = transforms.Compose([
            transforms.Resize((512, 512)),  # Reduced from 2048
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),  # Reduced from 70
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Data transforms for depth
    elif input_type == 'depth':
        train_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    # Create full dataset
    full_dataset = GPSDataset(data_dir, exif_file, transform=train_transform, input_type=input_type)

    # Split into train/val
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # Apply validation transforms to validation set
    val_dataset.dataset.transform = val_transform

    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    # Optimal workers for H20 GPUs
    num_workers = 4 if os.name != 'nt' else 0

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            sampler=val_sampler, num_workers=num_workers, pin_memory=True,
                            persistent_workers=(num_workers > 0))

    return train_loader, val_loader, full_dataset


def train_model(rank, world_size, data_dir: str, exif_file: str, epochs=50, batch_size=8,
                learning_rate=1e-4, input_type='rgb'):
    """Main training function for GPS prediction only"""

    setup_ddp(rank, world_size)

    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    # Enable TF32 for H20 GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Create data loaders
    train_loader, val_loader, dataset = create_data_loaders(
        data_dir, exif_file, batch_size, input_type=input_type, distributed=True, rank=rank, world_size=world_size
    )

    # Initialize model
    model = GPSPredictionNet(input_type=input_type, num_clusters=dataset.num_clusters).to(device)
    model = DDP(model, device_ids=[rank])

    # Enable mixed precision training
    scaler = torch.amp.GradScaler('cuda')

    # Initialize loss with dataset stats
    criterion = GPSLoss(
        dataset.lat_mean, dataset.lat_std, 
        dataset.lon_mean, dataset.lon_std
    ).to(device)
    
    # Optimizer with reduced learning rate
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate*0.5, weight_decay=1e-4)
    
    # Learning rate warmup
    warmup_epochs = 5
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, 
        total_iters=warmup_epochs * len(train_loader)
    )
    main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-6
    )

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    if rank == 0:
        print(f"Starting training for {epochs} epochs...")
        print(f"Input type: {input_type.upper()}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss_sum = 0.0
        train_gps_loss_sum = 0.0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            inputs = batch['input'].to(device)
            targets = {k: v.to(device) for k, v in batch['targets'].items()}

            # Forward pass with mixed precision
            with torch.amp.autocast('cuda'):
                predictions = model(inputs)
                loss_dict = criterion(predictions, targets)
                loss = loss_dict['total_loss']

            # Backward pass with gradient clipping
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Learning rate warmup
            if epoch < warmup_epochs:
                warmup_scheduler.step()

            train_loss_sum += loss.item()
            train_gps_loss_sum += loss_dict['gps_haversine_loss'].item()

            if rank == 0 and batch_idx % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f} '
                      f'(GPS: {loss_dict["gps_haversine_loss"].item():.2f} km, '
                      f'MSE: {loss_dict["gps_mse_loss"].item():.4f})')

        # Average train loss across all processes
        train_loss_tensor = torch.tensor(train_loss_sum / len(train_loader)).to(device)
        train_gps_tensor = torch.tensor(train_gps_loss_sum / len(train_loader)).to(device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_gps_tensor, op=dist.ReduceOp.SUM)
        avg_train_loss = train_loss_tensor.item() / world_size
        avg_train_gps = train_gps_tensor.item() / world_size
        
        if rank == 0:
            train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        val_gps_loss_sum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = {k: v.to(device) for k, v in batch['targets'].items()}

                with torch.amp.autocast('cuda'):
                    predictions = model(inputs)
                    loss_dict = criterion(predictions, targets)

                val_loss_sum += loss_dict['total_loss'].item()
                val_gps_loss_sum += loss_dict['gps_haversine_loss'].item()

        # Average val loss across all processes
        val_loss_tensor = torch.tensor(val_loss_sum / len(val_loader)).to(device)
        val_gps_tensor = torch.tensor(val_gps_loss_sum / len(val_loader)).to(device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_gps_tensor, op=dist.ReduceOp.SUM)
        avg_val_loss = val_loss_tensor.item() / world_size
        avg_val_gps = val_gps_tensor.item() / world_size
        
        if rank == 0:
            val_losses.append(avg_val_loss)

            print(f'Epoch {epoch + 1}/{epochs}: '
                  f'Train Loss: {avg_train_loss:.4f} (GPS: {avg_train_gps:.2f} km), '
                  f'Val Loss: {avg_val_loss:.4f} (GPS: {avg_val_gps:.2f} km)')

            # Learning rate scheduling
            if epoch >= warmup_epochs:
                main_scheduler.step(avg_val_loss)

            # Save best model (only on rank 0)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_gps_error_km': avg_val_gps,
                    'input_type': input_type,
                    'dataset_stats': {
                        'lat_mean': dataset.lat_mean,
                        'lat_std': dataset.lat_std,
                        'lon_mean': dataset.lon_mean,
                        'lon_std': dataset.lon_std,
                        'num_clusters': dataset.num_clusters
                    }
                }, f'best_gps_model_{input_type}.pth')
                print(f'New best {input_type.upper()} model saved with val loss: {best_val_loss:.4f} '
                      f'(GPS error: {avg_val_gps:.2f} km)')

    cleanup_ddp()
    return model, train_losses, val_losses if rank == 0 else None


def main():
    # Configuration
    DATA_DIR = "/root/autodl-tmp/Contour/data"  # Using fast disk on AutoDL
    EXIF_FILE = "/root/autodl-tmp/Contour/data/exif_data.json"
    world_size = torch.cuda.device_count()  # Number of GPUs

    print(f"Found {world_size} GPUs")
    print("🚀 Training RGB Model (GPS only)...")
    print("=" * 50)
    mp.spawn(train_model,
             args=(world_size, DATA_DIR, EXIF_FILE, 50, 32, 5e-5, 'rgb'),  # Increased batch size
             nprocs=world_size,
             join=True)

    print("\n🚀 Training Depth Model (GPS only)...")
    print("=" * 50)
    mp.spawn(train_model,
             args=(world_size, DATA_DIR, EXIF_FILE, 50, 32, 5e-5, 'depth'),
             nprocs=world_size,
             join=True)

    print("\n✅ Training completed!")
    print(f"RGB model saved as: best_gps_model_rgb.pth")
    print(f"Depth model saved as: best_gps_model_depth.pth")

if __name__ == "__main__":
    main()