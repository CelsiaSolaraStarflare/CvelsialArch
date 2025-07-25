import os
import json
import torch
import numpy as np
from PIL import Image
from torch import nn
from tqdm import tqdm
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights
import time
import math
from sklearn.cluster import KMeans
import pickle

# Global coordinate bounds
LAT_MIN, LAT_MAX = -90.0, 90.0
LON_MIN, LON_MAX = -180.0, 180.0

print(f"GPS bounds: Lat [{LAT_MIN}, {LAT_MAX}], Lon [{LON_MIN}, {LON_MAX}]")


# ===== STAGE 1: CLUSTER CLASSIFICATION MODEL =====

class ClusterDataset(Dataset):
    """Dataset for training cluster classification"""
    def __init__(self, image_dir, json_path, cluster_model_path=None, is_training=True):
        self.image_dir = image_dir
        self.is_training = is_training
        
        with open(json_path, 'r') as f:
            data_dict = json.load(f)
        
        self.image_filenames = []
        self.raw_coords = []
        
        # Load valid images
        for fname, coords in data_dict.items():
            lat = coords.get("latitude")
            lon = coords.get("longitude")
            if lat is not None and lon is not None:
                if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:
                    full_path = os.path.join(self.image_dir, fname)
                    if os.path.isfile(full_path):
                        try:
                            with Image.open(full_path) as img:
                                img.verify()
                            self.image_filenames.append(full_path)
                            self.raw_coords.append([lat, lon])
                        except:
                            continue
        
        # Create or load clusters
        if cluster_model_path and os.path.exists(cluster_model_path):
            with open(cluster_model_path, 'rb') as f:
                cluster_data = pickle.load(f)
                self.kmeans = cluster_data['kmeans']
                self.cluster_centers = cluster_data['centers']
                self.n_clusters = cluster_data['n_clusters']
            print(f"Loaded cluster model with {self.n_clusters} clusters")
        else:
            # Adaptive clustering based on data density
            self.n_clusters = min(200, max(20, len(self.raw_coords) // 50))
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            self.kmeans.fit(self.raw_coords)
            self.cluster_centers = self.kmeans.cluster_centers_
            
            if cluster_model_path:
                with open(cluster_model_path, 'wb') as f:
                    pickle.dump({
                        'kmeans': self.kmeans,
                        'centers': self.cluster_centers,
                        'n_clusters': self.n_clusters
                    }, f)
                print(f"Created and saved cluster model with {self.n_clusters} clusters")
        
        # Assign cluster labels
        self.cluster_labels = []
        for lat, lon in self.raw_coords:
            cluster_id = self.kmeans.predict([[lat, lon]])[0]
            self.cluster_labels.append(cluster_id)
        
        # Transforms for cluster classification
        if is_training:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        print(f"Loaded {len(self.image_filenames)} images for cluster classification")
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_filenames[idx]).convert('RGB')
        img = self.transform(img)
        cluster_label = torch.tensor(self.cluster_labels[idx], dtype=torch.long)
        actual_coords = torch.tensor(self.raw_coords[idx], dtype=torch.float)
        return img, cluster_label, actual_coords


class ClusterClassifier(nn.Module):
    """Stage 1: Classify images into geographic clusters"""
    def __init__(self, n_clusters):
        super().__init__()
        self.n_clusters = n_clusters
        
        # Use ResNet50 backbone
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        num_features = self.backbone.fc.in_features
        
        # Replace final layer with cluster classification
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, n_clusters)
        )
        
    def forward(self, x):
        cluster_logits = self.backbone(x)
        return cluster_logits


def train_cluster_classifier():
    """Train Stage 1: Cluster Classification"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== STAGE 1: Training Cluster Classifier on {device} ===")
    
    # Paths
    image_dir = 'autodl-tmp/data/images_global'
    json_path = 'autodl-tmp/data/exif_data.json'
    cluster_model_path = 'gps_clusters.pkl'
    
    # Create dataset
    full_dataset = ClusterDataset(image_dir, json_path, cluster_model_path, is_training=True)
    if len(full_dataset) == 0:
        print("No images found")
        return None
    
    # Split dataset
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    model = ClusterClassifier(n_clusters=full_dataset.n_clusters).to(device)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    # Training loop
    epochs = 30
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        
        for imgs, cluster_labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device)
            cluster_labels = cluster_labels.to(device)
            
            optimizer.zero_grad()
            cluster_logits = model(imgs)
            loss = F.cross_entropy(cluster_logits, cluster_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (cluster_logits.argmax(dim=1) == cluster_labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        
        with torch.no_grad():
            for imgs, cluster_labels, _ in val_loader:
                imgs = imgs.to(device)
                cluster_labels = cluster_labels.to(device)
                
                cluster_logits = model(imgs)
                loss = F.cross_entropy(cluster_logits, cluster_labels)
                
                val_loss += loss.item()
                val_correct += (cluster_logits.argmax(dim=1) == cluster_labels).sum().item()
        
        # Metrics
        train_acc = train_correct / len(train_dataset)
        val_acc = val_correct / len(val_dataset)
        
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")
        
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'n_clusters': full_dataset.n_clusters,
                'cluster_centers': full_dataset.cluster_centers,
                'epoch': epoch,
                'val_acc': val_acc
            }, 'cluster_classifier.pth')
            print(f"  Saved new best model (val_acc={val_acc:.3f})")
    
    print(f"Stage 1 complete! Best validation accuracy: {best_val_acc:.3f}")
    return model, full_dataset


# ===== STAGE 2: MICROSHIFT PREDICTION MODEL =====

class MicroshiftDataset(Dataset):
    """Dataset for training microshift prediction within clusters"""
    def __init__(self, image_dir, json_path, cluster_model_path, cluster_classifier_path, is_training=True):
        self.image_dir = image_dir
        self.is_training = is_training
        
        # Load cluster model
        with open(cluster_model_path, 'rb') as f:
            cluster_data = pickle.load(f)
            self.kmeans = cluster_data['kmeans']
            self.cluster_centers = cluster_data['centers']
            self.n_clusters = cluster_data['n_clusters']
        
        # Load cluster classifier
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cluster_classifier = ClusterClassifier(self.n_clusters).to(self.device)
        checkpoint = torch.load(cluster_classifier_path)
        self.cluster_classifier.load_state_dict(checkpoint['model_state_dict'])
        self.cluster_classifier.eval()
        
        # Load data
        with open(json_path, 'r') as f:
            data_dict = json.load(f)
        
        self.image_filenames = []
        self.raw_coords = []
        self.relative_positions = []
        self.cluster_labels = []
        
        # Process images
        for fname, coords in data_dict.items():
            lat = coords.get("latitude")
            lon = coords.get("longitude")
            if lat is not None and lon is not None:
                if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:
                    full_path = os.path.join(self.image_dir, fname)
                    if os.path.isfile(full_path):
                        try:
                            with Image.open(full_path) as img:
                                img.verify()
                            
                            # Get cluster assignment
                            cluster_id = self.kmeans.predict([[lat, lon]])[0]
                            center_lat, center_lon = self.cluster_centers[cluster_id]
                            
                            # Compute relative position
                            rel_lat = lat - center_lat
                            rel_lon = lon - center_lon
                            
                            self.image_filenames.append(full_path)
                            self.raw_coords.append([lat, lon])
                            self.relative_positions.append([rel_lat, rel_lon])
                            self.cluster_labels.append(cluster_id)
                        except:
                            continue
        
        # Transforms - minimal augmentation to preserve location cues
        if is_training:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        print(f"Loaded {len(self.image_filenames)} images for microshift prediction")
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_filenames[idx]).convert('RGB')
        img_tensor = self.transform(img)
        
        # Get predicted cluster from Stage 1 model
        with torch.no_grad():
            cluster_logits = self.cluster_classifier(img_tensor.unsqueeze(0).to(self.device))
            predicted_cluster = cluster_logits.argmax(dim=1).item()
        
        true_cluster = torch.tensor(self.cluster_labels[idx], dtype=torch.long)
        relative_pos = torch.tensor(self.relative_positions[idx], dtype=torch.float)
        actual_coords = torch.tensor(self.raw_coords[idx], dtype=torch.float)
        
        return img_tensor, predicted_cluster, true_cluster, relative_pos, actual_coords


class MicroshiftPredictor(nn.Module):
    """Stage 2: Predict relative position within a cluster"""
    def __init__(self, n_clusters):
        super().__init__()
        self.n_clusters = n_clusters
        
        # Use EfficientNet for fine-grained location features
        self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Get feature dimension
        for layer in self.backbone.classifier:
            if isinstance(layer, nn.Linear):
                feature_dim = layer.in_features
                break
        else:
            feature_dim = 1280
            
        self.backbone.classifier = nn.Identity()
        
        # Feature processor
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Cluster-specific microshift heads
        self.microshift_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 2),
                nn.Tanh()  # Bounded output
            ) for _ in range(n_clusters)
        ])
        
        # Global fallback head
        self.global_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            nn.Tanh()
        )
        
    def forward(self, x, cluster_id):
        # Extract features
        features = self.backbone(x)
        processed_features = self.feature_processor(features)
        
        # Get microshift based on cluster
        batch_size = x.shape[0]
        microshifts = []
        
        for i in range(batch_size):
            if 0 <= cluster_id[i] < self.n_clusters:
                # Use cluster-specific head
                shift = self.microshift_heads[cluster_id[i]](processed_features[i:i+1])
            else:
                # Fallback to global head
                shift = self.global_head(processed_features[i:i+1])
            microshifts.append(shift)
        
        microshifts = torch.cat(microshifts, dim=0)
        
        # Scale to reasonable range (±3 degrees)
        microshifts = microshifts * 3.0
        
        return microshifts


def train_microshift_predictor():
    """Train Stage 2: Microshift Prediction"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== STAGE 2: Training Microshift Predictor on {device} ===")
    
    # Paths
    image_dir = 'autodl-tmp/data/images_global'
    json_path = 'autodl-tmp/data/exif_data.json'
    cluster_model_path = 'gps_clusters.pkl'
    cluster_classifier_path = 'cluster_classifier.pth'
    
    # Check if Stage 1 model exists
    if not os.path.exists(cluster_classifier_path):
        print("Error: Stage 1 cluster classifier not found. Train it first!")
        return None
    
    # Create dataset
    dataset = MicroshiftDataset(image_dir, json_path, cluster_model_path, cluster_classifier_path, is_training=True)
    if len(dataset) == 0:
        print("No images found")
        return None
    
    # Split dataset
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    model = MicroshiftPredictor(n_clusters=dataset.n_clusters).to(device)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    
    # Training loop
    epochs = 40
    best_val_error = float('inf')
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371.0  # km
        lat1_rad = lat1 * (math.pi / 180.0)
        lon1_rad = lon1 * (math.pi / 180.0)
        lat2_rad = lat2 * (math.pi / 180.0)
        lon2_rad = lon2 * (math.pi / 180.0)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        return R * c
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_within_100km = 0
        train_total = 0
        
        for imgs, pred_clusters, true_clusters, relative_pos, actual_coords in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device)
            relative_pos = relative_pos.to(device)
            actual_coords = actual_coords.to(device)
            
            # Convert pred_clusters to tensor
            pred_clusters = torch.tensor(pred_clusters, dtype=torch.long)
            
            optimizer.zero_grad()
            
            # Predict microshift
            microshifts = model(imgs, pred_clusters)
            
            # Loss: MSE on relative positions
            loss = F.mse_loss(microshifts, relative_pos)
            
            # Add penalty for predictions > 100km from actual
            batch_size = imgs.shape[0]
            for i in range(batch_size):
                if pred_clusters[i] < dataset.n_clusters:
                    center = torch.tensor(dataset.cluster_centers[pred_clusters[i]], device=device)
                    predicted_coords = center + microshifts[i]
                    
                    distance = haversine_distance(
                        predicted_coords[0], predicted_coords[1],
                        actual_coords[i, 0], actual_coords[i, 1]
                    )
                    
                    if distance > 100:
                        loss += (distance - 100) * 0.01  # Penalty for > 100km
                    else:
                        train_within_100km += 1
                    train_total += 1
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_distances = []
        val_within_100km = 0
        
        with torch.no_grad():
            for imgs, pred_clusters, true_clusters, relative_pos, actual_coords in val_loader:
                imgs = imgs.to(device)
                relative_pos = relative_pos.to(device)
                actual_coords = actual_coords.to(device)
                
                pred_clusters = torch.tensor(pred_clusters, dtype=torch.long)
                
                microshifts = model(imgs, pred_clusters)
                loss = F.mse_loss(microshifts, relative_pos)
                val_loss += loss.item()
                
                # Calculate actual distances
                batch_size = imgs.shape[0]
                for i in range(batch_size):
                    if pred_clusters[i] < dataset.n_clusters:
                        center = torch.tensor(dataset.cluster_centers[pred_clusters[i]], device=device)
                        predicted_coords = center + microshifts[i]
                        
                        distance = haversine_distance(
                            predicted_coords[0], predicted_coords[1],
                            actual_coords[i, 0], actual_coords[i, 1]
                        ).item()
                        
                        val_distances.append(distance)
                        if distance < 100:
                            val_within_100km += 1
        
        # Metrics
        avg_val_distance = np.mean(val_distances) if val_distances else float('inf')
        val_within_100km_pct = val_within_100km / len(val_distances) if val_distances else 0
        train_within_100km_pct = train_within_100km / train_total if train_total > 0 else 0
        
        print(f"Epoch {epoch+1}: "
              f"Train Loss={train_loss/len(train_loader):.4f}, "
              f"Val Dist={avg_val_distance:.1f}km, "
              f"Within 100km: Train={train_within_100km_pct:.3f}, Val={val_within_100km_pct:.3f}")
        
        scheduler.step()
        
        # Save best model
        if avg_val_distance < best_val_error:
            best_val_error = avg_val_distance
            torch.save({
                'model_state_dict': model.state_dict(),
                'n_clusters': dataset.n_clusters,
                'cluster_centers': dataset.cluster_centers,
                'epoch': epoch,
                'val_error': avg_val_distance,
                'within_100km': val_within_100km_pct
            }, 'microshift_predictor.pth')
            print(f"  Saved new best model (avg_error={avg_val_distance:.1f}km)")
    
    print(f"Stage 2 complete! Best validation error: {best_val_error:.1f}km")
    return model


# ===== INFERENCE: COMBINE BOTH MODELS =====

def predict_location(image_path, cluster_classifier_path='cluster_classifier.pth', 
                    microshift_predictor_path='microshift_predictor.pth'):
    """Use both models to predict GPS location"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    cluster_checkpoint = torch.load(cluster_classifier_path)
    n_clusters = cluster_checkpoint['n_clusters']
    cluster_centers = cluster_checkpoint['cluster_centers']
    
    cluster_model = ClusterClassifier(n_clusters).to(device)
    cluster_model.load_state_dict(cluster_checkpoint['model_state_dict'])
    cluster_model.eval()
    
    microshift_checkpoint = torch.load(microshift_predictor_path)
    microshift_model = MicroshiftPredictor(n_clusters).to(device)
    microshift_model.load_state_dict(microshift_checkpoint['model_state_dict'])
    microshift_model.eval()
    
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Stage 1: Predict cluster
        cluster_logits = cluster_model(img_tensor)
        cluster_probs = F.softmax(cluster_logits, dim=1)
        predicted_cluster = cluster_logits.argmax(dim=1).item()
        cluster_confidence = cluster_probs[0, predicted_cluster].item()
        
        # Stage 2: Predict microshift
        microshift = microshift_model(img_tensor, [predicted_cluster])
        
        # Combine predictions
        center_lat, center_lon = cluster_centers[predicted_cluster]
        delta_lat, delta_lon = microshift[0].cpu().numpy()
        
        final_lat = center_lat + delta_lat
        final_lon = center_lon + delta_lon
        
        print(f"\nPrediction Results:")
        print(f"  Stage 1 - Cluster: {predicted_cluster} (confidence: {cluster_confidence:.3f})")
        print(f"  Stage 1 - Cluster center: {center_lat:.4f}, {center_lon:.4f}")
        print(f"  Stage 2 - Microshift: {delta_lat:.4f}, {delta_lon:.4f}")
        print(f"  Final prediction: {final_lat:.6f}, {final_lon:.6f}")
        
        return final_lat, final_lon, predicted_cluster, cluster_confidence


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'stage1':
        # Train only Stage 1
        model, dataset = train_cluster_classifier()
    elif len(sys.argv) > 1 and sys.argv[1] == 'stage2':
        # Train only Stage 2
        model = train_microshift_predictor()
    else:
        # Train both stages
        print("Training both stages sequentially...")
        
        # Stage 1
        model1, dataset = train_cluster_classifier()
        
        if model1:
            # Stage 2
            model2 = train_microshift_predictor()
            
            if model2 and dataset and len(dataset.image_filenames) > 0:
                # Test on a sample
                sample_img = dataset.image_filenames[0]
                actual_lat, actual_lon = dataset.raw_coords[0]
                
                print(f"\nTesting on: {os.path.basename(sample_img)}")
                print(f"Actual GPS: {actual_lat:.6f}, {actual_lon:.6f}")
                
                predicted_lat, predicted_lon, cluster, confidence = predict_location(sample_img)
                
                # Calculate error
                from math import radians, sin, cos, sqrt, atan2
                R = 6371.0
                lat1, lon1 = radians(actual_lat), radians(actual_lon)
                lat2, lon2 = radians(predicted_lat), radians(predicted_lon)
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                distance = R * c
                
                print(f"Prediction error: {distance:.1f} km")
