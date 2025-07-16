# GPS Location, Orientation & Time Prediction from Images + Depth

This project implements a deep learning system that predicts GPS coordinates, orientation, altitude, and time of day from RGB images and depth maps using neural networks.

## Project Overview

The system uses computer vision and deep learning to extract spatial and temporal information from photographs. It combines RGB image data with depth information to make predictions about:

- GPS coordinates (latitude/longitude)
- Image orientation
- Altitude above sea level
- Time of day when photo was taken

## Key Features

- Multi-modal neural network architecture (RGB + Depth)
- Custom Haversine distance loss for GPS coordinate prediction
- Multi-task learning with shared feature extraction
- Support for both RGB-only and RGB+Depth input modes
- Comprehensive evaluation metrics and visualizations
- Interactive inference capabilities

## Project Structure

### Core Python Files:
- `exif_extractor.py` - Extracts EXIF metadata from images including GPS coordinates
- `depth_map_generator.py` - Generates depth maps using Depth-Anything-V2 model
- `gps_orientation_trainer.py` - Main training script for GPS prediction models
- `evaluate_gps_model.py` - Model evaluation and performance analysis
- `density.py` - Additional utility functions

### Jupyter Notebooks:
- `GPS_Prediction_Workflow.ipynb` - Complete workflow demonstration
- `GPS_RGB_vs_Depth_Workflow.ipynb` - Comparison between RGB and RGB+Depth models

### Data Structure:
```
data/
├── images/          # Original RGB images
├── depths/          # Generated depth maps (.png and .npy files)
├── exif_data.json   # Extracted EXIF metadata
└── exif_data.csv    # EXIF data in CSV format
```

### Model Files:
- `best_gps_model.pth` - Main trained model
- `best_gps_model_rgb.pth` - RGB-only model
- `best_gps_model_depth.pth` - RGB+Depth model

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install Depth-Anything-V2 (for depth map generation):
```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
# Follow their installation instructions
```

3. Download pre-trained Depth-Anything-V2 weights:
- Place model weights in `checkpoints/` directory
- Required: `depth_anything_v2_vits.pth`

## Usage

### 1. Extract EXIF Data from Images
```bash
python exif_extractor.py
```
- Place your images in `data/images/` directory
- Extracts GPS coordinates, timestamps, orientation, and camera metadata
- Outputs `exif_data.json` and `exif_data.csv`

### 2. Generate Depth Maps
```bash
python depth_map_generator.py
```
- Processes all images in `data/images/`
- Generates depth maps in `data/depths/`
- Creates both visualization (.png) and raw data (.npy) files

### 3. Train GPS Prediction Models
```bash
python gps_orientation_trainer.py
```
- Trains both RGB-only and RGB+Depth models
- Uses multi-task learning for GPS, time, orientation, and altitude prediction
- Saves best models as `.pth` files

### 4. Evaluate Models
```bash
python evaluate_gps_model.py
```
- Loads trained models and evaluates performance
- Generates comprehensive metrics and visualizations
- Creates performance comparison reports

### 5. Interactive Workflow (Recommended)
Open and run `GPS_Prediction_Workflow.ipynb` for a complete guided workflow with:
- Data exploration and visualization
- Model training with live progress monitoring
- Performance comparison between RGB-only and RGB+Depth models
- Interactive inference examples

## Model Architecture

### Multi-Modal GPS Prediction Network:
- **Backbone**: ResNet50 feature extractors
  - RGB branch: 3-channel input (pretrained ImageNet weights)
  - Depth branch: 1-channel input (adapted from RGB weights)
- **Feature Fusion**: Concatenation of RGB and depth features
- **Task-Specific Heads**:
  - GPS Head: 2 outputs (latitude, longitude)
  - Time Head: 2 outputs (hour, minute)
  - Orientation Head: 6 classes (Normal, Rotated 90°/180°/270°, Flipped H/V)
  - Altitude Head: 1 regression output

### Loss Functions:
- **GPS Loss**: Custom Haversine distance for geographic accuracy
- **Time Loss**: Mean Squared Error on normalized time values
- **Orientation Loss**: Cross-Entropy for classification
- **Altitude Loss**: Mean Squared Error for regression

## Dataset Requirements

### Image Requirements:
- Format: JPEG, PNG, BMP, TIFF
- Resolution: Any (automatically resized to 1024x1024 for training)
- EXIF data required with GPS coordinates and timestamps

### Minimum Dataset Size:
- At least 20-30 images for basic functionality
- 100+ images recommended for good performance
- 1000+ images for production-quality results

## Performance Metrics

The system evaluates performance using:
- **GPS Distance Error**: Haversine distance in kilometers
- **Time Prediction Error**: Absolute difference in minutes
- **Orientation Accuracy**: Classification accuracy percentage
- **Altitude Error**: Absolute difference in meters

## Hardware Requirements

### Minimum:
- 8GB RAM
- CPU-only training supported (slower)

### Recommended:
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- CUDA-compatible PyTorch installation

## Training Configuration

### Default Hyperparameters:
- Batch Size: 4-8 (adjust based on GPU memory)
- Learning Rate: 1e-4
- Epochs: 30-50
- Optimizer: AdamW with weight decay
- Image Size: 1024x1024
- Mixed Precision: Enabled for CUDA training

## Limitations

1. **Dataset Dependency**: Requires images with GPS EXIF data
2. **Geographic Scope**: Best performance on similar geographic regions
3. **Temporal Scope**: Time prediction accuracy depends on lighting conditions
4. **Hardware**: GPU recommended for reasonable training times
5. **Depth Quality**: Depth prediction quality affects RGB+Depth model performance

## Potential Applications

- **Geotagging**: Automatic location tagging for photos without GPS
- **Digital Forensics**: Image provenance and location verification
- **Surveillance**: Automated location identification from camera feeds
- **Robotics**: Visual navigation and localization
- **Augmented Reality**: Scene understanding and positioning

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**: Reduce batch size in training scripts
2. **Missing Depth Maps**: Run `depth_map_generator.py` first
3. **No GPS Data**: Ensure images have GPS EXIF metadata
4. **Import Errors**: Install all requirements and check Depth-Anything-V2 setup

### Performance Issues:
- Use GPU for training (CPU training is very slow)
- Ensure sufficient RAM for data loading
- Consider reducing image resolution for faster training

## Contributing

1. Ensure all images have GPS EXIF data
2. Test changes with small dataset first
3. Maintain compatibility with both RGB-only and RGB+Depth modes
4. Update documentation for any API changes

## License

This project is for educational and research purposes. Please respect the licenses of:
- Depth-Anything-V2 model and weights
- PyTorch and other dependencies
- Any images used for training (ensure you have rights to use them)

## Acknowledgments

- Depth-Anything-V2 team for the depth estimation model
- PyTorch team for the deep learning framework
- Contributors to the various Python libraries used

For questions or issues, please refer to the code comments or create an issue in the project repository.