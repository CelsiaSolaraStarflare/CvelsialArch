import streamlit as st
import torch
import folium
import numpy as np
from PIL import Image
from torch import nn
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights, ResNet50_Weights
import tempfile
import os
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Improved High-End GPS Location Predictor", 
    page_icon="🌍", 
    layout="wide"
)

LAT_MIN, LAT_MAX = 37.301, 37.900
LON_MIN, LON_MAX = -122.478, -121.902

class ProperTerrainClassifier(nn.Module):
    """Dedicated terrain classifier using ResNet backbone"""
    def __init__(self):
        super().__init__()
        
        # Use ResNet50 as backbone for terrain classification
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Replace final classification layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Terrain classification head
        self.terrain_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 6),  # 6 terrain classes: urban, water, forest, grassland, mountain, beach
            nn.Softmax(dim=1)
        )
        
        # Color analysis for basic terrain detection
        self.terrain_classes = ['urban', 'water', 'forest', 'grassland', 'mountain', 'beach']
        
    def analyze_color_features(self, image_tensor):
        """Analyze color characteristics to help classify terrain"""
        # Convert to numpy for analysis (assume batch size)
        batch_size = image_tensor.shape[0]
        device = image_tensor.device
        scores = torch.zeros((batch_size, 6), device=device)
        
        for b in range(batch_size):
            img_np = image_tensor[b].permute(1, 2, 0).cpu().numpy()
            
            # Color analysis
            mean_rgb = np.mean(img_np, axis=(0, 1))
            
            # Water detection (blue dominant)
            if mean_rgb[2] > mean_rgb[0] and mean_rgb[2] > mean_rgb[1]:
                scores[b, 1] += 0.3  # water
                
            # Forest detection (green dominant)
            if mean_rgb[1] > mean_rgb[0] and mean_rgb[1] > mean_rgb[2]:
                scores[b, 2] += 0.3  # forest
                
            # Urban detection (gray/mixed colors)
            color_variance = np.var(mean_rgb)
            if color_variance < 0.05:
                scores[b, 0] += 0.2  # urban
                
            # Grassland (moderate green)
            if 0.3 < mean_rgb[1] < 0.7 and mean_rgb[1] > mean_rgb[2]:
                scores[b, 3] += 0.2  # grassland
                
            # Mountain (brown/gray tones)
            if np.all(mean_rgb > 0.3) and np.all(mean_rgb < 0.7):
                scores[b, 4] += 0.2  # mountain
                
            # Beach (light colors, sandy)
            if np.mean(mean_rgb) > 0.6:
                scores[b, 5] += 0.2  # beach
                
        return scores
        
    def forward(self, x):
        # Extract features using ResNet
        features = self.backbone(x)
        
        # Terrain classification
        terrain_logits = self.terrain_head(features)
        
        # Enhance with color analysis
        color_scores = self.analyze_color_features(x)
        
        # Combine neural network prediction with color analysis
        combined_scores = 0.8 * terrain_logits + 0.2 * color_scores
        
        # Normalize to probabilities
        terrain_probs = torch.softmax(combined_scores, dim=1)
        
        return terrain_probs

class FoliumTerrainValidator:
    """Enhanced geographic validator with better SF Bay Area mapping"""
    def __init__(self):
        # More precise SF Bay Area geographic features
        self.water_areas = {
            'pacific_ocean': {'lat_range': (37.3, 37.9), 'lon_range': (-122.52, -122.46)},
            'sf_bay': {'lat_range': (37.4, 37.85), 'lon_range': (-122.45, -122.2)},
            'richardson_bay': {'lat_range': (37.85, 37.89), 'lon_range': (-122.52, -122.48)}
        }
        
        self.urban_areas = {
            'sf_downtown': {'lat_range': (37.77, 37.82), 'lon_range': (-122.42, -122.38)},
            'mission': {'lat_range': (37.74, 37.77), 'lon_range': (-122.42, -122.40)},
            'oakland_downtown': {'lat_range': (37.79, 37.83), 'lon_range': (-122.28, -122.26)},
            'berkeley': {'lat_range': (37.86, 37.88), 'lon_range': (-122.27, -122.25)},
            'south_bay_urban': {'lat_range': (37.33, 37.45), 'lon_range': (-122.1, -121.95)}
        }
        
        self.forest_areas = {
            'golden_gate_park': {'lat_range': (37.76, 37.77), 'lon_range': (-122.51, -122.45)},
            'presidio': {'lat_range': (37.79, 37.81), 'lon_range': (-122.47, -122.44)},
            'marin_headlands': {'lat_range': (37.82, 37.87), 'lon_range': (-122.52, -122.47)},
            'tilden_park': {'lat_range': (37.89, 37.92), 'lon_range': (-122.25, -122.22)}
        }
        
        self.mountain_areas = {
            'twin_peaks': {'lat_range': (37.75, 37.76), 'lon_range': (-122.45, -122.44)},
            'mount_davidson': {'lat_range': (37.73, 37.74), 'lon_range': (-122.45, -122.44)},
            'hills_peninsula': {'lat_range': (37.35, 37.5), 'lon_range': (-122.35, -122.2)}
        }
        
        self.beach_areas = {
            'ocean_beach': {'lat_range': (37.72, 37.78), 'lon_range': (-122.51, -122.50)},
            'baker_beach': {'lat_range': (37.79, 37.80), 'lon_range': (-122.48, -122.47)},
            'crissy_field': {'lat_range': (37.80, 37.81), 'lon_range': (-122.47, -122.45)}
        }
        
    def get_actual_terrain_at_location(self, lat, lon):
        """Get actual terrain type at coordinates using detailed geographic mapping"""
        
        # Check water areas first (highest priority)
        for area_name, bounds in self.water_areas.items():
            if (bounds['lat_range'][0] <= lat <= bounds['lat_range'][1] and 
                bounds['lon_range'][0] <= lon <= bounds['lon_range'][1]):
                return 'water'
        
        # Check beach areas
        for area_name, bounds in self.beach_areas.items():
            if (bounds['lat_range'][0] <= lat <= bounds['lat_range'][1] and 
                bounds['lon_range'][0] <= lon <= bounds['lon_range'][1]):
                return 'beach'
        
        # Check urban areas
        for area_name, bounds in self.urban_areas.items():
            if (bounds['lat_range'][0] <= lat <= bounds['lat_range'][1] and 
                bounds['lon_range'][0] <= lon <= bounds['lon_range'][1]):
                return 'urban'
        
        # Check forest areas  
        for area_name, bounds in self.forest_areas.items():
            if (bounds['lat_range'][0] <= lat <= bounds['lat_range'][1] and 
                bounds['lon_range'][0] <= lon <= bounds['lon_range'][1]):
                return 'forest'
                
        # Check mountain areas
        for area_name, bounds in self.mountain_areas.items():
            if (bounds['lat_range'][0] <= lat <= bounds['lat_range'][1] and 
                bounds['lon_range'][0] <= lon <= bounds['lon_range'][1]):
                return 'mountain'
        
        # Default classification based on general SF geography
        if lon < -122.42:  # Western SF - more residential/mixed
            return 'grassland'  # Residential areas with grass/parks
        elif 37.45 < lat < 37.7 and -122.2 < lon < -121.95:  # South Bay
            return 'urban'
        else:
            return 'grassland'  # Default to grassland for residential areas
    
    def validate_prediction(self, predicted_terrain, actual_lat, actual_lon):
        """Check if predicted terrain matches actual location"""
        actual_terrain = self.get_actual_terrain_at_location(actual_lat, actual_lon)
        
        # Define incompatible combinations (high penalty)
        incompatible = {
            'water': ['urban', 'mountain', 'grassland', 'forest'],
            'urban': ['water', 'beach'],
            'mountain': ['water', 'beach'],
            'beach': ['urban', 'mountain', 'forest']
        }
        
        # Check for major incompatibilities
        if actual_terrain in incompatible:
            if predicted_terrain in incompatible[actual_terrain]:
                return 'incompatible'  # Heavy penalty
        
        # Check for exact matches (reward)
        if predicted_terrain == actual_terrain:
            return 'exact_match'  # Reward
            
        # Partial matches (neutral)
        partial_matches = {
            'grassland': ['urban', 'forest'],
            'urban': ['grassland'],
            'forest': ['grassland', 'mountain'],
            'mountain': ['forest', 'grassland']
        }
        
        if actual_terrain in partial_matches:
            if predicted_terrain in partial_matches[actual_terrain]:
                return 'partial_match'
        
        return 'mismatch'  # Small penalty

class ImprovedHighEndGPSNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # GPS prediction backbone (EfficientNet for location)
        self.gps_backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Get EfficientNet features
        for layer in self.gps_backbone.classifier:
            if isinstance(layer, nn.Linear):
                gps_features = layer.in_features
                break
        else:
            gps_features = 1280
            
        self.gps_backbone.classifier = nn.Identity()
        
        # Terrain classifier (separate ResNet-based) - IMPROVED VERSION
        self.terrain_classifier = ProperTerrainClassifier()
        
        # Geographic reasoning module (combines GPS and terrain features)
        self.geo_reasoner = nn.Sequential(
            nn.Linear(gps_features + 6, 512),  # GPS features + 6 terrain classes
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )
        
        # Separate coordinate prediction heads
        self.lat_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
        self.lon_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),  # lat_confidence, lon_confidence
            nn.Sigmoid()  # 0-1 confidence scores
        )

    def forward(self, x):
        # Extract GPS-relevant features
        gps_features = self.gps_backbone(x)
        
        # Get terrain classification (IMPROVED terrain classifier)
        terrain_probs = self.terrain_classifier(x)
        
        # Combine GPS and terrain features for geographic reasoning
        combined_input = torch.cat([gps_features, terrain_probs], dim=1)
        geo_features = self.geo_reasoner(combined_input)
        
        # Coordinate prediction
        lat = self.lat_head(geo_features)
        lon = self.lon_head(geo_features)
        coords = torch.cat([lat, lon], dim=1)
        
        # Confidence estimation
        confidence = self.confidence_head(geo_features)
        
        return coords, terrain_probs, confidence

def denormalize_coordinates(norm_lat, norm_lon):
    lat = (norm_lat + 1.0) / 2.0 * (LAT_MAX - LAT_MIN) + LAT_MIN
    lon = (norm_lon + 1.0) / 2.0 * (LON_MAX - LON_MIN) + LON_MIN
    return lat, lon

@st.cache_resource
def load_improved_high_end_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedHighEndGPSNet().to(device)
    
    model_path = 'improved_high_end_gps_model.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different model save formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'Unknown')
        else:
            # If checkpoint is just the state dict directly
            model.load_state_dict(checkpoint)
            epoch = 'Unknown'
            
        model.eval()
        return model, device, epoch
    else:
        st.error(f"Improved model file '{model_path}' not found. Please train the model first with main2_high_improved.py")
        return None, None, None

def predict_improved_location(model, device, image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    x = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_coords, pred_terrain, pred_confidence = model(x)
        
        norm_lat, norm_lon = pred_coords.squeeze().tolist()
        lat, lon = denormalize_coordinates(norm_lat, norm_lon)
        
        # Get terrain prediction (using the 6 classes from ProperTerrainClassifier)
        terrain_classes = ['urban', 'water', 'forest', 'grassland', 'mountain', 'beach']
        terrain_probs = pred_terrain.squeeze()
        predicted_terrain_idx = torch.argmax(terrain_probs).item()
        predicted_terrain = terrain_classes[predicted_terrain_idx]
        terrain_confidence = terrain_probs[predicted_terrain_idx].item()
        
        # Get coordinate confidence
        lat_conf, lon_conf = pred_confidence.squeeze().tolist()
        
        return lat, lon, predicted_terrain, terrain_confidence, (lat_conf, lon_conf), terrain_probs

def create_advanced_map_with_comparison(lat, lon, terrain_info=None, confidence_info=None, validation_status=None):
    # Create map with satellite view option
    m = folium.Map(location=[lat, lon], zoom_start=15)
    
    # Add multiple tile layers for terrain analysis
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='Street Map',
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite',
        name='Satellite View',
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
        attr='Google Terrain',
        name='Terrain View',
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=h&x={x}&y={y}&z={z}',
        attr='Google Hybrid',
        name='Hybrid View',
        control=True
    ).add_to(m)
    
    # Color-code marker based on validation status
    marker_color = 'red'  # Default
    if validation_status == 'exact_match':
        marker_color = 'green'
    elif validation_status == 'partial_match':
        marker_color = 'orange'
    elif validation_status == 'incompatible':
        marker_color = 'darkred'
    
    # Main prediction marker with enhanced popup
    popup_text = f"<b>🌍 GPS Prediction (Improved AI)</b><br>"
    popup_text += f"<b>📍 Coordinates:</b> {lat:.6f}, {lon:.6f}<br>"
    if terrain_info:
        popup_text += f"<b>🏞️ Predicted Terrain:</b> {terrain_info['type']} ({terrain_info['confidence']:.1%})<br>"
    if confidence_info:
        popup_text += f"<b>📊 GPS Confidence:</b> Lat {confidence_info[0]:.1%}, Lon {confidence_info[1]:.1%}<br>"
    popup_text += f"<b>✅ Validation:</b> {validation_status.replace('_', ' ').title()}"
    
    folium.Marker(
        [lat, lon], 
        tooltip="Improved High-End GPS Prediction",
        popup=popup_text,
        icon=folium.Icon(color=marker_color, icon='crosshairs', prefix='fa')
    ).add_to(m)
    
    # Add confidence circle with enhanced styling
    if confidence_info:
        # Use average confidence to determine radius
        avg_confidence = (confidence_info[0] + confidence_info[1]) / 2
        # Lower confidence = larger radius
        radius_m = (1 - avg_confidence) * 800  # Max 800m radius for low confidence
        
        circle_color = 'blue'
        if validation_status == 'exact_match':
            circle_color = 'green'
        elif validation_status == 'incompatible':
            circle_color = 'red'
        elif validation_status == 'partial_match':
            circle_color = 'orange'
        
        folium.Circle(
            location=[lat, lon],
            radius=radius_m,
            popup=f"<b>🎯 Uncertainty Zone</b><br>Radius: {radius_m:.0f}m<br>Avg Confidence: {avg_confidence:.1%}<br>Model Quality: Improved",
            color=circle_color,
            fillColor=circle_color,
            fillOpacity=0.15,
            weight=2
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def main():
    st.title("🌍 Improved High-End GPS Location Predictor")
    st.markdown("**Advanced AI with Enhanced Terrain Detection and Geographic Intelligence**")
    
    model, device, epoch = load_improved_high_end_model()
    
    if model is None:
        st.stop()
    
    # Enhanced sidebar with model info
    st.sidebar.success(f"✅ Improved Model Loaded (Epoch {epoch})")
    st.sidebar.info(f"🎯 Target Area: San Francisco Bay Area")
    st.sidebar.info(f"📍 Lat: [{LAT_MIN:.3f}, {LAT_MAX:.3f}]")
    st.sidebar.info(f"📍 Lon: [{LON_MIN:.3f}, {LON_MAX:.3f}]")
    
    # Model improvements info
    with st.sidebar.expander("🔬 Model Improvements"):
        st.markdown("""
        **Enhanced Features:**
        - ✅ ResNet50 terrain classifier
        - ✅ Color analysis integration
        - ✅ Improved geographic validation
        - ✅ Enhanced loss functions
        - ✅ Better diversity enforcement
        - ✅ Multi-layer satellite mapping
        """)
    
    # Initialize session state for prediction results
    if 'improved_prediction_results' not in st.session_state:
        st.session_state.improved_prediction_results = None
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📷 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image taken in the San Francisco Bay Area"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Analysis options
            st.subheader("⚙️ Analysis Options")
            show_details = st.checkbox("Show detailed analysis", value=True)
            compare_satellite = st.checkbox("Compare with satellite imagery", value=True)
            
            if st.button("🚀 Predict Location (Improved AI)", type="primary"):
                with st.spinner("🧠 Analyzing with improved high-end AI..."):
                    try:
                        result = predict_improved_location(model, device, image)
                        lat, lon, predicted_terrain, terrain_confidence, coord_confidence, terrain_probs = result
                        
                        # Geographic validation
                        geo_validator = FoliumTerrainValidator()
                        actual_terrain = geo_validator.get_actual_terrain_at_location(lat, lon)
                        validation = geo_validator.validate_prediction(predicted_terrain, lat, lon)
                        
                        # Store results in session state
                        st.session_state.improved_prediction_results = {
                            'lat': lat,
                            'lon': lon,
                            'predicted_terrain': predicted_terrain,
                            'terrain_confidence': terrain_confidence,
                            'coord_confidence': coord_confidence,
                            'actual_terrain': actual_terrain,
                            'validation': validation,
                            'terrain_probs': terrain_probs.tolist(),
                            'show_details': show_details,
                            'compare_satellite': compare_satellite
                        }
                        
                        st.success("✅ Improved AI analysis complete!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
    
    # Display results if available
    if st.session_state.improved_prediction_results is not None:
        results = st.session_state.improved_prediction_results
        lat = results['lat']
        lon = results['lon']
        predicted_terrain = results['predicted_terrain']
        terrain_confidence = results['terrain_confidence']
        coord_confidence = results['coord_confidence']
        actual_terrain = results['actual_terrain']
        validation = results['validation']
        terrain_probs = results['terrain_probs']
        show_details = results.get('show_details', True)
        
        with col2:
            st.subheader("📍 Improved AI Prediction Results")
            
            # GPS Coordinates with enhanced styling
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Latitude", f"{lat:.6f}°", 
                         delta=f"±{coord_confidence[0]*100:.1f}% conf.")
            with col2b:
                st.metric("Longitude", f"{lon:.6f}°", 
                         delta=f"±{coord_confidence[1]*100:.1f}% conf.")
            
            # Enhanced Terrain Analysis
            st.subheader("🏞️ Advanced Terrain Analysis")
            
            col3a, col3b, col3c = st.columns(3)
            with col3a:
                st.metric("🤖 AI Prediction", predicted_terrain, 
                         delta=f"{terrain_confidence:.1%} confident")
            with col3b:
                st.metric("🗺️ Expected Type", actual_terrain)
            with col3c:
                # Validation with color coding
                validation_icons = {
                    'exact_match': '✅',
                    'partial_match': '🟡', 
                    'mismatch': '🟠',
                    'incompatible': '❌'
                }
                icon = validation_icons.get(validation, '❓')
                st.metric("🧠 AI Accuracy", f"{icon} {validation.replace('_', ' ').title()}")
            
            # Geographic Validation with enhanced feedback
            validation_color = {
                'exact_match': 'success',
                'partial_match': 'info', 
                'mismatch': 'warning',
                'incompatible': 'error'
            }
            
            validation_message = {
                'exact_match': '🎯 Perfect! AI correctly identified the terrain type.',
                'partial_match': '👍 Good! AI prediction is acceptable for this location.',
                'mismatch': '⚠️ Minor discrepancy between AI prediction and expected terrain.', 
                'incompatible': '🚨 Geographic impossibility detected! AI needs improvement for this case.'
            }
            
            if validation in validation_color:
                getattr(st, validation_color[validation])(validation_message[validation])
            
            if show_details:
                # Enhanced Terrain Probability Distribution
                st.subheader("📊 Detailed Terrain Classification")
                terrain_classes = ['urban', 'water', 'forest', 'grassland', 'mountain', 'beach']
                
                # Create a more detailed view
                terrain_data = {}
                terrain_details = {}
                for i, terrain_class in enumerate(terrain_classes):
                    prob = terrain_probs[i]
                    terrain_data[terrain_class] = prob
                    terrain_details[terrain_class] = f"{prob:.1%}"
                
                # Show bar chart
                st.bar_chart(terrain_data)
                
                # Show detailed percentages
                cols = st.columns(len(terrain_classes))
                for i, (terrain, percentage) in enumerate(terrain_details.items()):
                    with cols[i]:
                        st.metric(terrain.title(), percentage)
            
            # Bounds checking
            if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
                st.warning("⚠️ Predicted coordinates are outside the expected SF Bay Area bounds")
            
            st.subheader("🗺️ Interactive Map with Satellite Comparison")
            
            # Create advanced map with satellite layers
            terrain_info = {
                'type': predicted_terrain,
                'confidence': terrain_confidence
            }
            
            map_obj = create_advanced_map_with_comparison(lat, lon, terrain_info, coord_confidence, validation)
            st_folium(map_obj, width=700, height=500, key="improved_prediction_map")
            
            if results.get('compare_satellite', False):
                st.info("💡 **Tip:** Switch between map layers above to compare AI prediction with satellite imagery and verify terrain type!")
            
            # Clear results button
            col_clear1, col_clear2 = st.columns([1, 3])
            with col_clear1:
                if st.button("🔄 Clear Results"):
                    st.session_state.improved_prediction_results = None
                    st.rerun()
            with col_clear2:
                st.caption("Clear results to analyze a new image")
    
    with st.sidebar:
        st.subheader("ℹ️ About Improved Model")
        st.markdown("""
        **🚀 Next-Generation Features:**
        
        **🧠 Enhanced AI:**
        - ResNet50 terrain classifier
        - EfficientNet GPS backbone
        - Color analysis integration
        - Geographic reasoning module
        
        **🗺️ Advanced Mapping:**
        - 4 map layer types
        - Satellite imagery comparison
        - Real-time validation
        - SF Bay Area expertise
        
        **🎯 Smart Validation:**
        - Impossibility detection
        - Confidence scoring
        - Terrain-location matching
        - Multi-modal verification
        
        **✨ Key Improvements:**
        - No more "always beach" errors
        - Better terrain diversity
        - Enhanced geographic intelligence
        - Visual verification tools
        """)

if __name__ == "__main__":
    main()