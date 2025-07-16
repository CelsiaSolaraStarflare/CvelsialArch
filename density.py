import pandas as pd
import folium
from folium.plugins import HeatMap

# Load the data
df = pd.read_csv('data/exif_data.csv')

# Optional: Drop rows with missing or invalid coordinates
df = df.dropna(subset=['gps_latitude', 'gps_longitude'])

# Prepare the data for HeatMap: list of [lat, lon] pairs
heat_data = df[['gps_latitude', 'gps_longitude']].values.tolist()

# Create a folium map centered around the average location
map_center = [df['gps_latitude'].mean(), df['gps_longitude'].mean()]
m = folium.Map(location=map_center, zoom_start=10)

# Add the HeatMap layer
HeatMap(heat_data).add_to(m)

# Save map to HTML file or display in Jupyter Notebook
m.save("heatmap.html")
