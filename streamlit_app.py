import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import joblib
from io import StringIO

st.set_page_config(page_title="FireVizz - Wildfire Detection System", layout="wide")
st.title("🔥 FireVizz - Wildfire Detection System")
st.markdown("""
Upload your sensor CSV data and (optionally) a trained model (.pkl) to visualize fire risk and predictions on an interactive map.
""")

# File uploaders
col1, col2 = st.columns(2)
with col1:
    data_file = st.file_uploader("Upload Sensor Data (CSV)", type=["csv"])
with col2:
    model_file = st.file_uploader("Upload Model (.pkl, optional)", type=["pkl"])

# Helper for severity color
def severity_color(value, threshold=1.5):
    if value >= threshold * 1.2:
        return 'darkred'
    elif value >= threshold:
        return 'orange'
    else:
        return 'green'

# Main logic
if data_file:
    df = pd.read_csv(data_file)
    required_columns = {'latitude', 'longitude', 'co_level', 'air_quality', 'temperature', 'humidity', 'pressure', 'voc_level'}
    if not required_columns.issubset(df.columns):
        st.error(f"CSV must contain columns: {required_columns}")
        st.stop()

    st.subheader("Data Preview")
    st.dataframe(df.head(20))

    # Load model if provided
    model = None
    if model_file:
        try:
            model = joblib.load(model_file)
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()
    else:
        st.info("No model uploaded. Only sensor data will be shown.")

    # Prepare map
    center = [df['latitude'].mean(), df['longitude'].mean()]
    fmap = folium.Map(location=center, zoom_start=13, tiles='CartoDB positron')

    # Marker clusters
    from folium.plugins import MarkerCluster, HeatMap
    marker_cluster = MarkerCluster(name='Sensor Markers').add_to(fmap)
    prediction_group = folium.FeatureGroup(name='Fire Predictions').add_to(fmap)

    # Heatmap layers
    heatmap_layers = {}
    for feature in ['co_level', 'air_quality', 'temperature', 'humidity', 'pressure', 'voc_level']:
        heat_data = [[row['latitude'], row['longitude'], row[feature]] for _, row in df.iterrows()]
        layer = folium.FeatureGroup(name=f"Heatmap: {feature.title().replace('_', ' ')}")
        HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(layer)
        heatmap_layers[feature] = layer

    # Store fire predictions
    fire_predictions = []

    # Add markers
    for _, row in df.iterrows():
        severity_info = f"""
        <b>CO Level:</b> {row['co_level']}<br>
        <b>Air Quality:</b> {row['air_quality']}<br>
        <b>Temperature:</b> {row['temperature']} °C<br>
        <b>Humidity:</b> {row['humidity']} %<br>
        <b>Pressure:</b> {row['pressure']} hPa<br>
        <b>VOC:</b> {row['voc_level']}<br>
        """
        prediction = None
        if model is not None:
            features = row[['co_level', 'air_quality', 'temperature', 'humidity', 'pressure', 'voc_level']].to_frame().T
            prediction = model.predict(features)[0]
            severity_info += f"<b>Prediction:</b> {'🔥 FIRE DETECTED' if prediction == 1 else '✅ SAFE'}<br>"
            if prediction == 1:
                fire_predictions.append(row)
                folium.Marker(
                    location=(row['latitude'], row['longitude']),
                    icon=folium.Icon(color='red', icon='fire', prefix='fa'),
                    popup=folium.Popup(severity_info, max_width=300)
                ).add_to(prediction_group)

        folium.CircleMarker(
            location=(row['latitude'], row['longitude']),
            radius=6,
            color=severity_color(row['co_level'], threshold=1.5),
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(severity_info, max_width=300)
        ).add_to(marker_cluster)

    # Add heatmap layers
    for layer in heatmap_layers.values():
        layer.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    st.subheader("Interactive Map")
    st_folium(fmap, width=900, height=600)

    # Export fire predictions
    if model is not None and fire_predictions:
        fire_df = pd.DataFrame(fire_predictions)
        csv = fire_df.to_csv(index=False)
        st.download_button(
            label="Download Fire Predictions CSV",
            data=csv,
            file_name="fire_predictions.csv",
            mime="text/csv"
        ) 