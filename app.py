import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os

from src.predict import PlantDiseasePredictor
from src.gradcam import GradCAMVisualizer
from src.config import Config

# Must be the first streamlit command
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")

@st.cache_resource
def load_predictor():
    # Attempt to read classes from data directory, or just use generic numbered classes if not yet downloaded
    if os.path.exists(Config.TRAIN_DIR):
        class_names = sorted(os.listdir(Config.TRAIN_DIR))
    else:
        class_names = [f"Class_{i}" for i in range(Config.NUM_CLASSES)]
        
    model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    predictor = PlantDiseasePredictor(model_path, class_names)
    gradcam = GradCAMVisualizer(predictor.model)
    return predictor, gradcam

def main():
    st.title("🌿 Plant Disease Detection & Diagnosis")
    st.markdown("Upload a leaf image to detect potential diseases. The system uses an **EfficientNet-B0** model and provides a **Grad-CAM heatmap** to show which parts of the leaf the model focused on.")
    
    predictor, gradcam = load_predictor()
    
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        # Streamlit expects RGB, cv2 loads as BGR
        rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(rgb_image, caption="Uploaded Image", use_container_width=True)
            
        if st.button("Predict Disease"):
            with st.spinner("Analyzing image..."):
                start_time = time.time()
                
                # Predict
                results = predictor.predict(rgb_image)
                
                # Grad-CAM
                heatmap_img = gradcam.generate_heatmap(rgb_image)
                
                end_time = time.time()
                inference_time = end_time - start_time
                
            with col2:
                st.image(heatmap_img, caption="Grad-CAM Heatmap (Model Focus)", use_container_width=True)
                
            st.success(f"Inference complete in {inference_time:.2f} seconds.")
            
            st.subheader("Predictions")
            st.progress(results[0]['probability'])
            st.markdown(f"**Top Prediction:** {results[0]['class']} ({results[0]['probability']*100:.2f}%)")
            
            st.markdown("### Top-3 Candidates")
            for i, res in enumerate(results):
                st.write(f"{i+1}. **{res['class']}** - {res['probability']*100:.2f}%")

if __name__ == "__main__":
    main()
