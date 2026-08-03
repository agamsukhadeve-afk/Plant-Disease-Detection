import os
import cv2
import random
from src.config import Config
from src.predict import PlantDiseasePredictor

def test_single_inference():
    # Get all class names
    class_names = sorted(os.listdir(Config.TEST_DIR))
    
    # Pick a random class and a random image from that class
    random_class = random.choice(class_names)
    class_dir = os.path.join(Config.TEST_DIR, random_class)
    random_image_name = random.choice(os.listdir(class_dir))
    image_path = os.path.join(class_dir, random_image_name)
    
    print(f"Testing image: {image_path}")
    print(f"True Label: {random_class}")
    
    # Load image (OpenCV reads as BGR, we need RGB for our predictor)
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Initialize predictor
    model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    predictor = PlantDiseasePredictor(model_path, class_names)
    
    # Predict
    results = predictor.predict(image_rgb)
    
    print("\nTop 3 Predictions:")
    for i, res in enumerate(results):
        print(f"{i+1}. {res['class']} - Confidence: {res['probability']*100:.2f}%")
        
    if results[0]['class'] == random_class:
        print("\n✅ PREDICTION IS CORRECT!")
    else:
        print("\n❌ PREDICTION IS INCORRECT!")

if __name__ == "__main__":
    test_single_inference()
