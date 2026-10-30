import os
import json
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from tensorflow import keras
from PIL import Image

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Paths
BASE_DIR = r'E:\Rice Leaf Disease minor project'
MODEL_PATH = os.path.join(BASE_DIR, 'model.keras')
META_PATH = os.path.join(BASE_DIR, 'model_metadata.json')

# Load Metadata
with open(META_PATH, 'r') as f:
    metadata = json.load(f)

CLASS_NAMES = metadata['class_names']
IMG_SIZE = tuple(metadata['img_size'])
SEVERITY_MAP = metadata['severity_thresholds']

# Load Model
print("Loading model.keras...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

def estimate_severity(img_rgb, predicted_class):
    if predicted_class == 'Healthy':
        return 0.0, 'None (Healthy)'
    
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    leaf_mask = cv2.inRange(hsv, np.array([10, 25, 25]), np.array([95, 255, 245]))
    lesion_mask = cv2.inRange(hsv, np.array([5, 30, 20]), np.array([30, 255, 220]))
    
    leaf_pixels = np.count_nonzero(leaf_mask)
    lesion_pixels = np.count_nonzero(cv2.bitwise_and(lesion_mask, leaf_mask))
    
    if leaf_pixels == 0:
        return 0.0, 'Mild'
    
    infected_pct = min(100.0, (lesion_pixels / leaf_pixels) * 100.0)
    
    severity_label = 'Mild'
    for level, (low, high) in SEVERITY_MAP.items():
        if low <= infected_pct <= high:
            severity_label = level
            break
    return round(infected_pct, 2), severity_label

def generate_segmentation_image(img_rgb, predicted_class):
    """Generates the visual segmentation overlay (green leaf contour + red lesion spots) as Base64."""
    overlay = img_rgb.copy()
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # 1. Detect Leaf Boundary (Green Contour)
    leaf_mask = cv2.inRange(hsv, np.array([10, 25, 25]), np.array([95, 255, 245]))
    kernel = np.ones((3, 3), np.uint8)
    leaf_mask_clean = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(leaf_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    
    # 2. Highlight Disease Lesions in Red (if not Healthy)
    if predicted_class != 'Healthy':
        lesion_mask = cv2.inRange(hsv, np.array([5, 30, 20]), np.array([30, 255, 220]))
        isolated_lesions = cv2.bitwise_and(lesion_mask, leaf_mask_clean)
        
        red_layer = np.zeros_like(img_rgb)
        red_layer[:] = [255, 40, 40]  # Red highlight in RGB
        
        mask_bool = isolated_lesions > 0
        overlay[mask_bool] = cv2.addWeighted(img_rgb, 0.35, red_layer, 0.65, 0)[mask_bool]
        
        # Draw thin contour around lesions for clarity
        lesion_contours, _ = cv2.findContours(isolated_lesions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, lesion_contours, -1, (255, 0, 0), 1)
    
    # Encode RGB overlay to Base64 PNG string
    bgr_overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', bgr_overlay)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded in request'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400
        
        # Preprocess image
        pil_img = Image.open(file.stream).convert('RGB').resize(IMG_SIZE)
        img_arr = np.array(pil_img)
        input_tensor = np.expand_dims(img_arr / 255.0, axis=0).astype(np.float32)
        
        # Predict using model
        preds = model.predict(input_tensor, verbose=0)[0]
        pred_idx = np.argmax(preds)
        pred_cls = CLASS_NAMES[pred_idx]
        confidence = float(preds[pred_idx]) * 100
        
        # Estimate Severity
        sev_pct, sev_level = estimate_severity(img_arr, pred_cls)
        
        # Generate Segmentation Output
        seg_image_b64 = generate_segmentation_image(img_arr, pred_cls)
        
        return jsonify({
            'success': True,
            'prediction': pred_cls,
            'confidence': round(confidence, 2),
            'severity_level': sev_level,
            'severity_percentage': sev_pct,
            'segmentation_image': seg_image_b64
        })
    except Exception as e:
        import traceback
        traceback.print_exc()  # Prints exact error in your terminal console
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)