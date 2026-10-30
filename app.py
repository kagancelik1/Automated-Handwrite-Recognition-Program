#!/usr/bin/env python3
"""
Handwriting Recognition Web Application
Uses the trained best_model_improved.keras model with Flask
"""

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import numpy as np
import cv2
import os
from PIL import Image
import tensorflow as tf
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)
app.secret_key = 'handwriting_recognition_app_2024'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Character mapping used in the model (with CTC blank token)
CHARACTERS = " !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# CTC models need a blank token at the end - index 79 is the CTC blank token
# The model outputs 80 classes: 0-78 (characters) + 79 (blank)

# Load the trained model
model = None
print("🔍 Loading trained model...")

try:
    model_path = 'best_model_improved.keras'
    if os.path.exists(model_path):
        print(f"📁 Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ Model loaded successfully!")
        print(f"📊 Input shape: {model.input_shape}")
        print(f"📊 Output shape: {model.output_shape}")
    else:
        print(f"❌ Model file not found: {model_path}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """
    Preprocess uploaded image for model prediction
    FINAL FIX: Use EXACT same preprocessing as working quick_test_model.py
    """
    try:
        # Read image in grayscale - EXACT same as dataset loader
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ Could not read image: {image_path}")
            return None
        
        print(f"🔍 Original image shape: {img.shape}")
        print(f"🔍 Original value range: [{img.min()}, {img.max()}]")
        
        # Resize to model input size (256x32) - EXACT same as dataset loader
        img_resized = cv2.resize(img, (256, 32))
        
        mean_value = np.mean(img_resized)
        print(f"✅ Using image as-is (mean={mean_value:.1f}) - same as dataset loader")
        
        
        # Dataset loader: img.astype('float32') / 255.0
        # Double normalization
        img_normalized = img_resized.astype('float32') / 255.0
        img_normalized = img_normalized.astype('float32') / 255.0  
        
        # EXACT same channel addition as dataset loader
        
        img_normalized = np.expand_dims(img_normalized, axis=-1)
        
        print(f"🔧 Processed image range: [{img_normalized.min():.6f}, {img_normalized.max():.6f}]")
        print(f"🔧 Mean value: {img_normalized.mean():.6f}")
        
        # Add batch dimension for prediction
        img_final = np.expand_dims(img_normalized, axis=0)
        
        print(f"🔍 Final shape for model: {img_final.shape}")
        
        return img_final
        
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None

def decode_ctc_predictions(predictions):
    """
    Decode CTC predictions to text using TensorFlow's CTC decoder
    """
    try:
        print(f"🔍 Predictions shape: {predictions.shape}")
        
        # Transpose for CTC decoder (time, batch, classes)
        predictions_transposed = tf.transpose(predictions, [1, 0, 2])
        input_length = tf.fill([tf.shape(predictions)[0]], tf.shape(predictions)[1])
        
        # Use greedy CTC decoder
        decoded, _ = tf.nn.ctc_greedy_decoder(
            predictions_transposed,
            input_length,
            merge_repeated=True
        )
        
        # Convert sparse tensor to dense
        decoded_dense = tf.sparse.to_dense(decoded[0], default_value=-1).numpy()
        
        # Convert indices to characters
        decoded_texts = []
        for sequence in decoded_dense:
            chars = []
            for idx in sequence:
                
                if idx >= 0 and idx < len(CHARACTERS):
                    chars.append(CHARACTERS[idx])
            decoded_texts.append(''.join(chars).strip())
        
        result = decoded_texts[0] if decoded_texts else ""
        print(f"🔍 Decoded text: '{result}'")
        
        # Calculate confidence score
        max_probs = np.max(predictions[0], axis=-1)
        avg_confidence = np.mean(max_probs) * 100
        print(f"🔍 Confidence: {avg_confidence:.1f}%")
        
        return result, avg_confidence
        
    except Exception as e:
        print(f"❌ Error decoding predictions: {e}")
        return "Error during text recognition", 0.0

@app.route('/')
def index():
    """Main page with upload interface"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and make prediction"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected. Please choose an image file.')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected. Please choose an image file.')
            return redirect(request.url)
        
        # Check if model is available
        if model is None:
            flash('Model not available. Please ensure the model file exists.')
            return redirect(request.url)
        
        # Process valid file
        if file and allowed_file(file.filename):
            # Save uploaded file with timestamp
            filename = secure_filename(file.filename)
            timestamp = str(int(time.time()))
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            print(f"📁 Saved uploaded file: {filepath}")
            
            # Preprocess image
            processed_image = preprocess_image(filepath)
            
            if processed_image is None:
                flash('Error processing image. Please try with a different image.')
                return redirect(request.url)
            
            # Make prediction
            print("🔮 Making prediction...")
            predictions = model.predict(processed_image, verbose=0)
            
            # Decode predictions
            predicted_text, confidence = decode_ctc_predictions(predictions)
            
            print(f"📝 Final result: '{predicted_text}' (confidence: {confidence:.1f}%)")
            
            # Render results page
            return render_template('result.html', 
                                 predicted_text=predicted_text,
                                 confidence=f"{confidence:.1f}%",
                                 filename=filename)
        
        else:
            flash('Invalid file type. Please upload PNG, JPG, JPEG, GIF, or BMP files only.')
            return redirect(request.url)
    
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        flash('An error occurred during processing. Please try again.')
        return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files for display in results"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File too large. Maximum size is 16MB.')
    return redirect(request.url)

if __name__ == '__main__':
    print("🌐 Handwriting Recognition Web Application")
    print("=" * 50)
    
    if model is not None:
        print("✅ Model loaded and ready")
        print("📁 Upload folder ready")
        print("🎨 Templates ready")
        print("🚀 Starting server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Cannot start - model not available")
        print("💡 Please ensure 'best_model_improved.keras' exists in the current directory") 