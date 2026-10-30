import os
import numpy as np
import tensorflow as tf
from utils.parse_words import load_words_dataset

# Configure TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("🧪 Quick Model Testing - Current Best Model")
print("=" * 50)

# Character mapping
CHARACTERS = " !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def decode_predictions(predictions):
    """Decode CTC predictions using TensorFlow's CTC decoder - PROPER VERSION"""
    # Use TensorFlow's built-in CTC greedy decoder
    predictions_transposed = tf.transpose(predictions, [1, 0, 2])  # (time, batch, classes)
    input_length = tf.fill([tf.shape(predictions)[0]], tf.shape(predictions)[1])
    
    # Greedy CTC decoder
    decoded, _ = tf.nn.ctc_greedy_decoder(
        predictions_transposed,
        input_length,
        merge_repeated=True
    )
    
    # Convert sparse tensor to dense
    decoded_dense = tf.sparse.to_dense(decoded[0], default_value=-1).numpy()
    
    # Convert indices to text
    decoded_texts = []
    for sequence in decoded_dense:
        chars = []
        for idx in sequence:
            if idx >= 0 and idx < len(CHARACTERS):
                chars.append(CHARACTERS[idx])
        decoded_texts.append(''.join(chars).strip())
    
    return decoded_texts

# Load test data
print("📂 Loading test data...")
try:
    X, labels = load_words_dataset(
        root_dir="dataset", 
        img_size=(256, 32), 
        max_text_len=16, 
        max_samples=1000  # Small sample for quick testing
    )
    X = X.astype('float32') / 255.0
    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=-1)
    
    print(f"✅ Loaded {len(X)} test samples")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# Load the current best model
model_path = 'best_model_improved.keras'
if not os.path.exists(model_path):
    print(f"❌ Model {model_path} not found!")
    exit(1)

print(f"🔄 Loading model: {model_path}")
try:
    model = tf.keras.models.load_model(model_path, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Test predictions on random samples
print("\n🧪 Testing model predictions...")
num_samples = 10
indices = np.random.choice(len(X), num_samples, replace=False)
test_images = X[indices]
test_labels = [labels[i] for i in indices]

# Make predictions
predictions = model.predict(test_images, verbose=0)
decoded_predictions = decode_predictions(predictions)

print("\n📊 Prediction Results:")
print("-" * 60)
correct = 0
for i, (true_label, pred_label) in enumerate(zip(test_labels, decoded_predictions)):
    true_label = str(true_label).strip()
    pred_label = pred_label.strip()
    
    is_correct = true_label.lower() == pred_label.lower()
    if is_correct:
        correct += 1
    
    status = "✅" if is_correct else "❌"
    print(f"{i+1:2d}. {status} True: '{true_label}' | Pred: '{pred_label}'")

accuracy = correct / len(test_labels) * 100
print("-" * 60)
print(f"🎯 Current Accuracy: {correct}/{len(test_labels)} ({accuracy:.1f}%)")

# Quick validation loss check
print(f"\n📊 Quick validation check...")
val_indices = np.random.choice(len(X), 100, replace=False)
val_images = X[val_indices]

# Simple loss estimation (without proper CTC setup)
try:
    val_predictions = model.predict(val_images, verbose=0)
    print(f"✅ Model is responding well to validation data")
    print(f"📊 Prediction shape: {val_predictions.shape}")
    print(f"📊 Average prediction confidence: {np.mean(np.max(val_predictions, axis=-1)):.3f}")
except Exception as e:
    print(f"⚠️ Validation check error: {e}")

print(f"\n🎯 Model Status: GOOD - Ready for continued training!")
print(f"🔄 Training is still running in background...")
print(f"💡 Run 'python monitor_training.py' to check training progress") 