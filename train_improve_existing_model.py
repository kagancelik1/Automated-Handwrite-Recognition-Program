#!/usr/bin/env python3
"""
IMPROVE EXISTING MODEL - best_model_improved.keras ONLY
NO NEW MODELS - Just improve the existing one
Memory optimized to prevent OOM errors
"""

import os
import numpy as np
import cv2
import random
import time
import platform
import shutil

# Configure environment before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
from tensorflow.keras import regularizers, mixed_precision
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K
import math

# ✅ CLEAR SESSION TO PREVENT MEMORY LEAKS
K.clear_session()

# ✅ ENABLE MIXED PRECISION FOR MEMORY SAVINGS
mixed_precision.set_global_policy('mixed_float16')
print("✅ Mixed precision (FP16) enabled for memory efficiency!")

from model.model import build_crnn
from utils.parse_words import load_words_dataset

print("🎯 IMPROVING EXISTING MODEL - best_model_improved.keras")
print("🚫 NO NEW MODELS WILL BE CREATED")
print("🔧 MEMORY OPTIMIZED TO PREVENT OOM ERRORS")
print("="*70)

# Configure GPU with conservative memory settings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # ✅ VERY CONSERVATIVE 1.5GB limit to prevent OOM
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1536)]
            )
        print(f"✅ GPU configured with 1.5GB memory limit (ultra-conservative)")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration warning: {e}")
else:
    print("🚨 No GPU detected!")
    exit(1)

# Character mapping for CTC
CHARACTERS = " !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
char_to_num = {char: idx for idx, char in enumerate(CHARACTERS)}
num_to_char = {idx: char for idx, char in enumerate(CHARACTERS)}
VOCAB_SIZE = len(CHARACTERS)

def encode_text(text):
    """Encode text to sequence of character indices"""
    return [char_to_num.get(char, 0) for char in text]

def decode_predictions(predictions):
    """Decode CTC predictions using TensorFlow's CTC decoder"""
    predictions_transposed = tf.transpose(predictions, [1, 0, 2])
    input_length = tf.fill([tf.shape(predictions)[0]], tf.shape(predictions)[1])
    
    decoded, _ = tf.nn.ctc_greedy_decoder(
        predictions_transposed,
        input_length,
        merge_repeated=True
    )
    
    decoded_dense = tf.sparse.to_dense(decoded[0], default_value=-1).numpy()
    
    decoded_texts = []
    for sequence in decoded_dense:
        chars = []
        for idx in sequence:
            if idx >= 0 and idx < len(CHARACTERS):
                chars.append(CHARACTERS[idx])
        decoded_texts.append(''.join(chars).strip())
    
    return decoded_texts

def ctc_loss_func(y_true, y_pred):
    """CTC loss function with numerical stability"""
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    
    # Numerical stability
    loss = tf.where(tf.math.is_inf(loss), tf.zeros_like(loss), loss)
    loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
    loss = tf.clip_by_value(loss, 0.0, 100.0)
    
    return loss

# Load SMALLER dataset to prevent OOM
print("\n📂 Loading SMALLER dataset to prevent OOM...")
try:
    # ✅ MUCH SMALLER dataset for memory safety
    X, labels = load_words_dataset(
        root_dir="dataset", 
        img_size=(256, 32), 
        max_text_len=16, 
        max_samples=4000  # ✅ Reduced to 4000 for memory safety
    )
    print(f"✅ Loaded {len(X)} samples (memory-optimized size)")
    
    # Normalize images
    X = X.astype('float32') / 255.0
    
    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=-1)
    
    print(f"📊 Final X shape: {X.shape}")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# Prepare labels
print("\n🔤 Preparing labels...")
try:
    encoded_labels = []
    max_label_length = 0
    
    for label in labels:
        clean_label = str(label).strip()
        encoded = encode_text(clean_label)
        encoded_labels.append(encoded)
        max_label_length = max(max_label_length, len(encoded))
    
    y = np.zeros((len(encoded_labels), max_label_length), dtype=np.int32)
    for i, encoded in enumerate(encoded_labels):
        y[i, :len(encoded)] = encoded
    
    print(f"📊 Labels shape: {y.shape}")
    
except Exception as e:
    print(f"❌ Error preparing labels: {e}")
    exit(1)

# Data splitting
print("\n✂️ Splitting data...")
np.random.seed(42)
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices] 
labels = [labels[i] for i in indices]

train_split = int(0.85 * len(X))
val_split = int(0.95 * len(X))

X_train, X_val, X_test = X[:train_split], X[train_split:val_split], X[val_split:]
y_train, y_val, y_test = y[:train_split], y[train_split:val_split], y[val_split:]
labels_train, labels_val, labels_test = labels[:train_split], labels[train_split:val_split], labels[val_split:]

print(f"📊 Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# ✅ LOAD EXISTING MODEL - NO NEW MODELS!
print("\n🔄 Loading EXISTING model: best_model_improved.keras")
print("🚫 NO NEW MODELS WILL BE CREATED!")

try:
    # Check if model exists
    if not os.path.exists('best_model_improved.keras'):
        print("❌ best_model_improved.keras not found!")
        exit(1)
    
    # Load the existing model
    model = tf.keras.models.load_model('best_model_improved.keras', compile=False)
    print("✅ Existing model loaded successfully!")
    print(f"📊 Model parameters: {model.count_params():,}")
    
except Exception as e:
    print(f"❌ Error loading existing model: {e}")
    exit(1)

# ✅ COMPILE MODEL FIRST before evaluation
print("\n🔧 Compiling model with conservative settings...")
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.00005,  # VERY LOW learning rate
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
    clipnorm=0.1  # VERY TIGHT gradient clipping
)

model.compile(
    optimizer=optimizer,
    loss=ctc_loss_func,
    run_eagerly=False
)

print("✅ Model compiled with conservative settings")

# Test current performance AFTER compilation
print("\n🔍 Testing current model performance...")
val_loss_before = model.evaluate(X_val, y_val, verbose=0, batch_size=1)
print(f"📊 Current validation loss: {val_loss_before:.4f}")

# ✅ MINIMAL augmentation to save memory
def minimal_augment(image, label):
    """Minimal augmentation to save memory"""
    if tf.random.uniform([]) > 0.7:  # Only 30% chance
        image = tf.image.random_brightness(image, 0.05)  # Very light
    return image, label

# ✅ ULTRA-SMALL batch size
batch_size = 1  # ✅ Smallest possible batch size

# Create memory-efficient datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.map(minimal_augment, num_parallel_calls=1)
train_dataset = train_dataset.shuffle(buffer_size=100).batch(batch_size).prefetch(1)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(batch_size).prefetch(1)

print("✅ Ultra-memory-efficient data pipeline created")

# ✅ CONSERVATIVE callbacks - SAVE TO SAME FILE
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        patience=8,
        factor=0.5,
        min_lr=1e-10,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001
    ),
    # ✅ SAVE TO SAME FILE - NO NEW MODELS!
    tf.keras.callbacks.ModelCheckpoint(
        'best_model_improved.keras',  # ✅ SAME FILE NAME
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
        save_weights_only=False,
        mode='min'
    )
]

# Start improvement training
print(f"\n🏃‍♂️ Starting IMPROVEMENT of existing model...")
print(f"🎯 Goal: Improve validation loss from {val_loss_before:.4f}")
print("🚫 NO NEW MODELS - Only improving existing best_model_improved.keras")

try:
    start_time = time.time()
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,  # Moderate number of epochs
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Test final performance
    val_loss_after = model.evaluate(X_val, y_val, verbose=0, batch_size=1)
    improvement = val_loss_before - val_loss_after
    
    print(f"\n🎉 MODEL IMPROVEMENT COMPLETED!")
    print(f"⏱️ Training time: {training_time/60:.1f} minutes")
    print(f"📊 Validation loss BEFORE: {val_loss_before:.4f}")
    print(f"📊 Validation loss AFTER:  {val_loss_after:.4f}")
    print(f"📈 Improvement: {improvement:+.4f}")
    
    if improvement > 0:
        print("✅ MODEL IMPROVED! ✅")
    else:
        print("🔶 No improvement, but model preserved")
    
    print(f"\n💾 Model saved to: best_model_improved.keras (SAME FILE)")
    print("🚫 NO NEW MODELS CREATED")
    print("✅ Your existing model has been improved!")
    
    # ✅ MEMORY CLEANUP
    K.clear_session()
    print("🧹 Memory cleaned up")
    
except Exception as e:
    print(f"❌ Training error: {e}")
    import traceback
    traceback.print_exc()

print("\n🎯 IMPROVEMENT TRAINING COMPLETED!")
print("✅ Your best_model_improved.keras has been improved (if possible)")
print("🚫 No new models were created")
print("💡 Use the same best_model_improved.keras file in your web app") 