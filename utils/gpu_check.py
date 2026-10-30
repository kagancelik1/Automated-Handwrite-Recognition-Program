#!/usr/bin/env python3
"""
GPU Detection and Validation Utility
Ensures training only proceeds with proper GPU setup
"""

import tensorflow as tf
import os
import sys

def check_gpu_availability(require_gpu=True, min_memory_gb=4):
    """
    Check if GPU is available and properly configured for training
    
    Args:
        require_gpu (bool): If True, exit if no GPU found
        min_memory_gb (int): Minimum GPU memory required in GB
    
    Returns:
        tuple: (gpu_available, gpu_info)
    """
    print("=" * 60)
    print("🔍 GPU DETECTION AND VALIDATION")
    print("=" * 60)
    
    # Check TensorFlow GPU support
    print(f"🔧 TensorFlow version: {tf.__version__}")
    print(f"🔧 CUDA support built-in: {tf.test.is_built_with_cuda()}")
    
    # Get physical GPU devices
    physical_gpus = tf.config.list_physical_devices('GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    
    gpu_info = {
        'physical_count': len(physical_gpus),
        'logical_count': len(logical_gpus),
        'devices': physical_gpus,
        'memory_info': []
    }
    
    if len(physical_gpus) == 0:
        print("❌ NO GPU DETECTED!")
        print("   Make sure you have:")
        print("   1. NVIDIA GPU with CUDA support")
        print("   2. CUDA toolkit installed")
        print("   3. cuDNN installed")
        print("   4. tensorflow-gpu package installed")
        print("   5. Proper environment activated (e.g., venv_wsl_gpu)")
        
        if require_gpu:
            print("\n🚨 TRAINING STOPPED: GPU is required for this training script!")
            print("🔧 To continue with GPU, activate the GPU environment:")
            print("   source venv_wsl_gpu/bin/activate")
            print("   python3 test_wsl_gpu.py  # to verify GPU works")
            sys.exit(1)
        else:
            print("⚠️  Continuing with CPU training (will be much slower)")
            return False, gpu_info
    
    print(f"✅ GPU DETECTED: {len(physical_gpus)} physical GPU(s)")
    
    # Test GPU functionality
    try:
        for i, gpu in enumerate(physical_gpus):
            print(f"   GPU {i}: {gpu.name}")
            
            # Enable memory growth to avoid allocation issues
            tf.config.experimental.set_memory_growth(gpu, True)
            
            # Get GPU memory info if possible
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                compute_capability = gpu_details.get('compute_capability', 'Unknown')
                print(f"   Compute Capability: {compute_capability}")
            except:
                print(f"   Compute Capability: Unable to detect")
        
        # Test GPU computation
        print("🧪 Testing GPU computation...")
        with tf.device('/GPU:0'):
            # Simple matrix multiplication test
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            result = tf.reduce_sum(c)
            print(f"   GPU computation test: ✅ Success (result: {result.numpy():.2f})")
            
        # Check if mixed precision is supported
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("✅ Mixed precision (float16) enabled for faster training")
        except Exception as e:
            print(f"⚠️  Mixed precision not supported: {e}")
            tf.keras.mixed_precision.set_global_policy('float32')
        
        gpu_info['functional'] = True
        
    except Exception as e:
        print(f"❌ GPU FUNCTIONALITY TEST FAILED: {e}")
        if require_gpu:
            print("🚨 TRAINING STOPPED: GPU not functioning properly!")
            sys.exit(1)
        else:
            print("⚠️  Falling back to CPU training")
            return False, gpu_info
    
    print("=" * 60)
    print("🚀 GPU VALIDATION SUCCESSFUL - READY FOR TRAINING!")
    print("=" * 60)
    
    return True, gpu_info

def configure_gpu_for_training():
    """
    Configure GPU settings for optimal training performance
    """
    physical_gpus = tf.config.list_physical_devices('GPU')
    
    if len(physical_gpus) > 0:
        try:
            # Enable memory growth to prevent OOM errors
            for gpu in physical_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set mixed precision for faster training
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            # Enable XLA compilation for additional speedup
            tf.config.optimizer.set_jit(True)
            
            # Disable verbose device placement logging
            tf.debugging.set_log_device_placement(False)
            
            print("✅ GPU configured for optimal training performance")
            return '/GPU:0'
            
        except RuntimeError as e:
            print(f"⚠️  GPU configuration warning: {e}")
            return '/GPU:0'
    else:
        return '/CPU:0'

def get_optimal_batch_size(gpu_available=True, model_complexity='medium'):
    """
    Get optimal batch size based on GPU availability and model complexity
    """
    if gpu_available:
        if model_complexity == 'high':
            return 16
        elif model_complexity == 'medium':
            return 32
        else:
            return 64
    else:
        return 8  # Conservative for CPU

def print_training_environment_info():
    """
    Print comprehensive information about the training environment
    """
    print("\n" + "=" * 60)
    print("🌟 TRAINING ENVIRONMENT INFORMATION")
    print("=" * 60)
    
    # Python and TensorFlow info
    import sys
    print(f"🐍 Python version: {sys.version}")
    print(f"🔧 TensorFlow version: {tf.__version__}")
    
    # GPU info
    physical_gpus = tf.config.list_physical_devices('GPU')
    if physical_gpus:
        print(f"🚀 GPU devices: {len(physical_gpus)}")
        for i, gpu in enumerate(physical_gpus):
            print(f"   GPU {i}: {gpu.name}")
    else:
        print("💻 Using CPU for training")
    
    # Memory info
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"💾 System RAM: {memory.total // (1024**3)} GB")
        print(f"💾 Available RAM: {memory.available // (1024**3)} GB")
    except ImportError:
        print("💾 RAM info: psutil not available")
    
    # Environment variables
    print(f"🔧 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"🔧 TF_GPU_ALLOCATOR: {os.environ.get('TF_GPU_ALLOCATOR', 'Not set')}")
    
    print("=" * 60)

if __name__ == "__main__":
    # Test the GPU check function
    gpu_available, gpu_info = check_gpu_availability(require_gpu=False)
    print_training_environment_info()
    
    if gpu_available:
        device = configure_gpu_for_training()
        print(f"🎯 Training device: {device}")
        batch_size = get_optimal_batch_size(gpu_available=True)
        print(f"📦 Recommended batch size: {batch_size}")
    else:
        print("⚠️  Training will proceed on CPU (much slower)") 