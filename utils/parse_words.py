import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache
import time

class FastImageLoader:
    def __init__(self, max_workers=8, cache_size=1000):
        self.max_workers = max_workers
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.max_cache_size = cache_size
        
    @lru_cache(maxsize=2000)
    def _load_single_image_cached(self, img_path, target_size):
        """Load and preprocess a single image with caching"""
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            
            # Resize image
            img = cv2.resize(img, target_size)
            # Normalize to [0, 1]
            img = img.astype('float32') / 255.0
            # Add channel dimension
            img = np.expand_dims(img, axis=-1)
            
            return img
        except Exception as e:
            print(f"❌ Error loading {img_path}: {e}")
            return None
    
    def load_images_parallel(self, image_paths, target_size=(256, 32)):
        """Load multiple images in parallel"""
        images = []
        valid_indices = []
        
        def load_image_worker(args):
            idx, img_path = args
            img = self._load_single_image_cached(img_path, target_size)
            return idx, img
        
        # Prepare arguments for parallel processing
        args_list = [(i, path) for i, path in enumerate(image_paths)]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {executor.submit(load_image_worker, args): args[0] 
                           for args in args_list}
            
            # Collect results
            results = {}
            for future in as_completed(future_to_idx):
                try:
                    idx, img = future.result()
                    if img is not None:
                        results[idx] = img
                        valid_indices.append(idx)
                except Exception as e:
                    print(f"❌ Error in parallel loading: {e}")
        
        # Sort results by original index
        valid_indices.sort()
        images = [results[idx] for idx in valid_indices]
        
        return np.array(images), valid_indices

def load_words_dataset(root_dir="dataset", img_size=(256, 32), max_text_len=16, max_samples=15000):
    """
    Optimized dataset loading with parallel image processing
    """
    print(f"🚀 Starting optimized dataset loading...")
    start_time = time.time()
    
    # Try different possible locations for words.txt
    possible_label_files = [
        os.path.join(root_dir, "words_new.txt"),
        os.path.join(root_dir, "iam_words_new", "words_new.txt"),
        os.path.join(root_dir, "..", "words_new.txt"),
        "words_new.txt"
    ]
    
    label_file = None
    for lf in possible_label_files:
        if os.path.exists(lf):
            label_file = lf
            break
    
    # Try different possible locations for images
    possible_img_dirs = [
        os.path.join(root_dir, "iam_words", "words"),  # Correct IAM structure
        os.path.join(root_dir, "iam_words", "words_new"),
        os.path.join(root_dir, "words_new"),
        os.path.join(root_dir, "iam_words"),
    ]
    
    img_dir = None
    for id in possible_img_dirs:
        if os.path.exists(id):
            img_dir = id
            break
    
    print(f"📁 Found label file at: {label_file}")
    print(f"📁 Looking for images in: {img_dir}")
    
    if label_file is None:
        print(f"❌ ERROR: Label file not found in any location")
        return np.array([]), []
    
    if img_dir is None:
        print(f"❌ ERROR: Image directory not found")
        return np.array([]), []
    
    # First pass: collect all valid image paths and labels
    print("📋 Phase 1: Collecting image paths and labels...")
    image_paths = []
    texts = []
    processed_count = 0
    found_count = 0
    
    with open(label_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.startswith('#'):
                continue
            
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(' ')
            if len(parts) < 9:
                continue
                
            img_name = parts[0]
            processed_count += 1
            
            # Show progress every 1000 processed lines
            if processed_count % 1000 == 0:
                print(f"📊 Processed {processed_count} lines, found {found_count} valid images...")
            
            # Find image file (FAST - no os.walk)
            img_file = find_image_file(img_name, img_dir)
            
            if img_file and os.path.exists(img_file):
                found_count += 1
                try:
                    # Extract label
                    label = ' '.join(parts[8:]).strip()
                    if not label:
                        continue
                    
                    # Filter by text length for CTC stability
                    if len(label) < 3 or len(label) > max_text_len:
                        continue
                    
                    # Filter out labels with only special characters
                    if not any(c.isalnum() for c in label):
                        continue
                    
                    image_paths.append(img_file)
                    texts.append(label)
                    
                    # Limit dataset size
                    if len(image_paths) >= max_samples:
                        print(f"✅ Reached limit of {max_samples} samples")
                        break
                        
                except Exception as e:
                    continue
    
    print(f"📊 Phase 1 complete: Processed {processed_count} lines, found {found_count} images, kept {len(image_paths)} valid samples")
    
    print(f"📊 Found {len(image_paths)} valid samples")
    
    if len(image_paths) == 0:
        print("⚠️  No valid samples found")
        return np.array([]), []
    
    # Second pass: load images in parallel with MAXIMUM speed
    print("🖼️  Phase 2: Loading images with MAXIMUM speed...")
    loader = FastImageLoader(max_workers=12, cache_size=500)  # More workers, smaller cache for speed
    
    # Process in smaller batches for faster loading
    batch_size = 1000  # REDUCED batch size for faster processing
    all_images = []
    all_texts = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]
        
        print(f"📦 Processing batch {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1}")
        
        batch_images, valid_indices = loader.load_images_parallel(batch_paths, img_size)
        
        # Keep only valid samples
        valid_texts = [batch_texts[idx] for idx in valid_indices]
        
        all_images.extend(batch_images)
        all_texts.extend(valid_texts)
        
        print(f"✅ Batch loaded: {len(batch_images)} images")
    
    final_images = np.array(all_images)
    
    end_time = time.time()
    loading_time = end_time - start_time
    
    print(f"🎉 Dataset loading completed!")
    print(f"📊 Total samples: {len(final_images)}")
    print(f"⏱️  Loading time: {loading_time:.2f} seconds")
    print(f"🚀 Speed: {len(final_images)/loading_time:.1f} samples/second")
    
    if len(final_images) > 0:
        print(f"📏 Image shape: {final_images[0].shape}")
        print(f"📝 Sample labels: {all_texts[:5]}")
    
    return final_images, all_texts

def find_image_file(img_name, img_dir):
    """FAST image file finder - NO os.walk to prevent hanging"""
    possible_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    
    for ext in possible_extensions:
        potential_name = img_name + ext
        
        # IAM dataset structure: words/[first3]/[form_id]/[full_name].png
        # Example: a01-000u-00-00 -> words/a01/a01-000u/a01-000u-00-00.png
        
        # Extract form ID (e.g., "a01-000u" from "a01-000u-00-00")
        parts = img_name.split('-')
        if len(parts) >= 2:
            form_id = f"{parts[0]}-{parts[1]}"  # e.g., "a01-000u"
        else:
            form_id = img_name[:6]  # fallback
        
        possible_paths = [
            # Direct in img_dir
            os.path.join(img_dir, potential_name),
            # IAM standard structure: first 3 chars / form_id / filename
            os.path.join(img_dir, img_name[:3], form_id, potential_name),
            # Alternative patterns
            os.path.join(img_dir, img_name[:3], img_name[:6], potential_name),
            os.path.join(img_dir, img_name[:3], img_name.split('-')[0], potential_name),
            os.path.join(img_dir, img_name[:3], potential_name),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
    
    # NO os.walk - return None if not found in expected locations
    # This prevents the hanging issue completely
    return None

# Additional utility functions for data quality
def filter_dataset_for_quality(images, texts, min_char_variety=3):
    """Filter dataset to improve training quality"""
    filtered_images = []
    filtered_texts = []
    
    for img, text in zip(images, texts):
        # Check character variety
        unique_chars = len(set(text.lower()))
        if unique_chars >= min_char_variety:
            filtered_images.append(img)
            filtered_texts.append(text)
    
    print(f"🔍 Quality filter: {len(filtered_images)}/{len(images)} samples kept")
    return np.array(filtered_images), filtered_texts

def get_character_stats(texts):
    """Analyze character distribution in dataset"""
    char_count = {}
    for text in texts:
        for char in text:
            char_count[char] = char_count.get(char, 0) + 1
    
    print(f"📊 Character statistics:")
    print(f"   Total unique characters: {len(char_count)}")
    print(f"   Most common: {sorted(char_count.items(), key=lambda x: x[1], reverse=True)[:10]}")
    
    return char_count