import cv2
import numpy as np
import random
from typing import Tuple, List
import tensorflow as tf

class HandwritingAugmenter:
    """
    Data augmentation specifically designed for handwriting recognition
    """
    
    def __init__(self, 
                 rotation_range=3,
                 shear_range=0.1,
                 brightness_range=0.2,
                 noise_factor=0.05,
                 elastic_alpha=20,
                 elastic_sigma=3):
        """
        Initialize augmenter with parameters
        
        Args:
            rotation_range: Maximum rotation in degrees
            shear_range: Maximum shear transformation
            brightness_range: Brightness adjustment range
            noise_factor: Gaussian noise factor
            elastic_alpha: Elastic deformation alpha
            elastic_sigma: Elastic deformation sigma
        """
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.brightness_range = brightness_range
        self.noise_factor = noise_factor
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
    
    def random_rotation(self, image: np.ndarray) -> np.ndarray:
        """Apply random rotation"""
        if random.random() < 0.3:  # 30% chance
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, matrix, (w, h), 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=255)
        return image
    
    def random_shear(self, image: np.ndarray) -> np.ndarray:
        """Apply random shear transformation"""
        if random.random() < 0.2:  # 20% chance
            h, w = image.shape[:2]
            shear_x = random.uniform(-self.shear_range, self.shear_range)
            shear_y = random.uniform(-self.shear_range * 0.5, self.shear_range * 0.5)
            
            # Create shear matrix
            shear_matrix = np.array([[1, shear_x, 0],
                                   [shear_y, 1, 0]], dtype=np.float32)
            
            return cv2.warpAffine(image, shear_matrix, (w, h),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=255)
        return image
    
    def random_brightness(self, image: np.ndarray) -> np.ndarray:
        """Apply random brightness adjustment"""
        if random.random() < 0.4:  # 40% chance
            factor = random.uniform(1 - self.brightness_range, 1 + self.brightness_range)
            image = image.astype(np.float32)
            image = image * factor
            image = np.clip(image, 0, 255)
            return image.astype(np.uint8)
        return image
    
    def add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise"""
        if random.random() < 0.3:  # 30% chance
            noise = np.random.normal(0, self.noise_factor * 255, image.shape)
            image = image.astype(np.float32) + noise
            image = np.clip(image, 0, 255)
            return image.astype(np.uint8)
        return image
    
    def elastic_deformation(self, image: np.ndarray) -> np.ndarray:
        """Apply elastic deformation for more realistic variations"""
        if random.random() < 0.2:  # 20% chance
            h, w = image.shape[:2]
            
            # Create random displacement fields
            dx = np.random.uniform(-1, 1, (h//4, w//4)) * self.elastic_alpha
            dy = np.random.uniform(-1, 1, (h//4, w//4)) * self.elastic_alpha
            
            # Smooth the displacement fields
            dx = cv2.GaussianBlur(dx, (0, 0), self.elastic_sigma)
            dy = cv2.GaussianBlur(dy, (0, 0), self.elastic_sigma)
            
            # Resize to match image dimensions
            dx = cv2.resize(dx, (w, h))
            dy = cv2.resize(dy, (w, h))
            
            # Create coordinate grids
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            x_new = (x + dx).astype(np.float32)
            y_new = (y + dy).astype(np.float32)
            
            # Apply deformation
            return cv2.remap(image, x_new, y_new, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        return image
    
    def random_erasing(self, image: np.ndarray) -> np.ndarray:
        """Randomly erase rectangular regions"""
        if random.random() < 0.1:  # 10% chance
            h, w = image.shape[:2]
            
            # Random rectangle
            x1 = random.randint(0, w // 2)
            y1 = random.randint(0, h // 2)
            x2 = min(w, x1 + random.randint(5, w // 4))
            y2 = min(h, y1 + random.randint(2, h // 4))
            
            # Fill with random intensity
            fill_value = random.randint(200, 255)
            image = image.copy()
            image[y1:y2, x1:x2] = fill_value
            
        return image
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply all augmentations randomly"""
        # Apply augmentations in sequence
        image = self.random_rotation(image)
        image = self.random_shear(image)
        image = self.random_brightness(image)
        image = self.add_noise(image)
        image = self.elastic_deformation(image)
        image = self.random_erasing(image)
        
        return image
    
    def augment_batch(self, images: np.ndarray, labels: List[str], 
                     augment_factor: int = 2) -> Tuple[np.ndarray, List[str]]:
        """
        Augment a batch of images
        
        Args:
            images: Array of images (N, H, W) or (N, H, W, C)
            labels: List of corresponding text labels
            augment_factor: How many augmented versions per image
            
        Returns:
            Augmented images and corresponding labels
        """
        print(f"🔄 Augmenting {len(images)} images with factor {augment_factor}...")
        
        augmented_images = []
        augmented_labels = []
        
        # Keep original data
        for i, (img, label) in enumerate(zip(images, labels)):
            if len(img.shape) == 3 and img.shape[-1] == 1:
                img = img.squeeze(-1)  # Remove channel dimension for processing
            
            augmented_images.append(img)
            augmented_labels.append(label)
            
            # Generate augmented versions
            for _ in range(augment_factor):
                aug_img = self.augment_image(img.copy())
                augmented_images.append(aug_img)
                augmented_labels.append(label)
            
            if (i + 1) % 100 == 0:
                print(f"   Processed {i + 1}/{len(images)} images...")
        
        # Convert back to numpy array with channel dimension
        augmented_images = np.array(augmented_images)
        if len(augmented_images.shape) == 3:
            augmented_images = np.expand_dims(augmented_images, axis=-1)
        
        print(f"✅ Augmentation complete: {len(images)} -> {len(augmented_images)} images")
        
        return augmented_images, augmented_labels

def create_tf_augmentation_layer():
    """
    Create TensorFlow-based augmentation layer for training
    """
    def augment_fn(image):
        # Random brightness
        image = tf.image.random_brightness(image, 0.1)
        
        # Random contrast
        image = tf.image.random_contrast(image, 0.8, 1.2)
        
        # Random rotation (small angles)
        image = tf.image.rot90(image, k=tf.random.uniform([], 0, 1, dtype=tf.int32))
        
        # Ensure values are in valid range
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image
    
    return augment_fn

def improve_image_quality(image: np.ndarray) -> np.ndarray:
    """
    Improve image quality for better recognition
    """
    # Denoise
    denoised = cv2.fastNlMeansDenoising(image)
    
    # Adaptive thresholding for better binarization
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def analyze_image_quality(images: np.ndarray, labels: List[str]) -> dict:
    """
    Analyze image quality and provide statistics
    """
    print("📊 Analyzing image quality...")
    
    stats = {
        'mean_brightness': [],
        'contrast': [],
        'noise_level': [],
        'text_density': []
    }
    
    for img in images:
        if len(img.shape) == 3:
            img = img.squeeze(-1)
        
        # Mean brightness
        stats['mean_brightness'].append(np.mean(img))
        
        # Contrast (standard deviation)
        stats['contrast'].append(np.std(img))
        
        # Noise level (Laplacian variance)
        stats['noise_level'].append(cv2.Laplacian(img, cv2.CV_64F).var())
        
        # Text density (ratio of non-white pixels)
        text_pixels = np.sum(img < 200)
        total_pixels = img.size
        stats['text_density'].append(text_pixels / total_pixels)
    
    # Calculate summary statistics
    summary = {}
    for key, values in stats.items():
        summary[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    print("✅ Image quality analysis complete")
    return summary

if __name__ == "__main__":
    # Test augmentation
    print("🧪 Testing data augmentation...")
    
    # Create dummy image
    test_image = np.ones((32, 256), dtype=np.uint8) * 255
    cv2.putText(test_image, "Test", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    
    # Test augmenter
    augmenter = HandwritingAugmenter()
    
    # Apply augmentations
    for i in range(5):
        augmented = augmenter.augment_image(test_image.copy())
        print(f"✅ Augmentation {i+1} applied successfully")
    
    print("🎉 Data augmentation test completed!") 