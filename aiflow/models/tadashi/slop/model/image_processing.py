import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import glob

class ImageProcessor:
    """Processor for image data"""
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process_image(self, image_path):
        """Process image for the model"""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image).unsqueeze(0)  # Add batch dimension
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    def extract_image_features(self, image_tensor, model, device):
        """Extract features using the model's image extractor"""
        # Move tensor to device
        image_tensor = image_tensor.to(device)

        # Get the image extractor from the model
        image_extractor = model.image_extractor

        # Set to eval mode
        image_extractor.eval()

        with torch.no_grad():
            # Extract features
            features = image_extractor(image_tensor)

        return features

class AIImageDetector:
    """Detector for AI-generated images"""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.processor = ImageProcessor()

    def classify_image(self, file_path):
        """Classify image file as human or AI generated"""
        # Process the image
        input_tensor = self.processor.process_image(file_path)

        if input_tensor is None:
            return {"Human": 0, "AI": 0, "Error": "Failed to process image"}

        # Move to device
        input_tensor = input_tensor.to(self.device)

        self.model.eval()
        with torch.no_grad():
            # Perform inference
            logits = self.model(input_tensor, 'image')
            probabilities = F.softmax(logits, dim=1)

            human_percent = probabilities[0, 0].item() * 100
            ai_percent = probabilities[0, 1].item() * 100

        return {"Human": human_percent, "AI": ai_percent}

    def batch_process_images(self, directory_path, file_types=None):
        """Process all image files in a directory"""
        if file_types is None:
            file_types = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']

        results = {}

        # Get all matching files
        files = []
        for file_type in file_types:
            files.extend(glob.glob(os.path.join(directory_path, file_type)))

        for file_path in files:
            try:
                file_result = self.classify_image(file_path)
                results[os.path.basename(file_path)] = file_result
            except Exception as e:
                results[os.path.basename(file_path)] = f"Error: {str(e)}"

        return results

def analyze_image_artifacts(image_path, target_size=(512, 512)):
    """Analyze image for common AI generation artifacts"""
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        image = image.resize(target_size)

        # Convert to numpy array
        img_array = np.array(image)

        # Analysis results
        results = {}

        # Check for unnatural patterns in pixel distributions
        # (AI-generated images often have different statistical properties)
        pixel_stats = {}
        for channel, color in enumerate(['red', 'green', 'blue']):
            channel_data = img_array[:, :, channel].flatten()
            pixel_stats[color] = {
                'mean': np.mean(channel_data),
                'std': np.std(channel_data),
                'min': np.min(channel_data),
                'max': np.max(channel_data),
                'median': np.median(channel_data)
            }

        results['pixel_statistics'] = pixel_stats

        # Check for symmetry (AI images sometimes have unnatural symmetry)
        h, w, _ = img_array.shape
        left_half = img_array[:, :w//2, :]
        right_half = img_array[:, w//2:, :]
        right_half_flipped = np.flip(right_half, axis=1)

        # Trim if sizes don't match
        min_w = min(left_half.shape[1], right_half_flipped.shape[1])
        symmetry_score = np.mean(np.abs(left_half[:, :min_w, :] - right_half_flipped[:, :min_w, :]))
        results['symmetry_score'] = symmetry_score

        # Check for repeating patterns (common in AI-generated images)
        # This is a simplified approach - more sophisticated methods would use FFT or autocorrelation
        patch_size = 16
        patches = []
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = img_array[i:i+patch_size, j:j+patch_size, :]
                patches.append(patch.flatten())

        patches = np.array(patches)

        # Calculate pairwise distances between patches
        from scipy.spatial.distance import pdist
        if len(patches) > 1:
            distances = pdist(patches)
            results['patch_similarity'] = {
                'mean_distance': np.mean(distances),
                'min_distance': np.min(distances),
                'std_distance': np.std(distances)
            }

        return results

    except Exception as e:
        return {"error": str(e)}

def detect_facial_inconsistencies(image_path):
    """Detect inconsistencies in faces that might indicate AI generation"""
    try:
        # This would typically use a face detection library like dlib or face_recognition
        # For simplicity, we'll just outline the approach

        # 1. Detect faces in the image
        # 2. For each face:
        #    - Check for symmetry
        #    - Look for inconsistencies in facial features (eyes, ears, etc.)
        #    - Check for unnatural skin textures

        # For now, return a placeholder
        return {"message": "Face inconsistency detection would be implemented here"}

    except Exception as e:
        return {"error": str(e)}

def detect_image_boundaries(image_path):
    """Detect unnatural boundaries in images (common in AI-generated images)"""
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        img_array = np.array(image)

        # Convert to grayscale for edge detection
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2).astype(np.uint8)
        else:
            gray = img_array

        # Simple edge detection using Sobel filters
        from scipy import ndimage
        sobel_h = ndimage.sobel(gray, axis=0)
        sobel_v = ndimage.sobel(gray, axis=1)
        magnitude = np.sqrt(sobel_h**2 + sobel_v**2)

        # Analyze edge statistics
        edge_stats = {
            'mean_magnitude': np.mean(magnitude),
            'max_magnitude': np.max(magnitude),
            'std_magnitude': np.std(magnitude)
        }

        return edge_stats

    except Exception as e:
        return {"error": str(e)}
