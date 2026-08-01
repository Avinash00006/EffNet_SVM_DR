import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

def preprocess_fundus_image(pil_image, is_already_preprocessed=False):
    """
    Applies Ben Graham's preprocessing to a retinal fundus image.
    If the image is already preprocessed (gray background), it simply resizes the image.
    """
    # Convert PIL → OpenCV
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize to 512×512 (MANDATORY)
    img = cv2.resize(img, (512, 512))

    if not is_already_preprocessed:
        # Ben Graham preprocessing (MANDATORY for raw images)
        img = cv2.addWeighted(
            img, 4,
            cv2.GaussianBlur(img, (0, 0), 10),
            -4, 128
        )

    # Convert back to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return Image.fromarray(img)

# Transform pipeline to convert preprocessed image to tensor for EfficientNet model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def validate_fundus_image(pil_image):
    """
    Validates if the uploaded image is a standard retinal fundus photography.
    Supports both raw (black background, red-orange retina) and already preprocessed (gray background) images.
    Returns:
        tuple: (is_valid, error_message, is_already_preprocessed)
    """
    w, h = pil_image.size
    aspect_ratio = max(w, h) / min(w, h)
    
    # 1. Aspect Ratio check (relaxed to 1.6 to allow rectangular APTOS camera images)
    if aspect_ratio > 1.6:
        return False, "Image is not square (aspect ratio must be < 1.6).", False

    # Convert to numpy array for color/pixel analysis
    img = np.array(pil_image)
    if len(img.shape) != 3 or img.shape[2] != 3:
        return False, "Must be a 3-channel RGB color image.", False

    # Resize to a small resolution for speed
    small = cv2.resize(img, (100, 100))
    
    # 2. Check Corner Darkness
    corner_size = 10
    corner_patches = [
        small[0:corner_size, 0:corner_size],
        small[0:corner_size, 100-corner_size:100],
        small[100-corner_size:100, 0:corner_size],
        small[100-corner_size:100, 100-corner_size:100]
    ]
    corner_means = []
    for patch in corner_patches:
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        corner_means.append(np.mean(gray_patch))
        
    avg_corner_mean = np.mean(corner_means)
    
    # 3. Check Center color profile
    center = small[30:70, 30:70]
    mean_rgb = np.mean(center, axis=(0, 1))
    r, g, b = mean_rgb[0], mean_rgb[1], mean_rgb[2]
    max_diff = max(r, g, b) - min(r, g, b)

    # CASE A: Already Preprocessed (Gray background from Ben Graham method)
    # Corners are neutral gray (typically 100-150) and center has very low color variance (gray-like)
    is_preprocessed_corners = all(100 <= m <= 150 for m in corner_means)
    is_preprocessed_center = max_diff < 15 and (100 <= r <= 150)
    
    if is_preprocessed_corners and is_preprocessed_center:
        return True, "Valid preprocessed fundus image.", True
        
    # CASE B: Raw Fundus Image (Black background, red/orange retina)
    is_raw_corners = all(m < 55 for m in corner_means)
    if is_raw_corners:
        if r < 30: # Relaxed slightly for dark/underexposed photos
            return False, "Image center is too dark.", False
        if r <= g or r <= b:
            return False, "Incorrect color profile (red/orange hues must dominate).", False
        
        # Use ratios rather than absolute differences to support very dark/underexposed images
        red_blue_ratio = r / (b + 1e-5)
        red_green_ratio = r / (g + 1e-5)
        if red_blue_ratio < 1.18 or red_green_ratio < 1.03:
            return False, "Incorrect color distribution (insufficient red dominance).", False
        return True, "Valid raw fundus image.", False
        
    # If it fails both cases
    if avg_corner_mean > 55:
        return False, "Corners are not black (fundus images must have a solid black background surround).", False
        
    return False, "Incorrect retinal color profile.", False
