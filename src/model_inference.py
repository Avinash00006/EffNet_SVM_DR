import os
import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
from torchvision import models, transforms

#-----------GradCam Class Definition------------------------------------------------------

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple; grad_output[0] contains the gradients with respect to output
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        # Register hooks dynamically only for the duration of the CAM generation
        forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        backward_hook = self.target_layer.register_full_backward_hook(self.save_gradient)
        
        try:
            output = self.model(input_tensor)
            self.model.zero_grad()
            output.backward()

            gradients = self.gradients
            activations = self.activations

            # Compute weights based on average gradient per channel
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * activations, dim=1)
            cam = torch.relu(cam)

            cam = cam.squeeze().detach().numpy()
            cam = (cam - cam.min()) / (cam.max() + 1e-8)

            return cam
        finally:
            # Clean up hooks immediately to prevent memory leaks and channel shape pollution
            forward_hook.remove()
            backward_hook.remove()
            self.gradients = None
            self.activations = None


#----------Load Fine-Tuned Model--------------------------------------------------------

@st.cache_resource
def load_models(model_dir="model"):
    """
    Loads PyTorch EfficientNetV2 model weights, StandardScaler, and SVM classifier.
    Returns:
        tuple: (full_model, feature_extractor, scaler, svm, error_message)
    """
    best_effnet_path = os.path.join(model_dir, "best_effnet_binary.pth")
    scaler_path = os.path.join(model_dir, "scaler_binary.pkl")
    svm_path = os.path.join(model_dir, "svm_binary.pkl")

    # Verify if model assets exist
    missing = []
    if not os.path.exists(best_effnet_path):
        missing.append("best_effnet_binary.pth")
    if not os.path.exists(scaler_path):
        missing.append("scaler_binary.pkl")
    if not os.path.exists(svm_path):
        missing.append("svm_binary.pkl")

    if missing:
        error_msg = f"Missing model assets in '{model_dir}/' directory: {', '.join(missing)}"
        return None, None, None, None, error_msg

    try:
        # Load EfficientNetV2-S with modified binary classifier
        model = models.efficientnet_v2_s(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

        model.load_state_dict(
            torch.load(best_effnet_path, map_location=torch.device("cpu"))
        )
        model.eval()

        # Keep full model for Grad-CAM
        full_model = model

        # Feature extractor for SVM (excluding final classifier)
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        feature_extractor.eval()

        # Load StandardScaler and SVM model
        scaler = joblib.load(scaler_path)
        svm = joblib.load(svm_path)

        return full_model, feature_extractor, scaler, svm, None

    except Exception as e:
        return None, None, None, None, f"Exception occurred while loading models: {str(e)}"


#----------Explanation Engine--------------------------------------------------------

class ExplanationEngine:
    def __init__(self, full_model, feature_extractor, scaler, svm):
        self.full_model = full_model
        self.feature_extractor = feature_extractor
        self.scaler = scaler
        self.svm = svm
        
        # Instantiate Grad-CAM on the last feature block of EfficientNetV2
        target_layer = self.full_model.features[-1]
        self.grad_cam = GradCAM(self.full_model, target_layer)
        
        # Progress tracking variables
        self.progress_bar = None
        self.total_samples = 0
        self.processed_samples = 0

    def set_progress_tracker(self, progress_bar, total_samples):
        """Initializes the progress bar tracker before calling LIME explainer."""
        self.progress_bar = progress_bar
        self.total_samples = total_samples
        self.processed_samples = 0

    def lime_predict(self, images):
        """
        Prediction callback for LIME image explainer.
        Args:
            images (numpy.ndarray): Batch of images with shape (N, H, W, C) range [0, 1]
        Returns:
            numpy.ndarray: Probability array of shape (N, 2)
        """
        batch_size = len(images)
        
        # Update progress bar dynamically if registered
        if self.progress_bar is not None and self.total_samples > 0:
            self.processed_samples += batch_size
            fraction = min(self.processed_samples / self.total_samples, 1.0)
            current_count = min(self.processed_samples, self.total_samples)
            self.progress_bar.progress(
                fraction,
                text=f"LIME Analysis: {current_count} / {self.total_samples} perturbations processed..."
            )

        # Convert array to tensor and permute to (N, C, H, W)
        images_tensor = torch.tensor(images).permute(0, 3, 1, 2).float()

        # Normalize using PyTorch ImageNet standards
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Normalize each image in the batch
        normalized_tensors = [normalize(img) for img in images_tensor]
        images_tensor = torch.stack(normalized_tensors)

        with torch.no_grad():
            features = self.feature_extractor(images_tensor)
            features = torch.flatten(features, 1).numpy()

        features_scaled = self.scaler.transform(features)
        probs = self.svm.predict_proba(features_scaled)

        return probs
