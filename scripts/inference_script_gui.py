import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# ==========================================
# 1. Custom Grad-CAM Implementation
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        loss = output[0, class_idx]
        loss.backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        
        # Prevent division by zero if there are no positive activations
        max_val = torch.max(heatmap)
        if max_val > 0:
            heatmap /= max_val
            
        return heatmap.cpu().detach().numpy()

# ==========================================
# 2. Main Execution Logic
# ==========================================
def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Missing arguments. Usage: python inference_script_gui.py <input_img> <output_heatmap>"}))
        sys.exit(1)

    input_image_path = sys.argv[1]
    output_heatmap_path = sys.argv[2]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classes = ['NORMAL', 'PNEUMONIA']

    # Load Model
    try:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load('../models/pneumonia_resnet50_finetuned_v4.pth', weights_only=True))
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(json.dumps({"error": f"Failed to load model: {str(e)}"}))
        sys.exit(1)

    inference_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Process Image
    try:
        image_pil = Image.open(input_image_path).convert('RGB')
        input_tensor = inference_transforms(image_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            _, predicted_idx = torch.max(output, 1)
            predicted_idx = predicted_idx.item()

        # Target class index 1 (PNEUMONIA) specifically
        pneumonia_idx = 1
        pneumonia_prob = probabilities[pneumonia_idx].item()

        cam = GradCAM(model, model.layer4[-1])
        input_tensor.requires_grad_(True)
        heatmap = cam.generate(input_tensor, pneumonia_idx)

        # Scale heatmap intensity by the model's confidence in Pneumonia
        heatmap = heatmap * pneumonia_prob

        # Load image with OpenCV and resize heatmap to match
        img_cv = cv2.imread(input_image_path)
        heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
        
        # Convert to 8-bit image and apply JET colormap
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Dynamic alpha blending: opacity scales with the heatmap intensity
        # High pneumonia areas get up to 50% opacity; low/zero areas remain 0% opacity (clean image)
        alpha = np.expand_dims(heatmap_resized * 0.5, axis=-1)
        superimposed_img = (alpha * heatmap_colored + (1 - alpha) * img_cv).astype(np.uint8)

        # Save output
        cv2.imwrite(output_heatmap_path, superimposed_img)

        # ==========================================
        # Confidence Threshold Logic
        # ==========================================
        pneumonia_prob_percent = round(probabilities[1].item() * 100, 2)
        normal_prob_percent = round(probabilities[0].item() * 100, 2)
        
        # Set a threshold (e.g., 75%) for a definitive diagnosis
        if pneumonia_prob_percent >= 75.0:
            final_diagnosis = "PNEUMONIA"
            final_confidence = pneumonia_prob_percent
        elif normal_prob_percent >= 75.0:
            final_diagnosis = "NORMAL"
            final_confidence = normal_prob_percent
        else:
            final_diagnosis = "REQUIRES ATTENTION (Borderline)"
            final_confidence = max(pneumonia_prob_percent, normal_prob_percent)

        # Prepare JSON response
        result = {
            "diagnosis": final_diagnosis,
            "confidence": final_confidence,
            "normal_prob": normal_prob_percent,
            "pneumonia_prob": pneumonia_prob_percent,
            "heatmap_path": output_heatmap_path
        }
        
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": f"Inference failed: {str(e)}"}))
        sys.exit(1)

if __name__ == "__main__":
    main()