import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# 1. Set Device and Class Names
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classes = ['NORMAL', 'PNEUMONIA']

# 2. Rebuild the Model Architecture and Load Your Saved Weights
model = models.resnet50(weights=None) # We don't need ImageNet weights anymore
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('../models/pneumonia_resnet50_finetuned_v3.pth', weights_only=True))
model = model.to(device)
model.eval() # Set to evaluation mode

# 3. Define the exact same transforms used for the test set
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_xray(image_path):
    """Takes an image path, runs it through the model, and prints the diagnosis."""
    # Open image and convert to RGB (some X-rays are grayscale)
    image = Image.open(image_path).convert('RGB')
    
    # Apply transforms and add a batch dimension (C, H, W) -> (1, C, H, W)
    input_tensor = inference_transforms(image).unsqueeze(0).to(device)
    
    # Run the model
    with torch.no_grad():
        output = model(input_tensor)
        
        # Convert raw output logits to percentages using Softmax
        probabilities = F.softmax(output, dim=1)[0]
        
        # Get the predicted class index
        _, predicted_idx = torch.max(output, 1)
        predicted_idx = predicted_idx.item()
        
    diagnosis = classes[predicted_idx]
    confidence = probabilities[predicted_idx].item() * 100
    
    print(f"--- Analysis Complete ---")
    print(f"Diagnosis: {diagnosis}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"(Normal Probability: {probabilities[0].item()*100:.2f}% | Pneumonia Probability: {probabilities[1].item()*100:.2f}%)")

# ==========================================
# Test it out! Provide a path to a raw image:
predict_xray('../data/test/PNEUMONIA/person118_bacteria_560.jpeg')
# ==========================================