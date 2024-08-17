import streamlit as st
import torch
import numpy as np
from PIL import Image
from albumentations import Compose, Resize
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import timm
import torch.nn as nn


# Configuration class
class Config:
    model_name = 'vit_base_patch16_224_in21k'
    resize = (224, 224)
    folds = 3  # Number of folds

# Set device
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Model class
class VITModel(nn.Module):
    def __init__(self, model_name=Config.model_name, pretrained=False):
        super(VITModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=3)
        self.model.head = nn.Linear(self.model.head.in_features, 1)

    def forward(self, x):
        x = self.model(x)
        return x


def load_fold_models():
    models = []
    for fold in range(Config.folds):
        model = VITModel().to(DEVICE)
        model.load_state_dict(torch.load(f"models/{Config.model_name}_fold_{fold}.pt", map_location=DEVICE))
        model.eval()
        models.append(model)
    return models

# Predict function for a single image
def predict_single_image(models, image):
    img_np = np.array(image)  # For displaying later

    valid_augments = Compose([
        Resize(*Config.resize, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)

    img_tensor = valid_augments(image=img_np)['image'].unsqueeze(0)  # Add batch dimension
    img_tensor = img_tensor.to(DEVICE, dtype=torch.float)

    fold_preds = []
    with torch.no_grad():
        for model in models:
            outputs = torch.sigmoid(model(img_tensor))
            fold_preds.append(outputs.cpu().detach().numpy())

    ensemble_pred = np.mean(fold_preds, axis=0)
    return ensemble_pred

# Streamlit UI
st.title("Art Classification: AI-Generated vs. Real Artwork")
st.write("Upload an image to predict whether it's AI-generated or a real artwork.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Load models
    models = load_fold_models()

    

    if st.button('Submit'):

    # Predict
        prediction = predict_single_image(models, image)
        predicted_label = "Real Artwork" if prediction > 0.5 else "AI-Generated Artwork"

        st.write(f"**Prediction**: {predicted_label}")

        # Option to show actual label if known
        
        # Display image with prediction streamlit app/app.py