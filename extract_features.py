# File: extract_features.py
from PIL import Image
from aim.v2.utils import load_pretrained
from aim.v1.torch.data import val_transforms

# Path to the image
image_path = "/mnt/k/ml-aim/pest.1.jpg"

# Load the image
img = Image.open(image_path)

# Load the pretrained model
model = load_pretrained("aimv2-large-patch14-336", backend="torch")

# Apply transformations
transform = val_transforms(img_size=336)

# Prepare the input
inp = transform(img).unsqueeze(0)

# Extract features
features = model(inp)

# Print the extracted features
print(detected_image) 
