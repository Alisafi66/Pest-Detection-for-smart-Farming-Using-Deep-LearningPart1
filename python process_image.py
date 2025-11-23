from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# Path to your image
image_path = "/mnt/k/ml-aim/pest.1"

# Load the image
image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB format

# Load the pretrained processor and model
processor = AutoImageProcessor.from_pretrained("apple/aimv2-large-patch14-336")
model = AutoModel.from_pretrained("apple/aimv2-large-patch14-336", trust_remote_code=True)

# Process the image and prepare inputs for the model
inputs = processor(images=image, return_tensors="pt")

# Perform inference
outputs = model(**inputs)

# Display the output
print("Model outputs:", outputs)

