import argparse
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="AIM-v2 Inference on an Image")
parser.add_argument("-c", "--config", required=True, help="Path to model config or pre-trained identifier")
parser.add_argument("-p", "--weights", required=True, help="Path to model weights or pre-trained identifier")
parser.add_argument("-i", "--image", required=True, help="Path to input image")
parser.add_argument("-o", "--output", required=True, help="Directory to save the output image")
parser.add_argument("-t", "--text", required=True, help="Text prompt for visualization")
args = parser.parse_args()

# Load the image
image = Image.open(args.image).convert("RGB")

# Load the processor and model
processor = AutoImageProcessor.from_pretrained(args.config)
model = AutoModel.from_pretrained(args.weights, trust_remote_code=True)

# Preprocess the image
inputs = processor(images=image, return_tensors="pt")

# Perform inference
outputs = model(**inputs)

# Example visualization (replace with actual detection logic)
from PIL import ImageDraw, ImageFont

draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

# Dummy bounding box and label for demonstration
bounding_boxes = [(50, 50, 200, 200)]  # Example coordinates
labels = [args.text]

for box, label in zip(bounding_boxes, labels):
    x_min, y_min, x_max, y_max = box
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
    draw.text((x_min, y_min - 10), label, fill="red", font=font)

# Save the output
os.makedirs(args.output, exist_ok=True)
output_path = os.path.join(args.output, "annotated_image.jpg")
image.save(output_path)
print(f"Annotated image saved at: {output_path}")

