from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModel

# Load the image
image_path = "/mnt/k/ml-aim/pest.1.jpg"  # Replace with the actual path to your image
image = Image.open(image_path).convert("RGB")  # Ensure RGB format

# Load the processor and model
processor = AutoImageProcessor.from_pretrained("apple/aimv2-large-patch14-336")
model = AutoModel.from_pretrained("apple/aimv2-large-patch14-336", trust_remote_code=True)

# Process the image
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# Example: Dummy bounding boxes, labels, and confidence scores
# Replace this with actual detection outputs if available
bounding_boxes = [(50, 50, 200, 200), (150, 100, 300, 250)]  # Format: (x_min, y_min, x_max, y_max)
labels = ["Object A", "Object B"]
confidence_scores = [0.85, 0.92]  # Example confidence scores

# Draw bounding boxes and labels on the image
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

for box, label, score in zip(bounding_boxes, labels, confidence_scores):
    x_min, y_min, x_max, y_max = box
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)  # Draw rectangle
    draw.text(
        (x_min, y_min - 10),
        f"{label} ({score:.2f})",  # Label with confidence score
        fill="red",
        font=font
    )

# Save or display the annotated image
annotated_image_path = "/path/to/save/annotated_image.jpg"  # Save path for the detected image
image.save(annotated_image_path)
print(f"Annotated image saved to: {annotated_image_path}")
