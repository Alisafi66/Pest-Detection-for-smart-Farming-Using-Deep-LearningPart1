import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw

# Load the DETR processor and model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Load the image
image_path = "/mnt/k/ml-aim/pest.1.jpg"
image = Image.open(image_path).convert("RGB")

# Process the image
inputs = processor(images=image, return_tensors="pt")

# Perform inference
outputs = model(**inputs)

# Post-process results
target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

# Draw bounding boxes on the image
draw = ImageDraw.Draw(image)
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]  # Convert to list and round off
    draw.rectangle(box, outline="red", width=3)
    label_text = f"{model.config.id2label[label.item()]}: {score:.2f}"
    draw.text((box[0], box[1] - 10), label_text, fill="red")

# Save or display the annotated image
annotated_image_path = "/mnt/k/ml-aim/annotated_image.jpg"
image.save(annotated_image_path)
print(f"Annotated image saved at: {annotated_image_path}")
