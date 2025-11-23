import torch
from PIL import Image, ImageDraw
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# Load the processor and model
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Load your image
image_path = "/mnt/k/ml-aim/Plant_Pest-Header.jpg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")

# Define text queries
text_queries = ["pest"]  # Replace with your queries

# Preprocess inputs
inputs = processor(text=text_queries, images=image, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)

# Post-process outputs
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

# Draw bounding boxes on the image
draw = ImageDraw.Draw(image)
for box, score, label in zip(results[0]["boxes"], results[0]["scores"], results[0]["labels"]):
    if score >= 0.1:  # Adjust threshold as needed
        box = [round(i, 2) for i in box.tolist()]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f"{text_queries[label]}: {round(score.item(), 2)}", fill="red")

# Save or display the image
image.save("output_image.jpg")
image.show()
