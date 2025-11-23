import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModel
import requests
import matplotlib.pyplot as plt


# Define AIMv2 Object Detection Model
class AIMv2ObjectDetector(nn.Module):
    def __init__(self, base_model, embedding_dim=1536, num_classes=91):
        super().__init__()
        self.base_model = base_model  # AIMv2 base model
        self.bbox_head = nn.Linear(embedding_dim, 4)  # Adjust input dimensions to match AIMv2 output
        self.class_head = nn.Linear(embedding_dim, num_classes)  # Adjust input dimensions

    def forward(self, inputs):
        # Extract features from the base model
        features = self.base_model(**inputs).last_hidden_state
        pooled_features = features.mean(dim=1)  # Global pooling across all patches

        # Predict bounding boxes and classes
        bboxes = self.bbox_head(pooled_features)
        classes = self.class_head(pooled_features)
        return bboxes, classes



# Helper function to draw bounding boxes
def draw_bboxes(image, bboxes, labels):
    draw = ImageDraw.Draw(image)
    for bbox, label in zip(bboxes, labels):
        x, y, w, h = bbox
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
        draw.text((x, y), str(label), fill="red")
    return image


# Load Pretrained AIMv2 Model
def load_model():
    model_name = "apple/aimv2-huge-patch14-224"
    processor = AutoImageProcessor.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    detector_model = AIMv2ObjectDetector(base_model)
    return processor, detector_model


# Inference with AIMv2 for Object Detection
def infer_object_detection(processor, detector_model, image_url):
    # Load and preprocess the image
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt")

    # Predict bounding boxes and class scores
    with torch.no_grad():
        bboxes, class_scores = detector_model(inputs)

    # Post-process outputs
    bboxes = bboxes[0].tolist()  # Convert tensor to list
    classes = torch.argmax(class_scores, dim=1).tolist()  # Get class predictions

    return image, bboxes, classes


# Main Execution
if __name__ == "__main__":
    # Load the AIMv2 object detection model
    processor, detector_model = load_model()

    # URL of the image for detection
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    # Perform inference
    image, bboxes, classes = infer_object_detection(processor, detector_model, image_url)

    # Visualize the results
    result_image = draw_bboxes(image, bboxes, classes)
    plt.imshow(result_image)
    plt.axis("off")
    result_image.save("result_image.jpg")
