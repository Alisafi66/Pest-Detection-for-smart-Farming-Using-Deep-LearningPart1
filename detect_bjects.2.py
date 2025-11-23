import csv
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# Function to save results to a CSV file
def save_results_to_csv(image_name, detections, output_csv_path):
    """
    Save detection results to a CSV file.

    Args:
        image_name (str): Name of the image file.
        detections (list of dict): List of detections containing object name, pest status, and confidence level.
        output_csv_path (str): Path to save the CSV file.
    """
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(["Image Name", "Detected Object", "Is Pest", "Confidence Level"])
        # Write the detection results
        for detection in detections:
            writer.writerow([
                image_name,
                detection["object"],
                detection["is_pest"],
                detection["confidence"]
            ])
    print(f"Results saved to {output_csv_path}")


# Path to your image
image_path = "/mnt/k/ml-aim/pest.1.jpg"

# Load the image
image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB format

# Load the pretrained processor and model
processor = AutoImageProcessor.from_pretrained("apple/aimv2-large-patch14-336")
model = AutoModel.from_pretrained("apple/aimv2-large-patch14-336", trust_remote_code=True)

# Process the image and prepare inputs for the model
inputs = processor(images=image, return_tensors="pt")

# Perform inference
outputs = model(**inputs)

# Simulate detection results (replace this with actual detection results from your model)
# Here we use dummy data for demonstration purposes
# Replace with the actual model's outputs if available
detections = [
    {"object": "Insect", "is_pest": "Yes", "confidence": 0.98},
    {"object": "Leaf", "is_pest": "No", "confidence": 0.85}
]

# Save results to a CSV file
output_csv_path = "/mnt/k/ml-aim/detection_results.csv"
image_name = image_path.split("/")[-1]  # Extract the image name from the path
save_results_to_csv(image_name, detections, output_csv_path)
