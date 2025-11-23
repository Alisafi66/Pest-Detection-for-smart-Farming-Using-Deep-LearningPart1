import os
import csv
from PIL import Image
from transformers import AutoProcessor, AutoModel

# Define the folder path
folder_path = "/mnt/k/ml-aim/isolated_images"

# Define the output CSV file path
output_csv = "/mnt/k/ml-aim/detection_results.csv"

# Define labels
text = ["pest"]

# Load processor and model
processor = AutoProcessor.from_pretrained(
    "apple/aimv2-large-patch14-224-lit",
)
model = AutoModel.from_pretrained(
    "apple/aimv2-large-patch14-224-lit",
    trust_remote_code=True,
)

# Prepare the CSV file
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(["Image Name", "Detected Object", "Is Pest", "Confidence Level"])

    # Loop through images in the folder
    image_count = 0
    for filename in os.listdir(folder_path):
        # Construct the full image path
        image_path = os.path.join(folder_path, filename)

        # Skip directories and non-image files
        if not os.path.isfile(image_path) or not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        try:
            # Load image
            image = Image.open(image_path).convert("RGB")

            # Process inputs
            inputs = processor(
                images=image,
                text=text,
                add_special_tokens=True,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )

            # Get predictions
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=-1)

            # Find the label with the highest confidence
            max_prob_index = probs.argmax().item()
            detected_label = text[max_prob_index]
            confidence_level = probs[0][max_prob_index].item()   # Scale to percentage

            # Check if the detected object is "pest"
            is_pest = detected_label.lower() == "pest"

            # Write result to CSV
            writer.writerow([filename, detected_label, is_pest, f"{confidence_level:.2f}%"])

            # Print progress
            print(f"Processed: {filename}, Detected: {detected_label}, Confidence: {confidence_level:.2f}%")

            image_count += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

print(f"Processing completed. {image_count} images processed. Results saved to {output_csv}")
