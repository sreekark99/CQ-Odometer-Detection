#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
print(f"Pytorch will be running on GPU (CUDA): {torch.cuda.is_available()}")


# In[2]:


import os
pwd = os.getcwd()
print(f"Present Working Directory: {pwd}")


# In[13]:


import os
import pandas as pd

class OdometerDataset:
    def __init__(self, root_folder, transform=None):
        self.root_folder = root_folder
        self.transform = transform
        self.data = self.load_images()

    def load_images(self):
        data = []  # Initialize an empty list to store image info
        
        # Scan all files in the root_folder and collect image files (JPEG, PNG, etc.)
        for root, _, files in os.walk(self.root_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_path = os.path.join(root, file)
                    image_name = os.path.basename(image_path)
                    data.append({
                        'image_name': image_name,
                        'image_path': image_path
                    })

        # Convert the list of dictionaries into a DataFrame
        df = pd.DataFrame(data)
        return df

root_folder = os.getcwd() + "\\CQ_Test"  
dataset = OdometerDataset(root_folder)

df = dataset.data

# In[ ]:


#YOLOv5 model fine-tuned
#python detect.py --weights runs/train/exp3/weights/best.pt --img 1024 --conf-thres 0.5 --iou-thres 0.4 --source ../CQ_Test/ --save-txt

print("Running Stage1 YOLOv5 model to detect odometer regions")
import subprocess
import re
import sys

# Define the command and arguments
command = [
    "python", 
    "yolov5/detect.py", 
    "--weights", "yolov5/runs/train/exp3/weights/best.pt", 
    "--img", "1024", 
    "--conf-thres", "0.5", 
    "--iou-thres", "0.4", 
    "--source", "CQ_Test/", 
    "--save-txt"
]

try:
    # Run the subprocess and capture the output (stdout and stderr)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Display the console output from the detection process
    print("Detection Output:")
    print(result.stdout)  # Standard output
    
    # Check for errors or warnings
    if result.stderr:
        print("\nErrors or Warnings:")
        print(result.stderr)  # Standard error
    stdout_arr = str(result).split("\\")
    exp_folder = stdout_arr[-4]
    
    if result.returncode == 0:
        print("\nDetection completed successfully.")
    else:
        print("\nThere was an issue with the detection process.")
        sys.exit(1)

except subprocess.CalledProcessError as e:
    print(f"An error occurred while running the detection: {e}")
    sys.exit(1)




# In[14]:


import os
from PIL import Image
import pandas as pd

def crop_odometer_regions(results_folder, images_folder, output_folder, existing_df):
    labels_folder = os.path.join(results_folder, "labels")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each detection result file
    for file_name in os.listdir(labels_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(labels_folder, file_name)
            image_name = file_name.replace(".txt", ".jpg")
            image_path = os.path.join(images_folder, image_name)

            # Check if the image exists
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue

            # Load the image
            image = Image.open(image_path)
            image_width, image_height = image.size

            # Read and filter detections (get the highest confidence detection)
            with open(file_path, 'r') as file:
                detections = file.readlines()
                detections = [line.strip().split() for line in detections]
                detections = sorted(detections, key=lambda x: float(x[-1]), reverse=True)
                best_detection = detections[0] if detections else None

            if best_detection:
                # Get bounding box coordinates
                x_center, y_center, width, height = map(float, best_detection[1:5])
                
                # Convert normalized coordinates to actual image coordinates
                x_min = int((x_center - width / 2) * image_width)
                y_min = int((y_center - height / 2) * image_height)
                x_max = int((x_center + width / 2) * image_width)
                y_max = int((y_center + height / 2) * image_height)

                # Crop the image to the odometer region
                cropped_image = image.crop((x_min, y_min, x_max, y_max))
                
                # Save the cropped image
                cropped_image_name = image_name
                output_image_path = os.path.join(output_folder, cropped_image_name)
                cropped_image.save(output_image_path)

                # Find the row corresponding to the current image_name and update it
                matching_row = existing_df[existing_df['image_name'] == image_name]
                if not matching_row.empty:
                    existing_df.loc[existing_df['image_name'] == image_name, 'cropped_image_name'] = cropped_image_name
                    existing_df.loc[existing_df['image_name'] == image_name, 'cropped_image_path'] = output_image_path
                    existing_df.loc[existing_df['image_name'] == image_name, 'x_min'] = x_min
                    existing_df.loc[existing_df['image_name'] == image_name, 'y_min'] = y_min
                    existing_df.loc[existing_df['image_name'] == image_name, 'x_max'] = x_max
                    existing_df.loc[existing_df['image_name'] == image_name, 'y_max'] = y_max

                print(f"Cropped image saved: {output_image_path}")
            else:
                print(f"No detection found in {file_name}")

    # Return the updated DataFrame
    return existing_df

results_folder = os.path.join(os.getcwd(), "yolov5", "runs", "detect", exp_folder)  # Path to the experiment folder
images_folder = os.path.join(os.getcwd(), "CQ_Test")  # Path to your test images
output_folder = os.path.join(os.getcwd(), "CQ_OUT_cropped")  # Folder where cropped images will be saved

existing_df = df
updated_df = crop_odometer_regions(results_folder, images_folder, output_folder, existing_df)
updated_df


# In[12]:


columns_to_save = ['image_name', 'cropped_image_path']
csv_file_path = pwd + "\\CQ_Cropped_odometer_data.csv"
updated_df[columns_to_save].to_csv(csv_file_path, index=False)
print(f"CSV file saved at: {csv_file_path}")


# In[6]:
print("Running stage2 Vision Transformer to detect digits")

from huggingface_hub import login
login("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")
#The key is confidential, DO NOT SHARE


# In[8]:


import pandas as pd
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Load TrOCR model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")

# Function to perform OCR on an image
def detect_text(image_path):
    if not image_path:  # Check if the path is None or empty
        return ""  # Return an empty string if the path is invalid
    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        detected_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return detected_text
    except Exception as e:
        return ""  # Return an empty string in case of any error

# Load the .csv file
csv_file = pwd + "\\CQ_Cropped_odometer_data.csv" 
data = pd.read_csv(csv_file)

results = []

for _, row in data.iterrows():
    image_name = row["image_name"]
    file_path = row["cropped_image_path"]
    print(f"Running digit recognition on file: {image_name}")
    detected_text = detect_text(file_path)
    detected_text = detected_text.replace(" ", "")
    results.append({"image_name": image_name, "prediction": detected_text})

# Create DataFrame from results
out_df = pd.DataFrame(results)

out_df['prediction'] = out_df['prediction'].fillna("")
#out_df is the final dataframe that has the results.


# In[9]:


csv_file_path = pwd + "\\CQ_Final_Predictions.csv"
out_df.to_csv(csv_file_path, index=False)
print(f"CSV file saved at: {csv_file_path}")
print("Execution successfully completed!")

