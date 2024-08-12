import os
import json
import xml.etree.ElementTree as ET
from PIL import Image

def convert_annotations(annotation_dir, image_dir, output_json):
    annotations = []

    for xml_file in os.listdir(annotation_dir):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(annotation_dir, xml_file))
            root = tree.getroot()

            # Access elements using relative paths
            image_file_element = root.find('./filename')
            if image_file_element is None:
                print(f"Error: 'filename' element not found in {xml_file}")
                continue
            image_file = image_file_element.text
            image_path = os.path.join(image_dir, image_file)

            try:
                image = Image.open(image_path)
            except FileNotFoundError:
                print(f"Image file not found: {image_path}")
                continue

            width, height = image.size
            
            words = []
            boxes = []
            labels = []
            
            for obj in root.findall('./object'):
                label = obj.find('./name').text
                bbox = obj.find('./bndbox')
                xmin = int(bbox.find('./xmin').text) / width
                ymin = int(bbox.find('./ymin').text) / height
                xmax = int(bbox.find('./xmax').text) / width
                ymax = int(bbox.find('./ymax').text) / height
                box = [xmin, ymin, xmax, ymax]
                
                words.append(label)
                boxes.append(box)
                labels.append(label)
            
            annotations.append({
                'image_path': image_path,
                'words': words,
                'bbox': boxes,
                'ner_tags': labels
            })

    with open(output_json, 'w') as f:
        json.dump(annotations, f)

# Example usage
annotation_dir = "annotated_folder"  # Directory where XML annotation files are saved
image_dir = "pdf_images"            # Directory where the images are saved
output_json = "dataset.json"    # Output JSON file path
convert_annotations(annotation_dir, image_dir, output_json)
