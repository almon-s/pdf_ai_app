from pdf2image import convert_from_path
import os

def convert_pdf_to_images(pdf_files, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for pdf_file in pdf_files:
        # Extract the base name of the PDF file (without extension)
        base_name = os.path.splitext(os.path.basename(pdf_file))[0]
        
        # Convert PDF to images
        images = convert_from_path(pdf_file)
        
        # Save each page as an image
        for i, image in enumerate(images):
            image_path = os.path.join(output_folder, f"{base_name}_page_{i+1}.png")
            image.save(image_path, "PNG")
            print(f"Saved {image_path}")

# Example usage
pdf_files = ["/Users/almonsubba/Desktop/pdf_app/pdf_folder/sep23cloudflare.pdf", "/Users/almonsubba/Desktop/pdf_app/pdf_folder/sept23Alchemy.pdf", "pdf_folder/sept23figma.pdf"]
output_folder = "annotated_folder"
convert_pdf_to_images(pdf_files, output_folder)
