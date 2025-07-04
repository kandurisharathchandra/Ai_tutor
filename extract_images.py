import fitz  # PyMuPDF
import os

def extract_images_from_pdf(pdf_path, output_folder="data/images"):
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    all_images = []
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = images[img_index][0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            filename = f"{os.path.basename(pdf_path).split('.')[0]}_page{page_index+1}_{img_index}.{image_ext}"
            filepath = os.path.join(output_folder, filename)
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            all_images.append(filepath)
    return all_images
