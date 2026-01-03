import fitz
import pytesseract
import pdfplumber
from PIL import Image
import io
import json
from pathlib import Path
import os
import sys
sys.stderr = open(os.devnull, "w")




def images(pdf_path):
    doc = fitz.open(pdf_path)
    results = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)

        for img in images:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(image).strip()

            if text:
                results.append({
                    "content": text,
                    "page": page_index + 1,
                    "modality": "image_ocr",
                    "source": Path(pdf_path).name
                })

    return results


def texts(pdf_path):
    doc = fitz.open(pdf_path)
    results = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            results.append({
                "content": text,
                "page": page_num,
                "modality": "text",
                "source": Path(pdf_path).name
            })

    return results

def tables(pdf_path):
    results = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()

            for table in tables:
                table_text = "\n".join([" | ".join(cell if cell else "" for cell in row) for row in table]).strip()
                if table_text:
                    results.append({
                        "content": table_text,
                        "page": page_num,
                        "modality": "table",
                        "source": Path(pdf_path).name
                    })

    return results


def save_all(results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def print_summary(text_res, image_res, table_res):
    print("✅ Extraction Summary")
    print(f"Text pages   : {len(text_res)}")
    print(f"OCR images   : {len(image_res)}")
    print(f"Tables       : {len(table_res)}")
    print(f"Total chunks : {len(text_res) + len(image_res) + len(table_res)}")


if __name__ == "__main__":
    pdf_path = "data/raw_docs/qatar_test_doc.pdf"
    output_path = "data/processed/all_data.json"

    text_results = texts(pdf_path)
    image_results = images(pdf_path)
    table_results = tables(pdf_path)

    all_results = text_results + image_results + table_results

    save_all(all_results, output_path)
    print_summary(text_results, image_results, table_results)
