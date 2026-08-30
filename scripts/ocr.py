import pytesseract
from PIL import Image

image = Image.open("data/cropped/1954-P000002_page-0001_denoised_header_part.png")

text = pytesseract.image_to_string(
    image,
    lang="ara",
    config="--psm 4"
)

with open("output/ocr/ocr_result.txt", "w", encoding="utf-8") as file:
    file.write(text)

print("OCR completed!")