import pytesseract
from PIL import Image

croped_image = Image.open('output/crops/P000002_header_printed.png')
text = pytesseract.image_to_string(croped_image, lang='ara', config='--psm 4')
print(text)