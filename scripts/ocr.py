import cv2
import easyocr
import matplotlib.pyplot as plt



#read image should work on any device
image= cv2.imread('data/processed/1954-P000002_page-0001_denoised.png')
img = cv2.imread('image')




#the reading code 
reader = easyocr.Reader(['arb'],)
reader.readtext(img)
