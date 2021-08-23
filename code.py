import cv2
import numpy as np
import imutils
import easyocr
from matplotlib import pyplot as pl

img = cv2.imread("nomer-777vor.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #convert to gray image

img_filter = cv2.bilateralFilter(gray, 11, 15, 15)  #put a blur picture
edges = cv2.Canny(img_filter, 30, 200)  #find the counter|part 1

cont = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   #find the counter|part 2
cont = imutils.grab_contours(cont)  #take the finded counters
cont = sorted(cont, key=cv2.contourArea, reverse=True)  #sort the couters finded

pos = None
for c in cont:
    approx = cv2.approxPolyDP(c, 15, True)
    if len(approx) == 4:  #if we have 4 counters that located nearby
        pos = approx  #and put it to pos
        break

mask = np.zeros(gray.shape, np.uint8)
new_img = cv2.drawContours(mask, [pos], 0, 255, -1)
bitwise_img = cv2.bitwise_and(img, img, mask=mask)  #put this pos on the picture with mask

(x, y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))  #find the lower point of the number
(x2, y2) = (np.max(x), np.max(y))  #find the highest point of the number
crop = gray[x1:x2, y1:y2]

text = easyocr.Reader(['en'])
text = text.readtext(crop)  #read the text on this number
#print(text)
pl.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))  #show the number
pl.show()
