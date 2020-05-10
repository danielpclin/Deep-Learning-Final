import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

filename = 'dev/data01_dev/000007.jpg'
img = cv2.imread(filename)
clone = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (11, 11), 0)

binaryIMG = cv2.Canny(blurred, 20, 160)


imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img = cv2.drawContours(img, contours, -1, (0,255,0), 3)

cv2.imshow('1', img)
cv2.imshow('2', gray)
cv2.imshow('3', blurred)
cv2.imshow('4', binaryIMG)

cv2.waitKey(0)
cv2.destroyAllWindows()
