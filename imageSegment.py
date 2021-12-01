# -*- coding: utf-8 -*-
"""
imageSegment.py

YOUR WORKING FUNCTION

"""
import cv2
import numpy as np  


input_dir = 'dataset/test'
output_dir = 'dataset/output'

# you are allowed to import other Python packages above
##########################
# you are allowed to import other Python packages above
##########################
def segmentImage(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert colro to rgb 

    #sharpening the image 
    gaussian_blur_img = cv2.GaussianBlur(rgb_img,(5,5),0) #Gaussian as choose compare to others 
    avg_blur_img = cv2.blur(gaussian_blur_img, (5,5)) #blur again the gaussian filter's image     
    details = gaussian_blur_img.astype("float32") - avg_blur_img.astype("float32")
    shp = gaussian_blur_img.astype("float32") + details
    shp = np.clip(shp, 0, 255).astype("uint8")
    #hsv image 
    hsv_img = cv2.cvtColor(shp, cv2.COLOR_RGB2HSV)

    #black mask (hair and eye brows)
    black_low = np.array([0, 0, 0])
    black_high = np.array([180, 255, 60]) 
    hsv_black = cv2.inRange(hsv_img, black_low, black_high)
    morph_black = cv2.dilate(hsv_black, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),iterations=1)
    morph_black = cv2.morphologyEx(morph_black, cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1)

    #red mask (mouth) 
    red_low1 = np.array([0, 70, 50])
    red_high1 = np.array([10, 255, 255]) 
    hsv_red1 = cv2.inRange(hsv_img, red_low1, red_high1)
    red_low2 = np.array([170, 70, 50])
    red_high2 = np.array([180, 255, 255])
    hsv_red2 = cv2.inRange(hsv_img, red_low2, red_high2)
    hsv_red = np.logical_or(hsv_red1,hsv_red2)
    hsv_red = hsv_red.astype("uint8") #convert the true false back to unit8
    kernel = np.ones((3,3),np.uint8)
    mouth_erode = cv2.erode(hsv_red,kernel,iterations = 2)
    mask = np.zeros(mouth_erode.shape[:2], np.uint8)
    mask[350:420, 150:250] = 255
    res, mask = cv2.threshold(mask, 70, 100, cv2.THRESH_BINARY)
    cv2.floodFill(mask, None, (0,0), 255)
    cv2.floodFill(mask, None, (0,0), 0)
    mouth_erode = cv2.bitwise_and(mouth_erode,mouth_erode,mask = mask)
    
    #red mask (skin)
    red_low1 = np.array([0, 50, 50])
    red_high1 = np.array([20, 255, 255])
    red_low2 = np.array([150, 100, 100])
    red_high2 = np.array([179, 255, 255])
    lowerMask = cv2.inRange(hsv_img, red_low1, red_high1)
    upperMask = cv2.inRange(hsv_img, red_low2, red_high2)
    skin_mask = lowerMask + upperMask
    kernel = np.ones((7,7),np.uint8)
    skin_erosion = cv2.erode(skin_mask,kernel,iterations = 1)
    skin_morph = cv2.morphologyEx(skin_erosion, cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)), iterations=1) #Close holes, keep shape
    
    #eyes mask 
    bilateral = cv2.bilateralFilter(avg_blur_img, 9, 75, 75)
    mask_eyes = np.zeros(bilateral.shape[:2], np.uint8)
    mask_eyes[220:270, 70:330] = 255
    masked_eyes_img = cv2.bitwise_and(bilateral,bilateral,mask = mask_eyes)
    #bilateral can preserve the edges although it blur the image
    edges_eyes = cv2.Canny(masked_eyes_img,10,50)
    #remove the padding from the eyes 
    res, masked_eyes_img = cv2.threshold(edges_eyes, 90, 100, cv2.THRESH_BINARY)
    cv2.floodFill(edges_eyes, None, (0,0), 255)
    cv2.floodFill(edges_eyes, None, (0,0), 0)
    eyes_dilate = cv2.dilate(edges_eyes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),iterations=2)
    eyes_morph = cv2.morphologyEx(eyes_dilate, cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11)), iterations=1)

    #detect nose 
    #detect nose 
    mask_nose = np.zeros(bilateral.shape[:2], np.uint8)
    mask_nose[270:350, 160:240] = 255
    masked_nose_img = cv2.bitwise_and(bilateral,bilateral,mask = mask_nose)
    edges_nose = cv2.Canny(masked_nose_img,10,30)
    #remove the padding from the eyes 
    res, masked_nose_img = cv2.threshold(edges_nose, 180, 200, cv2.THRESH_BINARY)
    cv2.floodFill(edges_nose, None, (0,0), 255)
    cv2.floodFill(edges_nose, None, (0,0), 0)
    nose_dilate = cv2.dilate(edges_nose, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),iterations=2)
    nose_morph = cv2.morphologyEx(nose_dilate, cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)), iterations=2) #Close holes, keep shape

    hsv_mask = skin_morph.copy() 
    hsv_mask[morph_black == 255] = ([1]) 
    hsv_mask[mouth_erode == 1] = ([2]) 
    hsv_mask[eyes_morph == 255] = ([3]) 
    hsv_mask[nose_morph == 255] = ([4]) 
    hsv_mask[hsv_mask == 255] = ([5]) 
    outImg = hsv_mask

    return outImg   
