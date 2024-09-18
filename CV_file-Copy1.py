#!/usr/bin/env python
# coding: utf-8

# ### **<font style="color:rgb(134,19,348)"> Import the Libraries</font>**

# In[1]:


# Import the libraries
import cv2
import numpy as np
# import matplotlib.pyplot as plt



# In[2]:


#read Image
image = cv2.imread('Image.png')
# image = cv2.resize(image, None, fx=0.9,fy=0.9)
#convert image into grayscale mode
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('Image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


# gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#      cv2.imshow('grayscale image', gray)
    
# apply binary thresholding
# ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
# visualize the binary image
# cv2.imshow('Binary image', thresh)
# 
# cv2.imwrite('tree.jpg', thresh)
# cv2.destroyAllWindows()

# display the image
# plt.figure(figsize=[10,10])
# plt.imshow(Gray_image,cmap = 'gray');
# plt.title("original image");
# plt.axis("off")
# plt.show()


# ## **<font style="color:rgb(134,19,348)"> Detecting contours in an image </font>**
# OpenCV saves us the trouble of having to write lengthy algorithms for contour detection and provides a handy function **`findContours()`** that analysis the [topological structure of the binary image by border following](https://www.sciencedirect.com/science/article/abs/pii/0734189X85900167), a contour detection technique developed in 1985.
# 
# The **`findContours()`** functions takes a binary image as input. The foreground is assumed to be white, and the background is assumed to be black. If that is not the case, then you can invert the image using the [**```cv2.bitwise_not()```**](https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga0002cf8b418479f4cb49a75442baee2f) function.
# 
# #### **Function Syntax:**
# 
# 
# > [**```contours, hierarchy =   cv2.findContours(image, mode, method, contours, hierarchy, offset)```**](https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a)
# 
# **Parameters:**
# 
# * **```image```** - It is the input image (8-bit single-channel). Non-zero pixels are treated as 1's. Zero pixels remain 0's, so the image is treated as binary. You can use compare, inRange, threshold, adaptiveThreshold, Canny, and others to create a binary image out of a grayscale or color one.
# 
# * **```mode```** - It is the contour retrieval mode, ( RETR_EXTERNAL, RETR_LIST, RETR_CCOMP, RETR_TREE )
# 
# * **```method```** - It is the contour approximation method. ( CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE, CHAIN_APPROX_TC89_L1, etc )
# 
# * **```offset```** - It is the optional offset by which every contour point is shifted. This is useful if the contours are extracted from the image ROI, and then they should be analyzed in the whole image context.
# 
# **Returns:**
# 
# * **```contours```** - It is the detected contours. Each contour is stored as a vector of points.
# 
# * **```hierarchy```** - It is the optional output vector containing information about the image topology. It has been described in detail in the video above.
# 
# <br>
# 
# We will go through all the important parameters in detail. For now, let's detect some contours in the image that we read above.

# In[3]:


#find all contours in image
contours, hierarchy = cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#display total no of contours
print ("Number of Contours = {}".format(len(contours)))


# ## **<font style="color:rgb(134,19,348)">Visualizing the contours detected</font>**
# As you can see the **`cv2.findContours()`** function was able to correctly detect the 5 external shapes in the image. But to visualize these results we can use the **`cv2.drawContours()`** function which simply draws the contours onto an image.
# 
# #### **Function Syntax:**
# 
# 
# > [**```cv2.drawContours(image, contours, contourIdx, color, thickness, lineType, hierarchy, maxLevel, offset)```**](https://docs.opencv.org/3.4/d6/d6e/group__imgproc__draw.html#ga746c0625f1781f1ffc9056259103edbc)
# 
# **Parameters:**
# 
# * **```image```** - It is the image on which contours are to be drawn.
# * **```contours```** -  It is point vector(s) representing the contour(s). It is usually an array of contours.
# * **```contourIdx```** - It is the parameter, indicating a contour to draw. If it is negative, all the contours are drawn.
# * **```color```** - It is the color of the contours.
# * **```thickness```** - It is the thickness of lines the contours are drawn with. If it is negative (for example, thickness=FILLED ), the contour interiors are drawn.
# * **```lineType```** -  It is the type of line. You can find the possible options [here](https://docs.opencv.org/3.4/d0/de1/group__core.html#gaf076ef45de481ac96e0ab3dc2c29a777).
# * **```hierarchy```** - It is the optional information about hierarchy. It is only needed if you want to draw only some of the contours (see maxLevel ).
# * **```maxLevel```** -  It is the maximal level for drawn contours. If it is 0, only the specified contour is drawn. If it is 1, the function draws the contour(s) and all the nested contours. If it is 2, the function draws the contours, all the nested contours, all the nested-to-nested contours, and so on. This parameter is only taken into account when there is a hierarchy available.
# * **```offset```** - It is the optional contour shift parameter. Shift all the drawn contours by the specified offset=(dx, dy).
# 
# To prevent the original image from being overwritten, we use **`np.copy()`** for drawing the contours on a copy of the image.

# In[4]:


# Read the image in color mode for drawing purposes.
image1_copy = cv2.imread('Image.png')  


# Draw all the contours.
cv2.drawContours(image1_copy, contours, -1, (0,255,0), 3)

# Display the result
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 1000, 1000)  # Adjust the size as neede
cv2.imshow('image', image1_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ## **<font style="color:rgb(134,19,348)">Pre-processing images For Contour Detection</font>**
# 
# As you have seen above that the **`cv2.findContours()`** functions take in as input a single channel binary image, however, in most cases the original image will not be a binary image. Detecting contours in colored images require pre-processing to produce a single-channel binary image that can be then used for contour detection. 
# 
# The two most commonly used techniques for this pre-processing are:
# 
# * **Thresholding based Pre-processing**
# * **Edge Based Pre-processing**
# 
# Below we will see how you can accurately detect contours using these techniques.

# In[5]:


# Read the image
image2 = cv2.imread('tree.jpg') 

# Display the image
cv2.imshow('Original image', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




