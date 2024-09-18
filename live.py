#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    black = np.zeros((502,860,3), np.uint8)
    _,thresh = cv2.threshold(frame_gray,127,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(black, contours,-1, (0,255,0),1)
    cv2.imshow('contour',black)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




