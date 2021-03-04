import numpy as np
import cv2

img = np.zeros(shape=(512,512,3), dtype = np.int8)
while True:
    cv2.imshow('hh', img)
    if cv2.waitKey(1) == 27: break
    
cv2.destroyAllWindows()