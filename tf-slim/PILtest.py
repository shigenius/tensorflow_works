from PIL import Image
import numpy as np
import cv2

im = np.array(Image.open('/Users/shigetomi/Desktop/1.png'))
print(im)

im_cv = cv2.cvtColor(cv2.imread('/Users/shigetomi/Desktop/1.png'), cv2.COLOR_BGR2RGB)
print('\n')
print("im_cv", im_cv)