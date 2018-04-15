import cv2
import time
from my_cnn import CNN

'''
img = cv2.imread('/home/rui/2017winter/honoursProject/project/data_set/aligned/100003415@N08/landmark_aligned_face.2174.9523333835_c7887c3fde_o.jpg')
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

cnn  = CNN(input_size=128)
cnn.plot_model()