# -*- coding: utf-8 -*-
import numpy as np
from keras.models import load_model
from keras.preprocessing import image


gender_model = load_model("my_cnn.h5")
test_image = image.load_img('/home/rui/2017winter/honoursProject/project/data_set/aligned/100003415@N08/landmark_aligned_face.2174.9523333835_c7887c3fde_o.jpg', target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0) # the third dimenssion is for batch
result = gender_model.predict(test_image)
#training_set.class_indices
if result[0][0] == 1:
    prediction = 'f'
else:
    prediction = 'm'

print(prediction)