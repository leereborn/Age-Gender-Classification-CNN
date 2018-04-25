import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import argparse
from config import IMG_SIZE

def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="./my_cnn.h5",
                        help="path to the trained model file")
    parser.add_argument("--image", type=str, default= None,
                        help="path to the trained model file")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    model_path = args.model
    image_path = args.image

    gender_model = load_model(model_path)
    test_image = image.load_img(image_path, target_size = (IMG_SIZE, IMG_SIZE))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0) # the third dimenssion is for batch
    result = gender_model.predict(test_image)
    #training_set.class_indices
    if result[0][0] == 1:
        prediction = 'f'
    else:
        prediction = 'm'

    print(prediction)
    print(result[0][0])

main()