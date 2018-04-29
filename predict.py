import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import argparse
from config import IMG_SIZE
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="./my_cnn.h5",
                        help="path to the trained model file")
    parser.add_argument("--image", type=str, default= None,
                        help="path to the trained model file")
    parser.add_argument("-p", type=int, default= 0,
                        help="o is gender and 1 is age")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    model_path = args.model
    image_path = args.image
    pred = args.p

    model = load_model(model_path)
    test_image = image.load_img(image_path, target_size = (IMG_SIZE, IMG_SIZE))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0) # the third dimenssion is for batch
    result = model.predict(test_image)

    if pred == 0:
        #training_set.class_indices
        if result[0][0] == 1:
            prediction = 'f'
        else:
            prediction = 'm'

        print(prediction)
        print(result[0][0])
    else:
        print(result[0])
        print(np.argmax(result[0]))
        print(get_age(np.argmax(result[0])))     

def get_age(result):
    if result == 0:
        return '(0, 2)'
    elif result == 1:
        return '(4, 6)'
    elif result == 2:
        return '(8, 12)'
    elif result == 3:
        return '(15, 20)'
    elif result == 4:
        return '(25, 32)'
    elif result == 5:
        return '(38, 43)'
    elif result == 6:
        return '(48, 53)'
    elif result == 7:
        return '(60, 100)'

#main()