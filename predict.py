import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import argparse
from config import IMG_SIZE
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
from keras import utils
from utils import plot_confusion_matrix
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="./gender_cnn.h5",
                        help="path to the trained model file")
    parser.add_argument("--image", type=str, default= None,
                        help="path to the image file")
    parser.add_argument("-p", type=int, default= 0,
                        help="0 is gender and 1 is age")
    args = parser.parse_args()
    return args

def draw_confusion_matrix():
    #args = get_args()
    #model_path = args.model
    model = load_model('./age_cnn.h5')
    test_sample = loadmat('./age_fold_0.mat')
    test_x = test_sample['image']
    test_y = test_sample['age']
    print(test_y)
    for i, value in np.ndenumerate(test_y):
        if value == '(0, 2)   ':
            np.put(test_y,i,0)
        elif value == '(4, 6)   ':
            np.put(test_y,i,1)
        elif value == '(8, 12)  ':
            np.put(test_y,i,2)
        elif value == '(15, 20) ':
            np.put(test_y,i,3)
        elif value == '(25, 32) ':
            np.put(test_y,i,4)
        elif value == '(38, 43) ':
            np.put(test_y,i,5)
        elif value == '(48, 53) ':
            np.put(test_y,i,6)
        elif value == '(60, 100)':
            np.put(test_y,i,7)
        else:
            print(value)
            print(len(value))
            raise Exception
    test_y=test_y.astype(int)
    print(test_y)
    rounded_predictions = model.predict_classes(test_x,batch_size=10)
    print(rounded_predictions)
    cm = confusion_matrix(test_y,rounded_predictions)
    print(cm)
    classes = ['(0,2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
    plt.figure()
    plot_confusion_matrix(cm,classes,normalize=True, title='Age Confusion matrix')
    plt.show()

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
            prediction = 'female'
        else:
            prediction = 'male'

        print(prediction)
        print(result[0][0])
    else:
        print(result[0])
        print(np.argmax(result[0]))
        print(get_age(np.argmax(result[0])))
        print(get_label(np.argmax(result[0])))     

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

def get_label(prediction):
    if prediction <= 1:
        age_display = "baby"
    elif prediction ==2:
        age_display = "child"
    elif prediction ==3:
        age_display = "youth"
    elif prediction >=4 and prediction <=6:
        age_display = "adult"
    else:
        age_display = "senior"
    return age_display

if __name__ == '__main__':
    main()
    #draw_confusion_matrix()