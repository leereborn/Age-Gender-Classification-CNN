import argparse
from scipy.io import loadmat
import numpy as np
from my_cnn import CNN
from cnn_age import AgeCNN
from keras import utils
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from config import *
from utils import mk_dir
import time

def get_args():
    parser = argparse.ArgumentParser(description="Train the cnn model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", type=int, default = 0,
                        help = "choose one of the fold as validation set")
    parser.add_argument("-t", type=int, default = 0,
                        help = "choose between gender (0) and age (1) to train")
    args = parser.parse_args()
    return args

def train_gender():
    args = get_args()
    validation_fold = args.v
    test_set = loadmat('./fold_{}.mat'.format(validation_fold))
    test_x = test_set['image']
    test_y = test_set['gender']
    fold_set = [0,1,2,3,4]
    fold_set.remove(validation_fold)
    l = []
    for i in fold_set:
        l.append(loadmat('./fold_{}.mat'.format(i)))
    train_x = l[0]['image']
    train_y = l[0]['gender']
    for i in range(1,len(l)):
        train_x = np.concatenate((train_x,l[i]['image']))
        train_y = np.concatenate((train_y,l[i]['gender']))

    #test_y = utils.to_categorical(test_y,num_classes=2)
    #train_y = utils.to_categorical(test_y,num_classes=2)
    print(test_y)
    for i, value in np.ndenumerate(test_y):
        if value == 'm':
            np.put(test_y,i,0)
        elif value == 'f':
            np.put(test_y,i,1)
        else:
            raise Exception
    print(test_y)

    for i, value in np.ndenumerate(train_y):
        if value == 'm':
            np.put(train_y,i,0)
        elif value == 'f':
            np.put(train_y,i,1)
        else:
            raise Exception

    cnn = CNN(input_size = IMG_SIZE)
    model = cnn.get_classifier()
    
    if IMG_AUG == 1:
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
    else:
        train_datagen = ImageDataGenerator(rescale = 1./255)
        
    test_datagen = ImageDataGenerator(rescale = 1./255)
    training_set = train_datagen.flow(train_x,train_y,batch_size=BATCH_SIZE)
    test_set = test_datagen.flow(test_x,test_y,batch_size=BATCH_SIZE)

    mk_dir("checkpoints")
    callbacks = [ModelCheckpoint("checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto") 
                ]
    hist = model.fit_generator(training_set, steps_per_epoch = len(train_x)//BATCH_SIZE, epochs = NUM_EPOCHS, validation_data = test_set, validation_steps = len(test_set),callbacks=callbacks)
    
    model.save('my_cnn.h5')
    mk_dir("history")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    pd.DataFrame(hist.history).to_hdf("./history/history-{}.h5".format(timestr), "history")

def train_age():
    args = get_args()
    validation_fold = args.v
    test_set = loadmat('./age_fold_{}.mat'.format(validation_fold))
    test_x = test_set['image']
    test_y = test_set['age']
    test_y = [x.strip() for x in test_y] # since Matrices are of rectangular shapes in matlab
    test_y = np.array(test_y)

    fold_set = [0,1,2,3,4]
    fold_set.remove(validation_fold)
    l = []
    for i in fold_set:
        l.append(loadmat('./age_fold_{}.mat'.format(i)))
    train_x = l[0]['image']
    train_y = l[0]['age']
    for i in range(1,len(l)):
        train_x = np.concatenate((train_x,l[i]['image']))
        train_y = np.concatenate((train_y,l[i]['age']))
    train_y = [x.strip() for x in train_y]
    train_y = np.array(train_y)

    print(test_y[:50])
    for i, value in np.ndenumerate(test_y):
        if value == '(0, 2)':
            np.put(test_y,i,0)
        elif value == '(4, 6)':
            np.put(test_y,i,1)
        elif value == '(8, 12)':
            np.put(test_y,i,2)
        elif value == '(15, 20)':
            np.put(test_y,i,3)
        elif value == '(25, 32)':
            np.put(test_y,i,4)
        elif value == '(38, 43)':
            np.put(test_y,i,5)
        elif value == '(48, 53)':
            np.put(test_y,i,6)
        elif value == '(60, 100)':
            np.put(test_y,i,7)
        else:
            print(value)
            print(len(value))
            raise Exception
    print(test_y)

    for i, value in np.ndenumerate(train_y):
        if value == '(0, 2)':
            np.put(train_y,i,0)
        elif value == '(4, 6)':
            np.put(train_y,i,1)
        elif value == '(8, 12)':
            np.put(train_y,i,2)
        elif value == '(15, 20)':
            np.put(train_y,i,3)
        elif value == '(25, 32)':
            np.put(train_y,i,4)
        elif value == '(38, 43)':
            np.put(train_y,i,5)
        elif value == '(48, 53)':
            np.put(train_y,i,6)
        elif value == '(60, 100)':
            np.put(train_y,i,7)
        else:
            print(value)
            print(len(value))
            raise Exception
    test_y = utils.to_categorical(test_y,num_classes=8)
    train_y = utils.to_categorical(train_y,num_classes=8)

    #import pdb; pdb.set_trace()

    cnn = AgeCNN(input_size = IMG_SIZE)
    model = cnn.get_classifier()
    
    if IMG_AUG == 1:
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
    else:
        train_datagen = ImageDataGenerator(rescale = 1./255)
        
    test_datagen = ImageDataGenerator(rescale = 1./255)
    training_set = train_datagen.flow(train_x,train_y,batch_size=BATCH_SIZE)
    test_set = test_datagen.flow(test_x,test_y,batch_size=BATCH_SIZE)

    mk_dir("checkpoints")
    callbacks = [ModelCheckpoint("checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto") 
                ]
    hist = model.fit_generator(training_set, steps_per_epoch = len(train_x)//BATCH_SIZE, epochs = NUM_EPOCHS, validation_data = test_set, validation_steps = len(test_set),callbacks=callbacks)
    
    model.save('age_cnn.h5')
    mk_dir("history")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    pd.DataFrame(hist.history).to_hdf("./history/age_history-{}.h5".format(timestr), "history")

if __name__ == '__main__':
    args = get_args()
    model = args.t
    if model == 0:
        train_gender()
    else:
        train_age()