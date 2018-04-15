import argparse
from scipy.io import loadmat
import numpy as np
from my_cnn import CNN
from keras import utils
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from config import *
import os
from utils import mk_dir

def get_args():
    parser = argparse.ArgumentParser(description="Train the cnn model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--aug", action="store_true",
                        help="Use data augmentation if set true")
    parser.add_argument("--validation_fold", type=int, default = 0,
                        help = "choose one of the fold as validation set")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    validation_fold = args.validation_fold
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

    train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
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
    pd.DataFrame(hist.history).to_hdf("history.h5", "history")

main()