import os

for i in range(0,5):
    os.system("python train.py --validation_fold {}".format(str(i)))