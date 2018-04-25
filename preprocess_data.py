import cv2
from tqdm import tqdm
import numpy as np
import scipy.io
import argparse
from config import IMG_SIZE

def main():
    args = get_args()
    f = args.fold
    for k in range(f):
        fold_files_path = './data_set/5_folds/fold_{}_data.txt'.format(k)
        data_root = './data_set/aligned/'
        #load_fold_gender(fold_files_path,data_root,k)
        load_fold_age(fold_files_path,data_root,k)

def get_args():
    parser = argparse.ArgumentParser(description="This script loads images into np array and divide into 5 folds",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--fold",type=int, default=5, help = "total number of folds")

    args = parser.parse_args()
    return args

def load_fold_gender(path,img_root,fold):
    gender_arr = []
    img_arr = []
    lines = []
    with open(path) as f:
        lines = f.readlines()
    lines = lines[1:]
    for i in range(len(lines)):
        lines[i] = lines[i].split('\t')[:5]

    lines = [x for x in lines if x[-1] == 'f' or x[-1] == 'm']
    print("total: {}".format(len(lines)))

    m_count = 0
    f_count = 0
    for i in tqdm(range(len(lines))):
        gender_arr.append(lines[i][-1])
        if lines[i][-1] == 'm':
            m_count += 1
        elif lines[i][-1] == 'f':
            f_count += 1
        p = img_root + "{}/landmark_aligned_face.{}.{}".format(lines[i][0],lines[i][2],lines[i][1])
        img = cv2.imread(p)
        img_arr.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
    print("male: "+ str(m_count))
    print("female: "+ str(f_count))
    output = {"image": np.array(img_arr), "gender": np.array(gender_arr)}
    out_path = "./fold_{}.mat".format(fold)
    scipy.io.savemat(out_path, output)

def load_fold_age(path,img_root,fold):
    age_arr = []
    img_arr = []
    lines = []
    with open(path) as f:
        lines = f.readlines()
    lines = lines[1:]
    for i in range(len(lines)):
        lines[i] = lines[i].split('\t')[:5]
    '''
    count = 0
    for i in lines:
        if i[-2] != '(0, 2)' and i[-2] != '(4, 6)' and i[-2] != '(8, 12)' and i[-2] != '(15, 20)' and i[-2] != '(25, 32)' and i[-2] != '(38, 43)' and i[-2] != '(48, 53)' and i[-2] != '(60, 100)':
            print(i[-2])
            count += 1
    '''


    lines = [x for x in lines if x[-2] == '(0, 2)' or x[-2] == '(4, 6)' or x[-2] == '(8, 12)' or x[-2] == '(15, 20)' or x[-2] == '(25, 32)' or x[-2] == '(38, 43)' or x[-2] == '(48, 53)' or x[-2] == '(60, 100)']
    print("total: {}".format(len(lines)))

    #m_count = 0
    #f_count = 0
    for i in tqdm(range(len(lines))):
        age_arr.append(lines[i][-2])
        #if lines[i][-2] == '(0, 2)':
        #    m_count += 1
        #elif lines[i][-1] == 'f':
        #    f_count += 1
        p = img_root + "{}/landmark_aligned_face.{}.{}".format(lines[i][0],lines[i][2],lines[i][1])
        img = cv2.imread(p)
        img_arr.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
    #print("male: "+ str(m_count))
    #print("female: "+ str(f_count))
    output = {"image": np.array(img_arr), "age": np.array(age_arr)}
    out_path = "./age_fold_{}.mat".format(fold)
    scipy.io.savemat(out_path, output)


main()