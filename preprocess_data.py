import cv2
from tqdm import tqdm
import numpy as np
import scipy.io
import argparse
from config import IMG_SIZE

def main():
    args = get_args()
    f = args.fold
    fold_files_path = '../data_set/5_folds/fold_{}_data.txt'.format(f)
    data_root = '../data_set/aligned/'

    load_fold_gender(fold_files_path,data_root,f)

def get_args():
    parser = argparse.ArgumentParser(description="This script loads images into np array and divide into 5 folds",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--fold",type=str, default=0, help = "choose the fold to load")

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

    print(len(lines))
    lines = [x for x in lines if x[-1] == 'f' or x[-1] == 'm']
    print(len(lines))

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
main()