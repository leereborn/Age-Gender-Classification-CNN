# This script is imported from https://github.com/yu4u/age-gender-estimation
import os
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from keras.models import load_model
from keras.preprocessing import image
from config import IMG_SIZE
from predict import get_age

def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--width", type=int, default=1080,
                        help="Width of the display")
    parser.add_argument("--height", type=int, default=720,
                        help="Height of the display")
    args = parser.parse_args()
    return args

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=2, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images(w,h):
    # capture video
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img


def main():
    args = get_args()
    WIDTH = args.width
    HEIGHT = args.height
    model = load_model("gender_cnn.h5")
    age_model = load_model("age_cnn.h5")

    # for face detection
    detector = dlib.get_frontal_face_detector()
    img_size = IMG_SIZE
    for img in yield_images(WIDTH,HEIGHT):
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))

        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - 0.4 * w), 0)
                yw1 = max(int(y1 - 0.4 * h), 0)
                xw2 = min(int(x2 + 0.4 * w), img_w - 1)
                yw2 = min(int(y2 + 0.4 * h), img_h - 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

            # predict ages and genders of the detected faces
            gender_result = model.predict(faces)
            age_result = age_model.predict(faces)
            if gender_result[0][0] == 1:
                gender_prediction = 'female'
            else:
                gender_prediction = 'male'
            
            prediction = np.argmax(age_result[0])
            age_range = get_age(prediction)
            print(gender_prediction)
            print(age_range)

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

            # draw results
            for i, d in enumerate(detected):
                label = gender_prediction + " " +  age_display
                draw_label(img, (d.left(), d.top()), label)


        cv2.imshow("Age and Gender Estimation Demo", img)
        key = cv2.waitKey(30)

        if key == 27: #Press esc to exit
            break
main()