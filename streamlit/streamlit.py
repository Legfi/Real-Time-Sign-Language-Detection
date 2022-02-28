import streamlit as st
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import keras
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau


st.write("""#Insert text here""")
button = st.button('Start Model')

filepath = '/Users/fyra/github/Real-Time-Sign-Language-Detection/streamlit/sign_mnist_cnn_20_Epochs.h5

model = keras.models.load_model(filepath)

if button:
    
    def getLetter(result):
        classLabels = { 0: 'A',
                        1: 'B',
                        2: 'C',
                        3: 'D',
                        4: 'E',
                        5: 'F',
                        6: 'G',
                        7: 'H',
                        8: 'I',
                        9: 'K',
                        10: 'L',
                        11: 'M',
                        12: 'N',
                        13: 'O',
                        14: 'P',
                        15: 'Q',
                        16: 'R',
                        17: 'S',
                        18: 'T',
                        19: 'U',
                        20: 'V',
                        21: 'W',
                        22: 'X',
                        23: 'Y'}
        try:
            res = int(result)
            return classLabels[res]
        except:
            return "Error"

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        ##############################
        frame=cv2.flip(frame, 1)

        #define region of interest
        roi = frame[100:400, 320:620]
        cv2.imshow('roi', roi)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, (28, 28), interpolation = cv2.INTER_AREA)

        cv2.imshow('roi sacled and gray', roi)
        copy = frame.copy()
        cv2.rectangle(copy, (320, 100), (620, 400), (255,0,0), 5)

        roi = roi.reshape(1,28,28,1) 

        result = str(np.argmax(model.predict(roi, 1, verbose = 0)[0]))
        cv2.putText(copy, getLetter(result), (300 , 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
        cv2.imshow('frame', copy)

        if cv2.waitKey(1) == 13: #13 is the Enter Key
            break

    cap.release()
    cv2.destroyAllWindows()
