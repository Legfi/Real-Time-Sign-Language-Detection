# Imports

from PIL import Image
import cv2
import streamlit as st
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
import numpy as np

# Functions

def my_streamlit():
    
    #Title
    st.title("'Everyone has a Chance' Algozobi")
    
    # Background image
    image = Image.open('streamlit/Is-American-Sign-Language-ASL-Universal.jpg')
    st.image(image, caption="Sign Language for everyone!", use_column_width=True)

    #informatin for user
    st.write("""### This is a school project for helping peaple who use Sign Language. With using this app you can easily interpret sign language to English without need to a translater and onlyusing machine learning...""")
    
    st.subheader('To start you need to click on the start button: ')
    
    #checkbox for starting the application
    start_button = st.button('Start')
    stop_button = st.button('stop')
    #run = st.checkbox('Start')
    FRAME_WINDOW = st.image([])
    cam = cv2.VideoCapture(0)
    @st.cache(allow_output_mutation=True)
    def load_model():
        model = tf.keras.models.load_model('our_model_20_Epochs.hdf5', compile=False)
        return model
    model1 = load_model()
    
    while start_button == True:

        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = frame[100:400, 320:620]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, (28, 28), interpolation = cv2.INTER_AREA)
        copy = frame.copy()
        cv2.rectangle(copy, (320, 100), (620, 400), (255,0,0), 5)
        roi = roi.reshape(1,28,28,1) 

        result = str(np.argmax(model1.predict(roi, 1, verbose = 0)[0]))
        cv2.putText(copy, getLetter(result), (300 , 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
        FRAME_WINDOW.image(copy)
    if stop_button:
        st.balloons()
        


#Create function to match label to letter
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




def main():
    """When running this script this is the main module that handles the appplications logic"""
    
    my_streamlit()



# Mainmethod

if __name__ == "__main__":
    main()
