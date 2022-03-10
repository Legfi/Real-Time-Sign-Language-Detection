# Imports

from PIL import Image
import cv2
import tensorflow as tf
import numpy as np
import base64
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av

# Functions
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('our_model_20_Epochs.hdf5', compile=False)
    return model

def reshape_image(img):
    basewidth = 28
    img = img.convert('L')
    width_percent = (basewidth / float(img.size[0]))
    hight_size = int((float(img.size[1]) * float(width_percent)))
    if img.size[0] != img.size[1]:
        hight_size = basewidth
    img = img.resize((basewidth, hight_size), Image.ANTIALIAS)
    img_np = np.array(img) / 255 # noramlize
    img_np.reshape(1, 28, 28, 1)
    img_np = np.array([img_np]) # add dimesion

    return img_np

def my_streamlit():
    
    #Title
    st.title("'Everyone has a Chance'")
    
    # Background image
    image = Image.open('American-Sign-Language-ASL-Universal.jpg')
    st.image(image, caption="Sign Language for everyone!", use_column_width=True)

    #informatin for user
    st.write("""### This is a school project helping pueple who use Sign Language. Using this app you can easily interpret sign language to English without needing a translater and only using machine learning...""")
    st.subheader('Our application has 2 versions:')
    st.write("""#### In version 1 you can translate images of sign language. You don't have any photo? No problem. You can use our version 2 and use your camera and translate sign language. you can pick which version you want to use on the sidebar! """)
    
    purpose = st.sidebar.selectbox("which version would you like to use?",("Version1", "Version2"))

    if purpose == "Version1":

        model1 = load_model()
        #Demo of version 1
        Demo = st.button("Demo")
        if Demo == True:
            Stop = st.button("Stop Demo")
            file_ = open("sign-language-Visual-Studio-Code-2022-02-22-12-18-04.gif", "rb")
            contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            file_.close()
            st.markdown(
                f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
                unsafe_allow_html=True,)
            if Stop:
                Demo == False
        
        # Create list of labels
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
                'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

        # Load image and assinge it to a variable 
        file = st.file_uploader(label = 'Upload an image', type = ['png', 'jpg'] )

        # When images is loaded condition is met
        if file is not None:
            img = Image.open(file)
            col1, col2, col3 = st.columns([0.2, 0.4, 0.2])
            col2.image(img, use_column_width=True)
            reshaped_image = reshape_image(img)

            prediction = model1.predict(reshaped_image)[0]
            prediction_index = np.argmax(prediction)
            if prediction[prediction_index ] > 0.9 :
                label = labels[prediction_index]
                st.write(f'The model predicts letter = {label}')
            else:
                st.write('Model foud no label match')
                
    #Version2 of the App
    if purpose == "Version2":
        model1 = load_model()
        #Demo of version2
        Demo = st.button("Demo")
        if Demo == True:
            Stop = st.button("Stop Demo")
            file_ = open("Real time classification.gif", "rb")
            contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            file_.close()
            st.markdown(
                f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
                unsafe_allow_html=True,)
            if Stop:
                Demo == False
        
        #checkbox for starting the application
        """We had some issues during the deploymet of this app becouse Opencv doesn't let our app access the local camera of our users. But our genius developers came with the idea of using another library which called "webrtc"! Right now we are working to give even better service to our users using object detection models"""
        """Click on start button and begin interprating(This might take a few sec! please w8!):"""
        class videoprocessor:
            def recv(self, frame):
                frm = frame.to_ndarray(format="bgr24")
                frm = cv2.flip(frm, 1)
                roi = frm[100:400, 320:620]
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi = cv2.resize(roi, (28, 28), interpolation = cv2.INTER_AREA)
                cv2.rectangle(frm, (320, 100), (620, 400), (255,0,0), 5)
                roi = roi.reshape(1,28,28,1)
                result = str(np.argmax(model1.predict(roi, 1, verbose = 0)[0]))
                cv2.putText(frm, getLetter(result), (300 , 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
                return av.VideoFrame.from_ndarray(frm, format="bgr24")
        
        webrtc_streamer(key="key", video_processor_factory=videoprocessor,
				rtc_configuration=RTCConfiguration(
					{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
					)
	    )

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