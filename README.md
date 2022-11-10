# Real Time Sign Language Detection
## Brief Summary:

In this project we used deep learning for predicting sign language. 
This app has two version! In the first version we classify images and user can drag and drop sing language images and interprate
American sign language letters. 
On version 2 user can classify sign language using camera and real time classification(We didn't used any pretrained model for object detection
for education perposes and only used Open cv for predicting diffrent frames)! 

## Clone the repository
Go to a folder of choice and open a terminal that can access git. Write "git clone https://github.com/Legfi/Real-Time-Sign-Language-Detection.git"

Ofcoures you can use github desktop if you wish to clone the repository in a easier way.
## Create a conda environment
If conda is installed with the environment path variables then it should work with only printing. Else open the Anaconda navigator and open a terminal from there.

then type "conda create --name pythonenv python=3.9"
## Update environment with needed packages
Browse to the git repo folder with the terminal. print "pip install -r requirements.txt"

## Start the python script
Browse to the location of the files.
Then type "streamlit run main.py" When the script is started the default browser will open at localhost:8501 if it doesn't. Open your preferred browser and browse to "localhost:8501" .

## Link to streamlit app incase you want to try the application: 
https://share.streamlit.io/legfi/real-time-sign-language-detection/main/main.py
