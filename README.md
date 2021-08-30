# SpeakerRecognitionOnRPi

This project contains a speaker recognition model training pipeline which is intended to run on a Raspberry Pi equipped with a ReSpeaker microphone array.

It is based on this tutorial: https://keras.io/examples/audio/speaker_recognition_using_cnn/

The code for audio recording and LED manipulation comes from https://github.com/respeaker/4mics_hat

ReSpeaker documentation: https://wiki.seeedstudio.com/ReSpeaker_4_Mic_Array_for_Raspberry_Pi/

The tools used for model training are TensorFlow and Keras. They are used directly on the Raspberry Pi.

To run the pipeline follow these steps:
```
cd SpeakerRecognitionOnRPi/scripts
pip install -r requirements.txt
python pipeline.py
```

This script gives the user instructions on the screen.
It performs this sequence of steps:
- Data collection - the script gives the user a short text which they are asked to read and records them for 35 seconds.
- Data processing - the collected data is processed to be compatible with the model's input requirements
- Model training - a speaker recognition model with pre-defined architecture and parameters is trained for 100 epochs.
- The script does real time testing: the scripts asks the user(s) to say something to the RPi and prints a greeting with
the predicted user's name. It does this 4 times.
- The trained model is saved as an h5 file in the `scripts` folder. The model filename contains a timestamp for easy identification.

Note: The pipeline automatically adds an `Unknown` class to the data collected from the users.
The data for the "Unknown" class was downloaded from https://librivox.org

Project Paper: https://drive.google.com/file/d/1bdu-ovMm4BUT_BK6bRo4N47xcoCs2SZ-/view?usp=sharing
