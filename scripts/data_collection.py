# Step 1: Collect user info0
#RPi asks about the number of users and takes their names

import os
import constants
import shutil
import record
import pixels
import time
import io
from contextlib import redirect_stdout, redirect_stderr

FULL_AUDIO_LOCATION = constants.FULL_AUDIO_LOCATION
SPLIT_AUDIO_LOCATION = constants.SPLIT_AUDIO_LOCATION
AUDIO_LONG_TERM_STORAGE = '/home/pi/SpeakerRecognitionOnRPi/data/storage/'

shutil.rmtree(FULL_AUDIO_LOCATION)
os.mkdir(FULL_AUDIO_LOCATION)
shutil.rmtree(SPLIT_AUDIO_LOCATION)
os.mkdir(SPLIT_AUDIO_LOCATION)

def ask_about_the_number_of_users():
    num_users = input ("""\n\n### Bobo says: Hello! I am Bobo the vacuum cleaner. 
                      I am happy to meet you! 
                      How many people are there in your household?\n\n""")
    return num_users

def collect_user_name_and_audio(num_users, record_seconds):
    user_names = []
    for i in range(num_users):
        name = input (f"\n\n### Bobo says: Hey person {i+1}, what's your name? ")
        user_names.append(name)
        collected_data_path = os.path.join(FULL_AUDIO_LOCATION, name)
        if os.path.exists(collected_data_path) and os.path.isdir(collected_data_path):
            shutil.rmtree(collected_data_path)
        processed_data_path = os.path.join(SPLIT_AUDIO_LOCATION, name)
        if os.path.exists(processed_data_path) and os.path.isdir(processed_data_path):
            shutil.rmtree(processed_data_path)
        full_recording_filename = f"{FULL_AUDIO_LOCATION}/{name}/{name}_recording.wav"
        os.makedirs(collected_data_path)
        os.makedirs(processed_data_path)
        print("\n\n### Bobo says: Please read out the following text when you see the recording indication:\n")
        print("**************************************")
        print("""Hey Bobo, start cleaning.
                Stop cleaning and return to the dock.
                Show me your schedule.
                Create a new schedule for cleaning every Monday, Wednesday, Friday and Sunday at 9 in the morning.
                Vacuum and mop the living room with water level 2.
                Mop the bathroom with water level 3.
                Vacuum the bedroom in carpet mode.
                Pause.
                Delete all schedules.
                Delete the current map and create a new map of my flat.
                Create a no-go area on the map.""")
        print("***************************************\n")
        print(" . . . Recording . . .")
        f = io.StringIO()
        with redirect_stdout(f), redirect_stderr(f):
            recording_result = record.record_audio(full_recording_filename, record_seconds)
        if recording_result == 0:
            print(f"\n\n### Bobo says: Thanks {name} :) Your audio has been recorded successfully!\n\n")
        else:
            print(f"\n\n### Bobo says: Oh sorry! Something went wrong when recording your audio!\n\n")
    return user_names

