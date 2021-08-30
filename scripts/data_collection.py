# Step 1: Collect user info0
#RPi asks about the number of users and takes their names

import os
import constants
import shutil
import record
import pixels
import time
import io
import time
from contextlib import redirect_stdout, redirect_stderr
from termcolor import colored
from distutils.dir_util import copy_tree

time_stamp = time.strftime("%Y%m%d-%H%M%S")
FULL_AUDIO_LOCATION = constants.FULL_AUDIO_LOCATION
SPLIT_AUDIO_LOCATION = constants.SPLIT_AUDIO_LOCATION
AUDIO_LONG_TERM_STORAGE = '/home/pi/SpeakerRecognitionOnRPi/data/storage/'
UNKNOWN_CLASS_AUDIO_LOCATION = '/home/pi/SpeakerRecognitionOnRPi/data/storage/Unknown/'

shutil.rmtree(FULL_AUDIO_LOCATION)
os.mkdir(FULL_AUDIO_LOCATION)
shutil.rmtree(SPLIT_AUDIO_LOCATION)
os.mkdir(SPLIT_AUDIO_LOCATION)
#Add the prepared data for the Unknown class to the SPLIT_AUDIO_LOCATION
os.mkdir(f"{SPLIT_AUDIO_LOCATION}/Unknown")
copy_tree(UNKNOWN_CLASS_AUDIO_LOCATION, f"{SPLIT_AUDIO_LOCATION}/Unknown/")


def ask_about_the_number_of_users():
    num_users = input (colored("""\n\n### Bobo says: Hello! I am Bobo the vacuum cleaner. 
                      I am happy to meet you! 
                      How many people are there in your household?\n\n""", 'yellow', attrs=["bold"]))
    return num_users

def collect_user_name_and_audio(num_users, record_seconds):
    user_names = []
    for i in range(num_users):
        name = input (colored(f"\n\n### Bobo says: Hey person {i+1}, what's your name? ", "yellow", attrs=["bold"]))
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
        print(colored("\n\n### Bobo says: Please read out the following text when you see the recording indication:\n", "yellow", attrs=["bold"]))
        print(colored("**************************************", "cyan", attrs=["bold"]))
        print(colored("Hey Bobo, start cleaning.\nStop cleaning and return to the dock.\nShow me your schedule.\nCreate a new schedule for cleaning every Monday, Wednesday, Friday and Sunday at 9 in the morning.\nVacuum and mop the living room with water level 2.\nMop the bathroom with water level 3.\nVacuum the bedroom in carpet mode.\nPause.\nDelete all schedules.\nDelete the current map and create a new map of my flat.\nCreate a no-go area on the map.", "cyan", attrs=["bold"]))
        print(colored("***************************************\n", "cyan", attrs=["bold"]))
        print(colored(" . . . Recording . . .", "cyan", attrs=["bold"]))
        f = io.StringIO()
        with redirect_stdout(f), redirect_stderr(f):
            recording_result = record.record_audio(full_recording_filename, record_seconds)
        if recording_result == 0:
            print(colored(f"\n\n### Bobo says: Thanks {name} :) Your audio has been recorded successfully!\n\n", "yellow", attrs=["bold"]))
        else:
            print(colored(f"\n\n### Bobo says: Oh sorry! Something went wrong when recording your audio!\n\n", "yellow", attrs=["bold"]))
    #copy recording to the storage directory
    shutil.copyfile(full_recording_filename, f"{AUDIO_LONG_TERM_STORAGE}/{name}_recording_{time_stamp}.wav")
    return user_names

