# Step 1: Collect user info0
#RPi asks about the number of users and takes their names

import os
import shutil
import record
import pixels
import time

FULL_AUDIO_LOCATION = '/home/pi/SpeakerRecognitionOnRPi/data/collected_audio/'
SPLIT_AUDIO_LOCATION = '/home/pi/SpeakerRecognitionOnRPi/data/audio/'

def ask_about_the_number_of_users():
    num_users = input ("Hello! I am Bobo the vacuum cleaner. I am happy to meet you! How many people are there in your household?\n")
    return num_users

def collect_user_name_and_audio(num_users):
    user_names = []
    for i in range(num_users):
        name = input (f"Hey person {i}, what's your name? ")
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
        print("Please read out the following text when you see the recording indication:")
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
        recording_result = record.record_audio(full_recording_filename)
        if recording_result == 0:
            print(f"Thanks {name} :) Your audio has been recorded successfully!")
        else:
            print(f"Oh sorry! Something went wrong when recording your audio!")


        

if __name__ == "__main__":
    num_users = ask_about_the_number_of_users()
    collect_user_name_and_audio(int(num_users))
