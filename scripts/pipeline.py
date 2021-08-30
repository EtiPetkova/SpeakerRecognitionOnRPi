import constants
import data_collection
import data_prep
import train
import predict
import pixels
import time
from termcolor import colored


def run_pipeline():
    leds = pixels.Pixels()
    leds.wakeup()
    time.sleep(1)
    leds.off()
    leds.wakeup()
    time.sleep(1)
    leds.off()
    num_users = data_collection.ask_about_the_number_of_users()
    leds.wakeup()
    leds.off()
    leds.think()
    user_names = data_collection.collect_user_name_and_audio(int(num_users), constants.RECORD_SECONDS_TRAINING)
    remove_sil_and_split = data_prep.remove_silence_and_split_audio(user_names, constants.RAW_AUDIO)
    if remove_sil_and_split != 0:
        exit("[ERROR] Something went wrong in splitting the audio")
    train_ds, valid_ds, test_ds = data_prep.create_datasets(data_prep.DATASET_AUDIO_PATH, constants.VALID_SPLIT, constants.TEST_SPLIT)
    leds.off()
    time.sleep(2)
    leds.speak()
    trained_model_filename = train.train_model(train_ds, valid_ds, test_ds)
    leds.off()
    if trained_model_filename:
        leds.wakeup()
        time.sleep(1)
        print(colored("\n\n###Bobo says: I can recognise your voices now :) Let's test this. Say Hey Bobo and I will guess who is talking to me :) \n\n", "yellow", attrs=["bold"]))
        leds.off()
    else:
        print(colored("\n\n###Bobo says: Something went wrong... I wasn't able to learn :(\n\n", "yellow", attrs=["bold"]))
    for i in range(0, 4):
        predicted_speaker = predict.predict(constants.PREDICTION_DATA_LOCATION, trained_model_filename, sorted(user_names))
        leds.wakeup()
        time.sleep(1)
        print(colored(f"\n\n ###Bobo says: Hi there {predicted_speaker}\n\n", "yellow", attrs=["bold"]))
        leds.off()
    


if __name__ == '__main__':
    run_pipeline()
