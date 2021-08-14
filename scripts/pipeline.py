import data_collection
import data_prep
import train

def run_pipeline():
    print("Step 1: Establish the number of users")
    num_users = data_collection.ask_about_the_number_of_users()
    print("Step 2: Get user names and record audio")
    user_names = data_collection.collect_user_name_and_audio(int(num_users))
    print("USERS: ", user_names)
    print("Step 3 data prep")
    print("Remove silence and split audio")
    remove_sil_and_split = data_prep.remove_silence_and_split_audio(user_names)
    if remove_sil_and_split != 0:
        exit("[ERROR] Something went wrong in splitting the audio")
    train_ds, valid_ds, test_ds = data_prep.create_datasets(data_prep.DATASET_AUDIO_PATH)
    print("Step 4: Training")
    training = train.train_model(train_ds, valid_ds, test_ds)
    if training == 0:
        print("I can recognise your voice now :)")
    else:
        print("Something went wrong...")


if __name__ == '__main__':
    run_pipeline()
