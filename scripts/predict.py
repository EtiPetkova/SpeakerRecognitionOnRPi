import record
import constants
import data_prep
from tensorflow import keras
import os
import shutil
from pathlib import Path

PREDICTION_DATA_LOCATION = "/home/pi/SpeakerRecognitionOnRPi/data/audio_for_predictions/"
MODEL = "speaker_recognizer_2_speakers_20210814-225044.h5"

def predict(PREDICTION_DATA_LOCATION, trained_model, classes):
   model = keras.models.load_model(trained_model)
   shutil.rmtree(f"{PREDICTION_DATA_LOCATION}/speaker")
   os.mkdir(f"{PREDICTION_DATA_LOCATION}/speaker")
   shutil.rmtree(constants.SPLIT_AUDIO_LOCATION)
   os.mkdir(constants.SPLIT_AUDIO_LOCATION)
   full_recorded_filename = f"{PREDICTION_DATA_LOCATION}/speaker/recording.wav"
   recording_result = record.record_audio(full_recorded_filename, constants.RECORD_SECONDS_PREDICTION)
   if recording_result != 0:
       exit("[ERROR] Couldn't record audio")
   remove_sil_and_split = data_prep.remove_silence_and_split_audio(["speaker"], PREDICTION_DATA_LOCATION)
   test_labels = ["speaker"]
   dir_path = Path("/home/pi/SpeakerRecognitionOnRPi/data/audio/speaker/")
   test_paths = [os.path.join(dir_path, filepath) for filepath in os.listdir(dir_path) if filepath.endswith("wav")]
   test_ds = data_prep.paths_and_labels_to_dataset(test_paths, test_labels)
   test_ds = test_ds.shuffle(buffer_size=constants.BATCH_SIZE*8, seed=constants.SEED).batch(constants.BATCH_SIZE)
   test_ds = test_ds.map(lambda x,y: (data_prep.add_noise(x, scale=constants.SCALE), y))
   predictions = []
   for audios, labels in test_ds.take(1):
       ffts = data_prep.audio_to_fft(audios)
       predict_prob = model.predict(ffts)
       predict_classes = predict_prob.argmax(axis=-1)
       for p in predict_classes:
           predictions.append(classes[p])
   return most_common(predictions)

def most_common(li):
   return max(set(li), key=li.count)

