import os
DATASET_ROOT = "/home/pi/SpeakerRecognitionOnRPi/data/"
RAW_AUDIO = "/home/pi/SpeakerRecognitionOnRPi/data/collected_audio/"
FEATURE_EXTRACTION_MODEL = "/home/pi/SpeakerRecognitionOnRPi/models/feature_extraction_models/model_10_speakers_dropout.h5"
AUDIO_SUBFOLDER = "audio"
NOISE_SUBFOLDER = "noise"
DATASET_AUDIO_PATH = os.path.join(DATASET_ROOT, AUDIO_SUBFOLDER)
DATASET_NOISE_PATH = os.path.join(DATASET_ROOT, NOISE_SUBFOLDER)
SEED = 43
VALID_SPLIT = 0.1
TEST_SPLIT = 0.1
SAMPLING_RATE = 16000
SCALE = 0.5
BATCH_SIZE = 64
EPOCHS = 10
