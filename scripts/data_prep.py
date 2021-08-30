from pydub.silence import detect_nonsilent
from pydub.utils import make_chunks
import os
import shutil
import numpy as np
import tensorflow as tf
from pathlib import Path
from pydub import AudioSegment
import constants
from termcolor import colored

# Remove silence at audio start and end

def remove_start_end_silence(path_in, path_out, format="wav"):
    sound = AudioSegment.from_file(path_in, format=format)
    non_sil_times = detect_nonsilent(sound, min_silence_len=50, silence_thresh=sound.dBFS * 1.5)
    if len(non_sil_times) > 0:
        non_sil_times_concat = [non_sil_times[0]]
        if len(non_sil_times) > 1:
            for t in non_sil_times[1:]:
                if t[0] - non_sil_times_concat[-1][-1] < 200:
                    non_sil_times_concat[-1][-1] = t[1]
                else:
                    non_sil_times_concat.append(t)
        non_sil_times = [t for t in non_sil_times_concat if t[1] - t[0] > 350]
        sound[non_sil_times[0][0]: non_sil_times[-1][1]].export(path_out, format='wav')
    return 0

# Split audio into 1 second wavs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATASET_ROOT = constants.DATASET_ROOT
RAW_AUDIO = constants.RAW_AUDIO

def split_wav_into_one_second_wavs(input_wav_path, output_location):
    if not os.path.isdir(output_location):
        os.mkdir(output_location)
    basename = os.path.basename(input_wav_path)
    myaudio = AudioSegment.from_file(input_wav_path, "wav")
    chunk_length_ms = 1000
    chunks = make_chunks(myaudio, chunk_length_ms)
    for i, chunk in enumerate(chunks):
        chunk_name = f"{output_location}/{os.path.splitext(basename)[0]}_{i}.wav"
        print(colored(f"exporting {chunk_name}", "cyan", attrs=["bold"]))
        chunk.export(chunk_name, format="wav")

def remove_silence_and_split_audio(user_names, input_location):
    for user in user_names:
        for f in os.listdir(f"{input_location}/{user}"):
            filename = os.fsdecode(f)
            clean_filename = filename.replace(".wav", "_clean.wav")
            remove_start_end_silence(f"{input_location}/{user}/{filename}", f"{input_location}/{user}/{clean_filename}")
            print(colored(f"Processing: {input_location}/{user}/{clean_filename}", "cyan", attrs=["bold"]))
            split_wav_into_one_second_wavs(f"{input_location}/{user}/{clean_filename}", f"{constants.DATASET_ROOT}/audio/{user}")
    print(colored("[SUCCESS] Data successfully split in to 1s wavs.", "cyan", attrs=["bold"]))
    return 0


AUDIO_SUBFOLDER = constants.AUDIO_SUBFOLDER
NOISE_SUBFOLDER = constants.NOISE_SUBFOLDER

DATASET_AUDIO_PATH = constants.DATASET_AUDIO_PATH
DATASET_NOISE_PATH = constants.DATASET_NOISE_PATH

VALID_SPLIT = constants.VALID_SPLIT
TEST_SPLIT = constants.TEST_SPLIT
SHUFFLE_SEED = constants.SEED
SAMPLING_RATE = constants.SAMPLING_RATE
SCALE = constants.SCALE

BATCH_SIZE = constants.BATCH_SIZE

def process_noise():
    if os.path.exists(DATASET_AUDIO_PATH) is False:
        os.makedirs(DATASET_AUDIO_PATH)

    if os.path.exists(DATASET_NOISE_PATH) is False:
        os.makedirs(DATASET_NOISE_PATH)

    for folder in os.listdir(constants.DATASET_ROOT):
        print (colored(f"Processing: {folder}", "cyan" , attrs=["bold"]))

    noise_paths = []
    for subdir in os.listdir(DATASET_NOISE_PATH):
        subdir_path = Path(DATASET_NOISE_PATH) / subdir
        if os.path.isdir(subdir_path):
            noise_paths += [
                    os.path.join(subdir_path, filepath)
                    for filepath in os.listdir(subdir_path)
                    if filepath.endswith(".wav")
            ]

    print(
        colored("Found {} files belonging to {} directories".format(
            len(noise_paths), len(os.listdir(DATASET_NOISE_PATH))), "cyan", attrs=["bold"])
        )

    command = (
        "for dir in `ls -1 " + DATASET_NOISE_PATH + "`; do "
        "for file in `ls -1 " + DATASET_NOISE_PATH + "/$dir/*.wav`; do "
        "sample_rate=`ffprobe -hide_banner -loglevel panic -show_streams "
        "$file | grep sample_rate | cut -f2 -d=`; "
        "if [ $sample_rate -ne 16000 ]; then "
        "ffmpeg -hide_banner -loglevel panic -y "
        "-i $file -ar 16000 temp.wav; "
        "mv temp.wav $file; "
        "fi; done; done"
    )

    os.system(command)

    def load_noise_sample(path):
        sample, sampling_rate = tf.audio.decode_wav(
            tf.io.read_file(path), desired_channels=1
        )
        if sampling_rate == SAMPLING_RATE:
            # Number of slices of 16000 each that can be generated from the noise sample
            slices = int(sample.shape[0] / SAMPLING_RATE)
            sample = tf.split(sample[: slices * SAMPLING_RATE], slices)
            return sample
        else:
            print(colored("Sampling rate for {} is incorrect. Ignoring it".format(path)), "cyan", attrs=["bold"])
            return None

    noises = []
    for path in noise_paths:
        sample = load_noise_sample(path)
        if sample:
            noises.extend(sample)
    noises = tf.stack(noises)

    print(
        colored("{} noise files were split into {} noise samples where each is {} sec. long".format(
            len(noise_paths), noises.shape[0], noises.shape[1] // SAMPLING_RATE
        ), "cyan", attrs=["bold"])
    )
    return noises

def paths_and_labels_to_dataset(audio_paths, labels):
    """Constructs a dataset of audios and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(lambda x: path_to_audio(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


def path_to_audio(path):
    """Reads and decodes an audio file."""
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    return audio


def add_noise(audio, noises=None, scale=0.5):
    print(colored(f"Add noise start shape audio: {audio}", "cyan", attrs=["bold"]))
    if noises is not None:
        # Create a random tensor of the same size as audio ranging from
        # 0 to the number of noise stream samples that we have.
        tf_rnd = tf.random.uniform(
            (tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
        )
        noise = tf.gather(noises, tf_rnd, axis=0)

       # Get the amplitude proportion between the audio and the noise
        prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
        prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)

        # Adding the rescaled noise to audio
        audio = audio + noise * prop * scale
    print(colored(f"Add noise end shape audio: {audio}", "cyan", attrs=["bold"]))
    return audio


def audio_to_fft(audio):
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])

def create_datasets (original_dataset_path, VALID_SPLIT, TEST_SPLIT):
    print(colored("Creating datasets . . .", "cyan", attrs=["bold"]))
    class_names = os.listdir(DATASET_AUDIO_PATH)
    print(colored(f"Class names {class_names}", "cyan", attrs=["bold"]))
    audio_paths = []
    test_paths = []
    labels = []
    test_labels = []
    noises = process_noise()

    for label, name in enumerate(class_names):
        print(colored(f"Processing speaker {name}", "cyan", attrs=["bold"]))
        dir_path = Path(DATASET_AUDIO_PATH) / name
        speaker_sample_paths = [
                          os.path.join(dir_path, filepath)
                          for filepath in os.listdir(dir_path)
                          if filepath.endswith("wav")
        ]
        audio_paths += speaker_sample_paths
        labels += [label] * len(speaker_sample_paths)
    print(colored(f"Found {len(audio_paths)} files belonging to {len(set(labels))} classes", "cyan", attrs=["bold"]))

    # Shuffle
    rng = np.random.RandomState(SHUFFLE_SEED)
    rng.shuffle(audio_paths)
    rng = np.random.RandomState(SHUFFLE_SEED)
    rng.shuffle(labels)

    num_val_samples = int(VALID_SPLIT * len(audio_paths))
    num_test_samples = int(TEST_SPLIT * len(audio_paths))
    print(colored(f"Using {num_val_samples} for validation and {num_test_samples} for testing", "cyan", attrs=["bold"]))

    train_audio_paths = audio_paths[:-(num_val_samples)]
    train_labels = labels[:-(num_val_samples)]

    valid_audio_paths = audio_paths[-(num_val_samples + num_test_samples): -num_test_samples]
    valid_labels = labels[-(num_val_samples + num_test_samples): -num_test_samples]

    test_audio_paths = audio_paths[-num_test_samples:]
    test_labels = labels[-num_test_samples:]

    # Create 3 datasets, one for training, another for validation and another for test
    train_ds = paths_and_labels_to_dataset(train_audio_paths, train_labels)
    train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(
        BATCH_SIZE
    )

    valid_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
    valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(BATCH_SIZE)

    test_ds = paths_and_labels_to_dataset(test_audio_paths, test_labels)
    test_ds = test_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(BATCH_SIZE)


    # Add noise to the training set
    train_ds = train_ds.map(
        lambda x, y: (add_noise(x, noises, scale=SCALE), y),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # Transform audio wave to the frequency domain using `audio_to_fft`
    train_ds = train_ds.map(
        lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    valid_ds = valid_ds.map(
        lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)

    test_ds = test_ds.map(
        lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    print(colored("DONE", "cyan", attrs=["bold"]))
    return(train_ds, valid_ds, test_ds)

