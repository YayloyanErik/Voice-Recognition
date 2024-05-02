import pickle
from pydub import AudioSegment
from pydub.utils import make_chunks
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import accuracy_score

def stereo_to_mono(import_path):
        sound = AudioSegment.from_wav(import_path)
        sound = sound.set_channels(1)
        sound.export(import_path, format="wav")
        
def Clean(Name):
    stereo_to_mono(Name)

    (data, rate) = librosa.load(Name, sr=None)

    a = sum(abs(data)) / len(data) # mean
    NewAudio = [x for x in data if abs(x) >= a/3] # Clear noise

    sf.write("Audio/AudioClean.wav", NewAudio, rate)

def Split(chunk_length):
    Audio = AudioSegment.from_file("Audio/AudioClean.wav", "wav")
    chunks = make_chunks(Audio, chunk_length)
    for index in range(1):
        chunk_name = "Audio/Audio{0}.wav".format(index)
        chunks[index].export(chunk_name, format="wav")


def FFT():
    data = []

    for i in range(1):
        chunkData = []
        chunk_name = "Audio/Audio{0}.wav".format(i)
        (NewAudio, r) = librosa.load(chunk_name)
        # FFT
        fft_result = np.fft.fft(NewAudio)
        fft_magnitudes = abs(fft_result)

        # Data
        for j in range(len(fft_magnitudes)):
            chunkData.append(fft_magnitudes[j])

        data.append(chunkData)
    df = pd.DataFrame(data)
    df.to_csv("Data/TestingData.csv", index=False)

def Predict(Name, model):
    data = pd.read_csv("Data/TestingData.csv")
    d0 = data.values
    model = pickle.load(open(model, 'rb'))
    pred = model.predict(d0)
    P1 = 0
    P2 = 0
    P3 = 0

    for i in pred:
        if i == 0:
            P1 += 1
        if i == 1:
            P2 += 1
        if i == 2:
            P3 += 1

    l = len(pred)
    P1 = (P1 / l) * 100
    P2 = (P2 / l) * 100
    P3 = (P3 / l) * 100

    all = [P1, P2, P3]

    print(all)

    if max(all) < 45:
        return -1
    elif P1 == max(all):
        return 0
    elif P2 == max(all):
        return 1
    elif P3 == max(all):
        return 2
