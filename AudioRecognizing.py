from pydub import AudioSegment
from pydub.utils import make_chunks
import os
import soundfile as sf
import librosa
import numpy as np
from numpy.fft import fft
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
        
def stereo_to_mono(import_path):
    sound = AudioSegment.from_wav(import_path)
    sound = sound.set_channels(1)
    sound.export(import_path, format="wav")

def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print("Error")

def CleanAudio(Input, Output):
    stereo_to_mono(Input)

    (audio, rate) = librosa.load(Input, sr=None)

    mean = sum(abs(audio)) / len(audio)
    NewAudio = [x for x in audio if abs(x) >= mean/3]

    sf.write(Output, NewAudio, rate)

def Split(Input, OutputFolder, length, label):
    Audio = AudioSegment.from_file(Input, "wav")
    chunk_length = length
    chunks = make_chunks(Audio, chunk_length)

    count = 0

    for i, chunk in enumerate(chunks):
        chunk_name = OutputFolder + "/{0}{1}.wav".format(label, i)
        chunk.export(chunk_name, format="wav")
        count += 1

    return count - 1

def FFT(InputFolder, segmentsCount, Output, label):
    data = []
    for i in range(segmentsCount):

        chunkData = []

        chunk_name = InputFolder + "/{0}{1}.wav".format(label, i)
        (NewAudio, r) = librosa.load(chunk_name)
        fft_result = np.fft.fft(NewAudio)
        fft_magnitudes = abs(fft_result)

        for j in range(len(fft_magnitudes)):
            chunkData.append(fft_magnitudes[j])

        data.append(chunkData)

    df = pd.DataFrame(data)
    df.to_csv(Output, index=False)


def CatBoostModel(P1, P2, P3, modelIteration, SaveModelPath, modelVerbose = 10):
    data1 = pd.read_csv(P1).values
    data2 = pd.read_csv(P2).values
    data3 = pd.read_csv(P3).values
    x = []
    y = []
    length = min(len(data1), len(data2), len(data3))
    for i in range(length):
        x.append(data1[i])
        y.append(0)
    for i in range(length):
        x.append(data2[i])
        y.append(1)
    for i in range(length):
        x.append(data3[i])
        y.append(2)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = CatBoostClassifier(iterations=modelIteration,  # Number of boosting iterations
                               depth=6,  # Depth of the tree
                               learning_rate=0.1,
                               loss_function='MultiClass',  # For multiclass classification
                               verbose=modelVerbose)
    model.fit(x_train, y_train)
    predictionss = model.predict(x_test)
    score = accuracy_score(y_test, predictionss)
    score *= 100

    pickle.dump(model, open(SaveModelPath, 'wb'))
    return score

def XGBoostModel(P1, P2, P3, SaveModelPath):
    data1 = pd.read_csv(P1).values
    data2 = pd.read_csv(P2).values
    data3 = pd.read_csv(P3).values
    x = []
    y = []
    length = min(len(data1), len(data2), len(data3))
    for i in range(length):
        x.append(data1[i])
        y.append(0)
    for i in range(length):
        x.append(data2[i])
        y.append(1)
    for i in range(length):
        x.append(data3[i])
        y.append(2)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = XGBClassifier()
    model.fit(x_train, y_train)
    predictionss = model.predict(x_test)
    score = accuracy_score(y_test, predictionss)
    score *= 100

    pickle.dump(model, open(SaveModelPath, 'wb'))
    return score

def RandomForestModel(P1, P2, P3, SaveModelPath):
    data1 = pd.read_csv(P1).values
    data2 = pd.read_csv(P2).values
    data3 = pd.read_csv(P3).values
    x = []
    y = []
    length = min(len(data1), len(data2), len(data3))
    for i in range(length):
        x.append(data1[i])
        y.append(0)
    for i in range(length):
        x.append(data2[i])
        y.append(1)
    for i in range(length):
        x.append(data3[i])
        y.append(2)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    predictionss = model.predict(x_test)
    score = accuracy_score(y_test, predictionss)
    score *= 100

    pickle.dump(model, open(SaveModelPath, 'wb'))
    return score

def DesicionTreeModel(P1, P2, P3, SaveModelPath):
    data1 = pd.read_csv(P1).values
    data2 = pd.read_csv(P2).values
    data3 = pd.read_csv(P3).values
    x = []
    y = []
    length = min(len(data1), len(data2), len(data3))
    for i in range(length):
        x.append(data1[i])
        y.append(0)
    for i in range(length):
        x.append(data2[i])
        y.append(1)
    for i in range(length):
        x.append(data3[i])
        y.append(2)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    predictionss = model.predict(x_test)
    score = accuracy_score(y_test, predictionss)
    score *= 100

    pickle.dump(model, open(SaveModelPath, 'wb'))
    return score
