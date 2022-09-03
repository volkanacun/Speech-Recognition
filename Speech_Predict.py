#keyword spotting service
import tensorflow.keras as keras
import numpy as np
import librosa 



MODEL_PATH = "model.h5"
NUM_SAMPLES_TO_CONSIDER = 22050 # 1 second worth of sound

class _Keyword_Spotting_Service: #singleton class, a class that can only have 1 instance in a program 

    model = None
    _mappings = [
        "dataset\\bed",
        "dataset\\bird",
        "dataset\\cat",
        "dataset\\dog",
        "dataset\\down",
        "dataset\\eight",
        "dataset\\five",
        "dataset\\four",
        "dataset\\go",
        "dataset\\happy",
        "dataset\\house",
        "dataset\\left",
        "dataset\\marvin",
        "dataset\\nine",
        "dataset\\no",
        "dataset\\off",
        "dataset\\on",
        "dataset\\one",
        "dataset\\right",
        "dataset\\seven",
        "dataset\\sheila",
        "dataset\\six",
        "dataset\\stop",
        "dataset\\three",
        "dataset\\tree",
        "dataset\\two",
        "dataset\\up",
        "dataset\\wow",
        "dataset\\yes",
        "dataset\\zero",
        "dataset\\_background_noise_"
    ]
    _instance = None

    def predict(self, file_path):

        # extract the MFCCs
        MFCCs = self.preprocess(file_path) #( #segments, # coefficients)

        # convert 2d MFCCs array in to 4D array --> #samples, #segments, #coefficients, #channels
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make a prediction
        predictions = self.model.predict(MFCCs) #[ [] ]
        predicted_index = np.argmax(predictions)
        prediced_keyword = self._mappings[predicted_index]

        return prediced_keyword


    def preprocess(self, file_path, n_mfcc=13, hop_length=512, n_fft=2048):

        # load the audio files
        signal, sr = librosa.load(file_path)

        # ensure consistency in the audio file lenght
        if len(signal) > NUM_SAMPLES_TO_CONSIDER: 
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]


        # extract the MFCCs
        MFCCs = librosa.feature.mfcc(signal,n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        return MFCCs.T
        

def Keyword_Spotting_Service():

    # ensure we only have 1 instance of KSS
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance


if __name__ == "__main__":

    kss = Keyword_Spotting_Service()

    keywords1 = kss.predict("Test/down.wav")
    keywords2 = kss.predict("Test/left.wav")

    print(f"Predicted keywords: {keywords1}, {keywords2}")
