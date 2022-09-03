import tensorflow.keras as keras
import numpy as np
import librosa 
from Speech_Predict import MODEL_PATH, NUM_SAMPLES_TO_CONSIDER

MODEL_PATH = "model.h5"
NUM_SAMPLES_TO_CONSIDER = 22050 


class _Keyword_Spotting_Service:
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
        MFCCs = self.preprocess(file_path) # number of segments  and coefficients determines the shape


        # convert 2d MFCCs array into 4D array -> (# samples, # segments, # coefficients, # channels)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make prediction
        predictions = self.model.predict(MFCCs) 
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword

    
    
    def preprocess(self, file_path, n_mfccs=13, n_fft=2048, hop_length=512):
        # load the audio file
        signal, sr=librosa.load(file_path)

        # ensure consistancy in the audio file lenght
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]


        # extract MFCCs
        MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfccs, n_fft=n_fft, hop_length=hop_length)
        
        return MFCCs.T
 


def Keyword_Spotting_Service():
    # ensure that we only have 1 instance of KSS
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance

if __name__ == "__main__":
    kss = Keyword_Spotting_Service()

    keyword1 = kss.predict("test/down.wav")
    keyword2 = kss.predict("test/0ab3b47d_nohash_0.wav")

    print(f"Predicted keywords: {keyword1}, {keyword2}")
