import requests

URL = "http://127.0.0.1:5000/predict"

TEST_AUDIO_FILE_PATH = "test/0ab3b47d_nohash_0.wav"

if __name__ == "__main___":
    audio_file = open(TEST_AUDIO_FILE_PATH, "rb")
    values = {"file": (TEST_AUDIO_FILE_PATH, audio_file, "audio/wav")}
    reponse = requests.post(URL, files=values)
    data = reponse.json()

    print(f"Predicted keyword is:{data['keyword']}")
    