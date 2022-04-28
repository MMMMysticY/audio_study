import torch
import torchaudio
print(torch.__version__)
print(torchaudio.__version__)
data_root_path = '/home/wy/audio_study/data/study_data/'
SPEECH_FILE = data_root_path + "speech.wav"
waveform, sample_rate = torchaudio.load(SPEECH_FILE)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
waveform = waveform.to(device)
print(waveform)
# import sys
# print(sys.path)