import os
import yaml
import librosa
import soundfile as sf
import numpy as np
import pickle

class wav_processor:
    def __init__(self,config):
        self.n_fft = config['transform']['n_fft']
        self.hop_length = config['transform']['hop_length']
        self.win_length = config['transform']['win_length']
        self.window = config['transform']['window']
        self.center = config['transform']['center']
        self.sr = config['transform']['sr']
        path_normalize = config['train']['path']+'/Separability_Checker/dict_normalize.ark'
        if os.path.exists(path_normalize):
            self.normalize = pickle.load(open(path_normalize, 'rb'))

    def stft(self,y):
        Y = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length,
                            win_length=self.win_length, window=self.window,
                            center=self.center)
        return Y.T

    def log_power(self,y,normalize=True):
        Y = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length,
                            win_length=self.win_length, window=self.window,
                            center=self.center)
        eps = np.finfo(float).eps
        log_power =  np.log(np.maximum(np.abs(Y),eps)).T

        if normalize:
            log_power = (log_power - self.normalize['mean']) / self.normalize['std']


        return log_power

    def istft(self, Y):
        Y = Y.T
        y = librosa.istft(Y, hop_length=self.hop_length,win_length=self.win_length,
                            window=self.window,center=self.center)
        return y


    def apply_normalize(self,Y):
        return (Y - self.normalize['mean']) / self.normalize['std']

    def read_wav(self,wav_path):
        y,_ = sf.read(wav_path)
        return y

    def read_scp(self,scp_path):
        with open(scp_path, 'r') as f:
            lines = f.readlines()
        scp_wav = {}
        for line in lines:
            line = line.split()
            if line[0] in scp_wav.keys():
                print(line[0])
                raise ValueError
            scp_wav[line[0]] = line[1]
        return scp_wav
