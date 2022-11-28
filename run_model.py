from dataclasses import dataclass
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

import torchaudio
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm
import numpy as np

import os
import sys
sys.path.append('.')

import FastSpeech2

import waveglow
import text
import audio
import utils


@dataclass
class MelSpectrogramConfig:
    num_mels = 80

@dataclass
class FastSpeechConfig:
    vocab_size = 300
    max_seq_len = 3000

    encoder_dim = 256
    encoder_n_layer = 4
    encoder_head = 2
    encoder_conv1d_filter_size = 1024
    
    pitch_min = 127
    pitch_max = 325
    energy_min = 4
    energy_max = 45
    
    n_bins = 256

    decoder_dim = 256
    decoder_n_layer = 4
    decoder_head = 2
    decoder_conv1d_filter_size = 1024

    fft_conv1d_kernel = (9, 1)
    fft_conv1d_padding = (4, 0)

    duration_predictor_filter_size = 256
    duration_predictor_kernel_size = 3
    dropout = 0.1
    
    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3

    PAD_WORD = '<blank>'
    UNK_WORD = '<unk>'
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'


@dataclass
class TrainConfig:
    checkpoint_path = "./weights"
    logger_path = "./logger"
    mel_ground_truth = "./mels"
    alignment_path = "./alignments"
    data_path = './data/train.txt'
    
    wandb_project = 'fastspeech_example'
    
    text_cleaners = ['english_cleaners']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cuda:0'

    batch_size = 16
    epochs = 2000
    n_warm_up_step = 4000

    learning_rate = 1e-3
    weight_decay = 1e-6
    grad_clip_thresh = 1.0
    decay_step = [500000, 1000000, 2000000]

    save_step = 3000
    log_step = 5
    clear_Time = 20

    batch_expand_size = 32
    

mel_config = MelSpectrogramConfig()
model_config = FastSpeechConfig()
train_config = TrainConfig()

WaveGlow = utils.get_WaveGlow()
WaveGlow = WaveGlow.cuda()

model = FastSpeech2.load_FastSpeech2('./weights/checkpoint_180916.pth.tar', model_config, train_config.device)
model = model.eval()
model = model.train(False)


def synthesis(model, text, alpha=1.0, pitch_alpha=1.0, energy_alpha=1.0):
    text = np.array(phn)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(train_config.device)
    src_pos = torch.from_numpy(src_pos).long().to(train_config.device)
    
    with torch.no_grad():
        mel = model.forward(sequence, src_pos, alpha=alpha, pitch_alpha=pitch, energy_alpha=energy)
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)


def get_data():
    tests = [ 
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space",
    ]
    data_list = list(text.text_to_sequence(test, train_config.text_cleaners) for test in tests)

    return data_list

data_list = get_data()
for pitch in [0.8, 1., 1.2]:
    for energy in [0.8, 1., 1.2]:
        for speed in [0.8, 1., 1.2]:
            for i, phn in tqdm(enumerate(data_list)):
                mel, mel_cuda = synthesis(model, phn, speed, pitch_alpha=pitch, energy_alpha=energy)
                
                os.makedirs("results", exist_ok=True)
                
                audio.tools.inv_mel_spec(
                    mel, f"results/s={speed}_p={pitch}_e={energy}_{i}.wav"
                )
                
                waveglow.inference.inference(
                    mel_cuda, WaveGlow,
                    f"results/s={speed}_p={pitch}_e={energy}_{i}_waveglow.wav"
                )