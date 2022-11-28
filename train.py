from dataclasses import dataclass
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm
import numpy as np

import time
import os
import sys
sys.path.append('.')

import FastSpeech2

import waveglow
import text
import audio
import utils

from torch.optim.lr_scheduler  import OneCycleLR
from wandb_writer import WanDBWriter
from text import text_to_sequence


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



def pad_1D(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_1D_tensor(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad_2D_tensor(inputs, maxlen=None):

    def pad(x, max_len):
        if x.size(0) > max_len:
            raise ValueError("not max_len")

        s = x.size(1)
        x_padded = F.pad(x, (0, 0, 0, max_len-x.size(0)))
        return x_padded[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.size(0) for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt

def load_pitch_energy(train_pitch_energy):
    with open(train_pitch_energy, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            if line != 'energy,pitch\n':
                txt.append(list(map(eval, line.split(','))))

        return txt

def get_data_to_buffer(train_config):
    buffer = list()
    text = process_text(train_config.data_path)
    energy_pitch = load_pitch_energy('./data/orderedstats.csv')

    start = time.perf_counter()
    for i in tqdm(range(len(text))):

        mel_gt_name = os.path.join(
            train_config.mel_ground_truth, "ljspeech-mel-%05d.npy" % (i+1))
        mel_gt_target = np.load(mel_gt_name)
        duration = np.load(os.path.join(
            train_config.alignment_path, str(i)+".npy"))
        character = text[i][0:len(text[i])-1]
        character = np.array(
            text_to_sequence(character, train_config.text_cleaners))

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        mel_gt_target = torch.from_numpy(mel_gt_target)
        
        energy, pitch = energy_pitch[i]
        energy = torch.sqrt(torch.sum(torch.exp(mel_gt_target)**2))
        energy_pitch[i][0] = energy
        
        if pitch is not None:
            buffer.append({"text": character, "duration": duration,
                           "mel_target": mel_gt_target,
                           "energy": energy,
                           "pitch": pitch})

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))

    return buffer


class BufferDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]


def reprocess_tensor(batch, cut_list):    
    texts = []
    mel_targets = []
    durations = []
    pitches = []
    energies = []
    for ind in cut_list:
        texts.append(batch[ind]["text"])
        mel_targets.append(batch[ind]["mel_target"])
        durations.append(batch[ind]["duration"])
        pitches.append(batch[ind]["pitch"])
        energies.append(batch[ind]["energy"])

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.size(0))

    src_pos = list()
    max_len = int(max(length_text))
    for length_src_row in length_text:
        src_pos.append(np.pad([i+1 for i in range(int(length_src_row))],
                              (0, max_len-int(length_src_row)), 'constant'))
    src_pos = torch.from_numpy(np.array(src_pos))

    length_mel = np.array(list())
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.size(0))

    mel_pos = list()
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(np.pad([i+1 for i in range(int(length_mel_row))],
                              (0, max_mel_len-int(length_mel_row)), 'constant'))
    mel_pos = torch.from_numpy(np.array(mel_pos))
    
    
    texts = pad_1D_tensor(texts)
    durations = pad_1D_tensor(durations)
    mel_targets = pad_2D_tensor(mel_targets)
    pitches = torch.tensor(pitches)
    energies = torch.tensor(energies)

    out = {"text": texts,
           "mel_target": mel_targets,
           "duration": durations,
           "mel_pos": mel_pos,
           "src_pos": src_pos,
           "mel_max_len": max_mel_len,
           "pitch": pitches,
           "energy": energies,}

    return out


def collate_fn_tensor(batch):
    len_arr = np.array([d["text"].size(0) for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    real_batchsize = batchsize // train_config.batch_expand_size

    cut_list = list()
    for i in range(train_config.batch_expand_size):
        cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

    output = list()
    for i in range(train_config.batch_expand_size):
        output.append(reprocess_tensor(batch, cut_list[i]))

    return output


buffer = get_data_to_buffer(train_config)

dataset = BufferDataset(buffer)

training_loader = DataLoader(
    dataset,
    batch_size=train_config.batch_expand_size * train_config.batch_size,
    shuffle=True,
    collate_fn=collate_fn_tensor,
    drop_last=True,
    num_workers=0
)




class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, duration_predicted, pitch, energy, mel_target, duration_predictor_target, pitch_target, energy_target):
        mel_loss = self.mse_loss(mel, mel_target)

        duration_predictor_loss = self.l1_loss(duration_predicted,
                                               duration_predictor_target.float())
        
        pitch_loss = self.l1_loss(torch.mean(pitch, -1), pitch_target)
        energy_loss = self.l1_loss(torch.mean(energy, -1), energy_target)

        return mel_loss, duration_predictor_loss, pitch_loss, energy_loss

model = FastSpeech2.load_FastSpeech2(None, model_config, train_config.device)

fastspeech_loss = FastSpeechLoss()
current_step = 0

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_config.learning_rate,
    betas=(0.9, 0.98),
    eps=1e-9)

scheduler = OneCycleLR(optimizer, **{
    "steps_per_epoch": len(training_loader) * train_config.batch_expand_size,
    "epochs": train_config.epochs,
    "anneal_strategy": "cos",
    "max_lr": train_config.learning_rate,
    "pct_start": 0.1
})

logger = WanDBWriter(train_config)

tqdm_bar = tqdm(total=train_config.epochs * len(training_loader) * train_config.batch_expand_size - current_step)


for epoch in range(train_config.epochs):
    for i, batchs in enumerate(training_loader):
        # real batch start here
        for j, db in enumerate(batchs):
            current_step += 1
            tqdm_bar.update(1)
            
            logger.set_step(current_step)

            # Get Data
            character = db["text"].long().to(train_config.device)
            mel_target = db["mel_target"].float().to(train_config.device)
            duration = db["duration"].int().to(train_config.device)
            mel_pos = db["mel_pos"].long().to(train_config.device)
            src_pos = db["src_pos"].long().to(train_config.device)
            max_mel_len = db["mel_max_len"]
            pitch_target = db['pitch'].to(train_config.device)
            energy_target = db['energy'].to(train_config.device)
            pitch_target_expanded = torch.broadcast_to(torch.unsqueeze(pitch_target, 1), mel_pos.shape)
            energy_target_expanded = torch.broadcast_to(torch.unsqueeze(energy_target, 1), mel_pos.shape)

            # Forward
            mel_output, duration_predictor_output, pitch_prediction, energy_prediction = model(character,
                                                              src_pos,
                                                              mel_pos=mel_pos,
                                                              mel_max_length=max_mel_len,
                                                              length_target=duration,
                                                              pitch_target=pitch_target_expanded,
                                                              energy_target=energy_target_expanded,
                                                              )

            # Calc Loss
            mel_loss, duration_loss, pitch_loss, energy_loss = fastspeech_loss(mel_output,
                                                                       duration_predictor_output,
                                                                       pitch_prediction,
                                                                       energy_prediction,
                                                                       mel_target,
                                                                       duration,
                                                                       pitch_target,
                                                                       energy_target)
            total_loss = mel_loss + duration_loss + pitch_loss + energy_loss

            # Logger
            t_l = total_loss.detach().cpu().numpy()
            m_l = mel_loss.detach().cpu().numpy()
            d_l = duration_loss.detach().cpu().numpy()

            logger.add_scalar("duration_loss", d_l)
            logger.add_scalar("mel_loss", m_l)
            logger.add_scalar("total_loss", t_l)

            # Backward
            total_loss.backward()

            # Clipping gradients to avoid gradient explosion
            nn.utils.clip_grad_norm_(
                model.parameters(), train_config.grad_clip_thresh)
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if current_step % train_config.save_step == 0:
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                )}, os.path.join(train_config.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                print("save model at step %d ..." % current_step)
