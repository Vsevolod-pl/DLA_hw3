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


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        # q, k, v: [ (batch_size * n_heads) x seq_len x hidden_size ]
        
        ### Your code here
        
        # attn: [ (batch_size * n_heads) x seq_len x seq_len ]
        attn = torch.bmm(q, k.transpose(-1, -2))
        attn /= self.temperature
        
        if mask is not None:
            attn = torch.masked_fill(attn, mask, -torch.inf)

        ### Your code here
        
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        # output: [ (batch_size * n_heads) x seq_len x hidden_size ]
        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(
            temperature=d_k**0.5) 
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()

    def reset_parameters(self):
         # normal distribution initialization better than kaiming(default in pytorch)
        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_v))) 
        
    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv
        
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in, d_hid, kernel_size=model_config.fft_conv1d_kernel[0], padding=model_config.fft_conv1d_padding[0])
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=model_config.fft_conv1d_kernel[1], padding=model_config.fft_conv1d_padding[1])

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output


class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


def create_alignment(base_mat, duration_predictor_output):
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + duration_predictor_output[i][j]
    return base_mat


class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)


class DurationPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, model_config: FastSpeechConfig):
        super(DurationPredictor, self).__init__()

        self.input_size = model_config.encoder_dim
        self.filter_size = model_config.duration_predictor_filter_size
        self.kernel = model_config.duration_predictor_kernel_size
        self.conv_output_size = model_config.duration_predictor_filter_size
        self.dropout = model_config.dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)
            
        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out


class VariancePredictor(nn.Module):
    """ Variance Predictor """

    def __init__(self, model_config: FastSpeechConfig):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config.encoder_dim
        self.filter_size = model_config.duration_predictor_filter_size
        self.kernel = model_config.duration_predictor_kernel_size
        self.conv_output_size = model_config.duration_predictor_filter_size
        self.dropout = model_config.dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)
            
        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, model_config):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(-1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)
        if target is not None:
            output = self.LR(x, target, mel_max_length)
            return output, duration_predictor_output
        else:
            duration_predictor_output = (duration_predictor_output*alpha + 0.5).int()
            output = self.LR(x, duration_predictor_output)
            mel_pos = torch.stack([torch.Tensor([i+1 for i in range(output.size(1))])]).long().to(train_config.device)
        
            return output, mel_pos



class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """
    def __init__(self, model_config: FastSpeechConfig):
        super(VarianceAdaptor, self).__init__()
        self.length_regulator = LengthRegulator(model_config)
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)
        
        n_bins = model_config.n_bins
        
        self.energy_bins = nn.Parameter(
                torch.linspace(model_config.energy_min, model_config.energy_max, n_bins - 1),
                requires_grad=False,)
        self.pitch_bins = nn.Parameter(
                torch.linspace(model_config.pitch_min, model_config.pitch_max, n_bins - 1),
                requires_grad=False,)
        
        self.energy_embedding = nn.Embedding(n_bins, model_config.encoder_dim)
        self.pitch_embedding = nn.Embedding(n_bins, model_config.encoder_dim)
        
    def get_embedding(self, pred, emb, bins, x, target, control):
        prediction = pred(x)
        if target is not None:
            embedding = emb(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = emb(torch.bucketize(prediction, self.energy_bins))
        return prediction, embedding
    
    def get_energy_embedding(self, x, target, control):
        return self.get_embedding(self.energy_predictor, self.energy_embedding, self.energy_bins, x, target, control)
    
    def get_pitch_embedding(self, x, target, control):
        return self.get_embedding(self.pitch_predictor, self.pitch_embedding, self.pitch_bins, x, target, control)
    
    def forward(self, x, mel_max_length=None, length_target=None, alpha=1.0, pitch_target=None, energy_target=None, pitch_alpha=1.0, energy_alpha=1.0):
        output, duration_predictor_output = self.length_regulator(x, alpha, length_target, mel_max_length)
        
        pitch_prediction, pitch_embedding = self.get_pitch_embedding(output, pitch_target, pitch_alpha)
        output2 = output + pitch_embedding
        energy_prediction, energy_embedding = self.get_energy_embedding(output, energy_target, energy_alpha)
        output3 = output2 + energy_embedding
        print(pitch_alpha, energy_alpha)
        
        return output3, duration_predictor_output, pitch_prediction, energy_prediction



def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(model_config.PAD).type(torch.float).unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(model_config.PAD)
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask



class Encoder(nn.Module):
    def __init__(self, model_config):
        super(Encoder, self).__init__()
        
        len_max_seq=model_config.max_seq_len
        n_position = len_max_seq + 1
        n_layers = model_config.encoder_n_layer

        self.src_word_emb = nn.Embedding(
            model_config.vocab_size,
            model_config.encoder_dim,
            padding_idx=model_config.PAD
        )

        self.position_enc = nn.Embedding(
            n_position,
            model_config.encoder_dim,
            padding_idx=model_config.PAD
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            model_config.encoder_dim,
            model_config.encoder_conv1d_filter_size,
            model_config.encoder_head,
            model_config.encoder_dim // model_config.encoder_head,
            model_config.encoder_dim // model_config.encoder_head,
            dropout=model_config.dropout
        ) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)
        
        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        

        return enc_output, non_pad_mask



class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, model_config):

        super(Decoder, self).__init__()

        len_max_seq=model_config.max_seq_len
        n_position = len_max_seq + 1
        n_layers = model_config.decoder_n_layer

        self.position_enc = nn.Embedding(
            n_position,
            model_config.encoder_dim,
            padding_idx=model_config.PAD,
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            model_config.encoder_dim,
            model_config.encoder_conv1d_filter_size,
            model_config.encoder_head,
            model_config.encoder_dim // model_config.encoder_head,
            model_config.encoder_dim // model_config.encoder_head,
            dropout=model_config.dropout
        ) for _ in range(n_layers)])

    def forward(self, enc_seq, enc_pos, return_attns=False):

        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos)
        non_pad_mask = get_non_pad_mask(enc_pos)

        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output
    


def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, 1, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask



class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self, model_config):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(model_config)
        self.decoder = Decoder(model_config)

        self.mel_linear = nn.Linear(model_config.decoder_dim, mel_config.num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None, alpha=1.0, pitch_target=None, energy_target=None, pitch_alpha=1.0, energy_alpha=1.0):
        x, non_pad_mask = self.encoder(src_seq, src_pos)
        
        if self.training:
            output, duration_predictor_output, pitch_prediction, energy_prediction = self.variance_adaptor(x, length_target=length_target, mel_max_length=mel_max_length, alpha=alpha, pitch_target=pitch_target, energy_target=pitch_target)
            output = self.decoder(output, mel_pos)
            
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            output = self.mel_linear(output)
            return output, duration_predictor_output, pitch_prediction, energy_prediction
        else:
            output, mel_pos, pitch_prediction, energy_prediction = self.variance_adaptor(x, alpha=alpha, pitch_alpha=pitch_alpha, energy_alpha=energy_alpha)
            output = self.decoder(output, mel_pos)
            output = self.mel_linear(output)
            return output



def load_FastSpeech2(ckpt_path=None, model_config=model_config, device='cuda:0'):
    model = FastSpeech(model_config)
    model = model.to(device)
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path, map_location=device)['model'])

    return model
