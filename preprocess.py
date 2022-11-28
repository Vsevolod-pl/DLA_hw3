import pyworld

from scipy.io import wavfile
import pandas as pd

with open('./data/LJSpeech-1.1/stats.csv', 'w') as statfile:
    statfile.write(f'filename, energy, pitch\n')
    for path, dirs, files in os.walk('./data/LJSpeech-1.1/wavs'):
        for file in files:
            if file.endswith('.wav'):
                samplerate, data = wavfile.read(path+'/'+file)
                energy = np.linalg.norm(data)
                data = data/np.max(np.abs(data))
                
                _pitch, t = pyworld.dio(data/32767, samplerate)
                pitch = pyworld.stonemask(data/32767, _pitch, t, samplerate)
                pitchmean = pitch[pitch.nonzero()].mean()
                statfile.write(f'{file}, {energy}, {pitchmean}\n')


stats = pd.read_csv('./data/LJSpeech-1.1/stats.csv')
text2file = pd.read_csv('./data/LJSpeech-1.1/metadata.csv', sep='|', header=None)
text = process_text(train_config.data_path)
ordered_energy_pitch = []
tl = text2file[1].apply(lambda s:s.lower())
i = 0
for t in text:
    i += 1
    mel_gt_name = os.path.join(
        train_config.mel_ground_truth, "ljspeech-mel-%05d.npy" % (i+1))
    mel_gt_target = np.load(mel_gt_name)
    t1 = t.lower()
    vs = list(text2file[tl == t1[:-1]][0])
    if vs:
        name = vs[0] + '.wav'
        cur = stats[stats['filename'] == name]
        #energy = list(cur[' energy'])[0]
        energy = torch.sqrt(torch.sum(torch.exp(mel_gt_target)**2))
        pitch = list(cur[' pitch'])[0]
        ordered_energy_pitch.append((energy, pitch))
    else:
        ordered_energy_pitch.append((None, None))
    break