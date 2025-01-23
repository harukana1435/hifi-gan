from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav, load_audio
from models import Generator
import csv
import librosa
import soundfile as sf

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

def get_audio_filelist(file):
    # トレーニングデータのファイルを読み込む
    with open(file, 'r', encoding='utf-8') as fi:
        reader = csv.reader(fi)
        next(reader)  # 1行目（カラム名）をスキップ
        training_files = [row[0]  # Audio Pathの部分（1列目）
                          for row in reader if len(row) > 0]
    return training_files

def _load_audio_file(path):
    audio_raw, rate = librosa.load(path, sr=h.sampling_rate, mono=False)
    return audio_raw, rate

def inference(a):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = get_audio_filelist(a.input_wavs_csv)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            basename = os.path.splitext(os.path.split(filname)[1])[0]
            output_file_dir = os.path.join(a.output_dir, 'result_'+basename)
            os.makedirs(output_file_dir, exist_ok=True)
            binaural_gt_outputfile = os.path.join(output_file_dir,'input_binaural.wav')
            mix_outputfile = os.path.join(output_file_dir,'mixed_mono.wav')
            binaural_pred_outputfile = os.path.join(output_file_dir,'predicted_binaural.wav')
            
            
            binaural_audio, sr = _load_audio_file(filname) #本物のバイノーラル
                   
            
            left_audio, right_audio = binaural_audio[0], binaural_audio[1]
            mix_audio = (left_audio + right_audio)/2 #ミックスしたもの
            
            
            mel_filepath = os.path.join(a.mel_dir, basename+'.npy')
            mel = torch.load(mel_filepath).to(device)
            mel = spectral_normalize_torch(mel)
            
            diff_audio = generator(mel)
            diff_audio = diff_audio.squeeze().cpu().numpy()
            
            left_audio_pred = (mix_audio + diff_audio)/2
            right_audio_pred = (mix_audio - diff_audio)/2
            
            binaural_pred = [left_audio_pred, right_audio_pred] #生成したバイノーラル音声
            
            sf.write(binaural_gt_outputfile, binaural_audio.transpose(),h.sampling_rate)     
            sf.write(mix_outputfile, mix_audio, h.sampling_rate)
            sf.write(binaural_pred_outputfile, binaural_pred.transpose(), h.sampling_rate)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_csv', default='/home/h-okano/DiffBinaural/FairPlay/splits_csv/split1/test.csv')
    parser.add_argument('--mel_dir', default='/home/h-okano/DiffBinaural/processed_data/generated_mel')
    parser.add_argument('--output_dir', default='/home/h-okano/DiffBinaural/results')
    parser.add_argument('--checkpoint_file', default='/home/h-okano/DiffBinaural/hifi-gan/cp_hifigan_mel64/g_00025000')
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

