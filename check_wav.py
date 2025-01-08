import os
from scipy.io import wavfile

# フォルダのパス
folder_path = '/home/h-okano/DiffBinaural/FairPlay/binaural_audios_16000Hz'

# フォルダ内のファイルを取得
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    # WAVファイルかどうかをチェック
    if filename.endswith('.wav'):
        try:
            # WAVファイルを読み込む
            samplerate, data = wavfile.read(file_path)
            
            # データの長さを取得（モノラル/ステレオに対応）
            sample_count = data.shape[0] if len(data.shape) > 1 else len(data)
            
            # サンプル数が10,000以下ならば出力
            if sample_count <= 800000:
                print(f"ファイル名: {filename}, サンプル数: {sample_count}")
        except Exception as e:
            print(f"ファイル {filename} の読み込みに失敗しました: {e}")
