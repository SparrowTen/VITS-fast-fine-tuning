import os
import argparse
import json
import torchaudio


class Preprocess:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
    
    def remove_lab(self):
        for root, dirs, files in os.walk(f'./dataset/{self.dataset_name}'):
            for file in files:
                if file.endswith('.lab'):
                    os.remove(os.path.join(root, file))
                    print(f'Remove {file}')
        print('Remove lab done')
    
    def reshape_data(self):
        with open('./configs/finetune_speaker.json', 'r', encoding='utf-8') as f:
            hps = json.load(f)
        target_sr = hps['data']['sampling_rate']
        filelist = list(os.walk(f'./dataset/{self.dataset_name}'))[0][2]
        if target_sr != 22050:
            for wavfile in filelist:
                wav, sr = torchaudio.load(f'./dataset/{self.dataset_name}/{wavfile}', frame_offset=0, num_frames=-1, normalize=True, channels_first=True)
                wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
                torchaudio.save(f'./dataset/{self.dataset_name}/{wavfile}', wav, target_sr, channels_first=True)
        print('Reshape data done')

    def get_dataset_size(self):
        print(f'Total: {len(os.listdir(f"./dataset/{self.dataset_name}"))}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess dataset')
    parser.add_argument('--dataset', type=str, help='Name of dataset')
    args = parser.parse_args()
    
    pre = Preprocess(args.dataset)
    pre.remove_lab()
    pre.reshape_data()
    pre.get_dataset_size()
