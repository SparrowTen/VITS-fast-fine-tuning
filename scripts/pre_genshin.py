import os
import argparse
import json
import torchaudio


class Preprocess:
    def __init__(self, dataset_name, language):
        self.dataset_name = dataset_name
        self.language = language
        self.lang2token = {
            'J': '[JA]',
        }
    
    def gen_anno(self, filename, language, lab):
        '''
            ./dataset/派蒙/processed_vo_ABDLQ001_1_paimon_01.wav|派蒙|[JA]あれ?ティマヨスいないみたいだなぁ[JA]
        '''
        return f'./dataset/{self.dataset_name}/{filename}.wav|{self.dataset_name}|{language}{lab}{language}\n'
    
    def lab2anno(self):
        labs = []
        for root, dirs, files in os.walk(f'./dataset/{self.dataset_name}'):
            for file in files:
                if file.endswith('.lab'):
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        line = f.readline()
                        labs.append(self.gen_anno(file.split('.')[0], self.lang2token[self.language], line))
                    os.remove(os.path.join(root, file))
                    print(f'Remove {file}')
        with open(f'./anno/short_character_anno_{self.dataset_name}.txt', 'w', encoding='utf-8') as f:
            for lab in labs:
                f.write(lab)
        print('Generate annotation done')
    
    def reshape_data(self):
        with open('./configs/finetune_speaker.json', 'r', encoding='utf-8') as f:
            hps = json.load(f)
        target_sr = hps['data']['sampling_rate']
        filelist = list(os.walk(f'./dataset/{self.dataset_name}'))[0][2]
        for wavfile in filelist:
            wav, sr = torchaudio.load(f'./dataset/{self.dataset_name}/{wavfile}', frame_offset=0, num_frames=-1, normalize=True, channels_first=True)
            if sr != target_sr:
                wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
                torchaudio.save(f'./dataset/{self.dataset_name}/{wavfile}', wav, target_sr, channels_first=True)
        print('Reshape data done')

    def get_dataset_size(self):
        print(f'Total: {len(os.listdir(f"./dataset/{self.dataset_name}"))}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess dataset')
    parser.add_argument('--dataset', type=str, help='Name of dataset')
    parser.add_argument('--language', default='CJE', help='Languages')
    args = parser.parse_args()
    
    pre = Preprocess(args.dataset, args.language)
    pre.lab2anno()
    pre.reshape_data()
    pre.get_dataset_size()
