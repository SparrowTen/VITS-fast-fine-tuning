import os
import scipy.io.wavfile as wavf
import torch
from torch import no_grad, LongTensor
import argparse
import commons
from mel_processing import spectrogram_torch
import utils
from models import SynthesizerTrn
from text import text_to_sequence, _clean_text
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
import logging
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

language_marks = {
    'Japanese': '',
    '日本語': '[JA]',
    '简体中文': '[ZH]',
    'English': '[EN]',
    'Mix': '',
}
lang = ['日本語', '简体中文', 'English', 'Mix']
def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def tts_fn(model, text, speaker_id, language, speed):
    if language is not None:
        text = language_marks[language] + text + language_marks[language]
    stn_tst = get_text(text, hps, False)
    with no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
        sid = LongTensor([speaker_id]).to(device)
        audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                            length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
    del stn_tst, x_tst, x_tst_lengths, sid
    return 'Success', (hps.data.sampling_rate, audio)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='./G_latest.pth', help='directory to your fine-tuned model')
    parser.add_argument('--config_dir', default='./finetune_speaker.json', help='directory to your model config file')
    parser.add_argument('--prompts', help='prompts text file for the model')
    parser.add_argument('--speaker', help='output dir speaker name')

    args = parser.parse_args()
    hps = utils.get_hparams_from_file(args.config_dir)

    net_g = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    _ = net_g.eval()
    _ = utils.load_checkpoint(args.model_dir, net_g, None)
    
    speaker_id = 0
    model = net_g
    counter = 1
    if os.path.exists(f'.\output\{args.speaker}') is False:
        os.makedirs(f'.\output\{args.speaker}')
    
    with open(args.prompts, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            orgin = line.split('|')[0].replace('.wav', '')
            prompt = line.split('|')[1].replace('\n', '')
            status, (sampling_rate, audio) = tts_fn(model, prompt, speaker_id, '简体中文', 1.0)
            wavf.write(f'.\output\{args.speaker}\{orgin}_f.wav', hps.data.sampling_rate, audio)
            print(f'Process: {counter}/{len(lines)}')
            counter += 1