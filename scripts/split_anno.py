# ./custom_character_voice/VO_paimon/processed_690.wav|VO_paimon|[ZH]应该是一个乐器,只不过太久没人保养,好像已经没法弹奏了。[ZH]
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='split anno')
    parser.add_argument('--txt', type=str, help='anno file')
    parser.add_argument('--speaker', type=str, help='speaker name')
    args = parser.parse_args()
    lang = ['[ZH]', '[EN]', '[JA]']
    
    with open(args.txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            raw = line.strip().split('|')
            orgin = raw[0].split('/')[-1]
            sub = raw[2]
            for l in lang:
                if l in sub:
                    sub = sub.replace(l, '')
                    break
            os.makedirs('./sub', exist_ok=True)
            with open(f'./sub/{args.speaker}_sub.txt', 'a', encoding='utf-8') as f:
                f.write(f'{orgin}|{sub}\n')