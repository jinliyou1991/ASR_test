import os, sys, pdb, argparse
import whisper
import speech_recognition as sr
from jiwer import wer,cer
import pandas as pd
from tqdm import tqdm
import torch
from utils import *
torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Noisy_path', type=str, default='/mnt/ASR_test/Data/TMHINTQI_V2_PCM')
    parser.add_argument('--text_path', type=str, default='/home/jin/mnt/ASR_test/Data/320.csv') #transformerencoder
    parser.add_argument('--score_floder', type=str, default='./score') #transformerencoder
    parser.add_argument('--results_floder', type=str, default='./results') #transformerencoder
    parser.add_argument('--task', type=str, default='TMHINT_QI_V2') #transformerencoder

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    # get parameter
    args = get_args()
    score_path = args.score_floder+'/'+args.task +'_'+args.Noisy_path.split('/')[-2]+'_'+args.Noisy_path.split('/')[-1]+'.csv'
    results_path = args.results_floder+'/'+args.task +'_'+args.Noisy_path.split('/')[-2]+'_'+args.Noisy_path.split('/')[-1] +'.csv'

    print('task =', args.task)
    print('Noisy_path =', args.Noisy_path)
    print('score_path = ', score_path)    
    print('results_path = ', results_path)
    print('Make Score floder...')

    check_folder(score_path)
    if os.path.exists(score_path):
        os.remove(score_path)
    with open(score_path, 'a') as f1:
            f1.write('Filename,google_CER,whisper_base_CER,whisper_large-v2_CER,whisper_large-v3_CER\n') 
    check_folder(results_path)
    if os.path.exists(results_path):
        os.remove(results_path)
    with open(results_path, 'a') as f2:
            f2.write('Filename,google,whisper_base,whisper_large-v2,whisper_large-v3\n') 

    file_path = get_filepaths(args.Noisy_path, ftype='wav')
    model1 = whisper.load_model("base", device=DEVICE)
    model2 = whisper.load_model("large-v2", device=DEVICE)
    model3 = whisper.load_model("large-v3", device=DEVICE)

    all_g_pred =''
    all_b_pred =''
    all_l2_pred =''
    all_l3_pred =''
    all_ans =''
    for path in tqdm(file_path):

        wave_name = path.split('/')[-1].split('.')[0]
        num = 10*(int(wave_name.split('_')[-2])-1) + (int(wave_name.split('_')[-1])-1) 
              
        g_pred = google_asr(path, language='zh-tw')
        b_pred = model1.transcribe(path, language="zh")['text']
        l2_pred = model2.transcribe(path, language="zh")['text']
        l3_pred = model3.transcribe(path, language="zh")['text']

        tt = open(args.text_path, 'r')
        txt = tt.read().split('\n')[int(num)]
        
        g_wer = ncer(txt,g_pred)
        b_wer = ncer(txt,b_pred)
        l2_wer = ncer(txt,l2_pred)
        l3_wer = ncer(txt,l3_pred)
        tt.close()
        
        all_g_pred = all_g_pred + ' '+ g_pred
        all_b_pred = all_b_pred + ' ' + b_pred
        all_l2_pred = all_l2_pred + ' ' + l2_pred
        all_l3_pred = all_l3_pred + ' ' + l3_pred
        all_ans = all_ans + txt

        with open(score_path, 'a') as f1:
            f1.write(f'{wave_name},{g_wer},{b_wer},{l2_wer},{l3_wer}\n')
        with open(results_path, 'a') as f2:
            f2.write(f'{wave_name},{g_pred},{b_pred},{l2_pred},{l3_pred}\n')
            

    data = pd.read_csv(score_path)
    g_wer_mean = data['google_CER'].to_numpy().astype('float').mean()
    b_wer_mean = data['whisper_base_CER'].to_numpy().astype('float').mean()
    l2_wer_mean = data['whisper_large-v2_CER'].to_numpy().astype('float').mean()
    l3_wer_mean = data['whisper_large-v3_CER'].to_numpy().astype('float').mean()

    with open(score_path, 'a') as f1:
        f1.write(','.join(('Average',str(g_wer_mean),str(b_wer_mean),str(l2_wer_mean),str(l3_wer_mean)))+'\n')
        
    all_g_wer = ncer(all_ans,all_g_pred)
    all_b_wer = ncer(all_ans,all_b_pred)
    all_l2_wer = ncer(all_ans,all_l2_pred)
    all_l3_wer = ncer(all_ans,all_l3_pred)
    
    with open(score_path, 'a') as f1:
        f1.write(','.join(('All_WER',str(all_g_wer),str(all_b_wer),str(all_l2_wer),str(all_l3_wer)))+'\n')
    with open(results_path, 'a') as f2:
        f2.write(','.join(('All_Pred',str(all_g_pred),str(all_b_pred),str(all_l2_pred),str(all_l3_pred)))+'\n')
