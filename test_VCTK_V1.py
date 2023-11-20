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
    parser.add_argument('--Noisy_path', type=str, default='/home/jin/mnt/Intern_SE/Data/clean_testset_wav')
    parser.add_argument('--text_path', type=str, default='/home/jin/mnt/VCTK_28spk/test_txt') #transformerencoder
    parser.add_argument('--score_WER_floder', type=str, default='/home/jin/mnt/ASR_test/score/WER') #transformerencoder
    parser.add_argument('--results_floder', type=str, default='/home/jin/mnt/ASR_test/results') #transformerencoder
    parser.add_argument('--task', type=str, default='VCTK_noisy') #transformerencoder

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    # get parameter
    args = get_args()
    score_WER_path = args.score_WER_floder+'/'+args.task +'.csv'
    results_path = args.results_floder+'/'+args.task +'.csv'

    # print('model name =', args.model)
    print('task =', args.task)
    print('Noisy_path =', args.Noisy_path)
    print('score_WER_path = ', score_WER_path)
    print('results_path = ', results_path)
               
    check_folder(score_WER_path)
    if os.path.exists(score_WER_path):
        os.remove(score_WER_path)
    with open(score_WER_path, 'a') as f1:
            f1.write('Filename,google_WER,whisper_base_WER,whisper_large-v2_WER,whisper_large-v3_WER\n') 
    check_folder(results_path)
    if os.path.exists(results_path):
        os.remove(results_path)
    with open(results_path, 'a') as f2:
            f2.write('Filename,google,whisper_base,whisper_large-v2,whisper_large-v3\n') 
            
    file_path = get_filepaths(args.Noisy_path, ftype='wav')
    model1 = whisper.load_model("base", device=DEVICE)
    model2 = whisper.load_model("large-v2", device=DEVICE)
    model3 = whisper.load_model("large", device=DEVICE)
    
    all_g_pred =''
    all_b_pred =''
    all_l2_pred =''
    all_l3_pred =''
    all_ans =''
    # pdb.set_trace()
    for path in tqdm(file_path):       
        wave_name = path.split('/')[-1].split('.')[0]
        g_pred = google_asr(path, language='us-en')
        b_pred = model1.transcribe(path, language="en")['text']
        l2_pred = model2.transcribe(path, language="en")['text']
        l3_pred = model3.transcribe(path, language="en")['text']
        
        t_path = os.path.join(args.text_path, wave_name + '.txt')
        tt = open(t_path, 'r')
        txt = tt.read().split('\n')[0]
                    
        g_wer = nwer(txt,g_pred)
        b_wer = nwer(txt,b_pred)
        l2_wer = nwer(txt,l2_pred)
        l3_wer = nwer(txt,l3_pred)
        tt.close()
        
        all_g_pred = all_g_pred + ' ' + g_pred
        all_b_pred = all_b_pred + ' ' + b_pred
        all_l2_pred = all_l2_pred + ' ' + l2_pred
        all_l3_pred = all_l3_pred + ' ' + l3_pred
        all_ans = all_ans + txt
        # pdb.set_trace()
        
        with open(score_WER_path, 'a') as f1:
            f1.write(f'{wave_name},{g_wer},{b_wer},{l2_wer},{l3_wer}\n')
        with open(results_path, 'a') as f2:
            f2.write(f'{wave_name},{g_pred},{b_pred},{l2_pred},{l3_pred}\n')
        
    data = pd.read_csv(score_WER_path)
    g_wer_mean = data['google_WER'].to_numpy().astype('float').mean()
    b_wer_mean = data['whisper_base_WER'].to_numpy().astype('float').mean()
    l2_wer_mean = data['whisper_large-v2_WER'].to_numpy().astype('float').mean()
    l3_wer_mean = data['whisper_large-v3_WER'].to_numpy().astype('float').mean()

    with open(score_WER_path, 'a') as f1:
        f1.write(','.join(('Average',str(g_wer_mean),str(b_wer_mean),str(l2_wer_mean),str(l3_wer_mean)))+'\n')
        
    all_g_wer = nwer(all_ans,all_g_pred)
    all_b_wer = nwer(all_ans,all_b_pred)
    all_l2_wer = nwer(all_ans,all_l2_pred)
    all_l3_wer = nwer(all_ans,all_l3_pred)
    
    with open(score_WER_path, 'a') as f1:
        f1.write(','.join(('All_WER',str(all_g_wer),str(all_b_wer),str(all_l2_wer),str(all_l3_wer)))+'\n')
        
    with open(results_path, 'a') as f2:
        f2.write(','.join(('All_Pred',str(all_g_pred),str(all_b_pred),str(all_l2_pred),str(all_l3_pred)))+'\n')
