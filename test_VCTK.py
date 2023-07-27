import os, sys, pdb, argparse
import whisper
from jiwer import wer,cer
import pandas as pd
from tqdm import tqdm

def check_dir(path):
    if not os.path.isdir('/'.join(list(path.split('/')[:-1]))):
        os.makedirs('/'.join(list(path.split('/')[:-1])))
def check_path(path):
    if not os.path.isdir(path): 
        os.makedirs(path)   
def check_folder(path):
    path_n = '/'.join(path.split('/')[:-1])
    check_path(path_n)
def get_filepaths(directory,ftype='.wav'):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ftype):
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.
    return sorted(file_paths)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Noisy_path', type=str, default='/work/jinliyou1991/VCTK_28spk/noisy_testset_wav')
    parser.add_argument('--Clean_path', type=str, default='/work/jinliyou1991/BLSTM_01_jin_0608_VCTK_IRM_epochs50_adam_mse_batch1_lr0.0005') #transformerencoder
    parser.add_argument('--text_path', type=str, default='/work/jinliyou1991/VCTK_28spk/test_txt') #transformerencoder
    parser.add_argument('--score_floder', type=str, default='/home/jinliyou1991/whisper/score') #transformerencoder
    parser.add_argument('--model', type=str, default='base')
    parser.add_argument('--task', type=str, default='VCTK_BLSTM') #transformerencoder

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    # get parameter
    args = get_args()
    score_path = args.score_floder+'/'+args.task +'_'+ args.model+'.csv'

    print('model name =', args.model)
    print('task =', args.task)
    print('Noisy_path =', args.Noisy_path)
    print('score_path = ', score_path)
    
    print('Make Score floder...')
    check_folder(score_path)
    if os.path.exists(score_path):
        os.remove(score_path)
    with open(score_path, 'a') as f:
            f.write('Filename,Noisy_WER,Clean_WER,Noisy_CER,Clean_CER\n')

    file_path = get_filepaths(args.Noisy_path, ftype='wav')
    # pdb.set_trace()
    for path in tqdm(file_path):

        wave_name = path.split('/')[-1].split('.')[0]
        model = whisper.load_model("base")

        n_pred = model.transcribe(path)
        c_path = os.path.join(args.Clean_path, path.split('/')[-1])
        c_pred = model.transcribe(c_path)

        t_path = os.path.join(args.text_path, wave_name + '.txt')
        tt = open(t_path, 'r')
        txt = tt.read().split('\n')[0]
    #     pdb.set_trace()
        n_wer = wer(txt,n_pred['text'])
        c_wer = wer(txt,c_pred['text'])
        n_cer = cer(txt,n_pred['text'])
        c_cer = cer(txt,c_pred['text'])
        tt.close()

        with open(score_path, 'a') as f:
            f.write(f'{wave_name},{n_wer},{c_wer},{n_cer},{c_cer}\n')

    data = pd.read_csv(score_path)
    n_wer_mean = data['Noisy_WER'].to_numpy().astype('float').mean()
    c_wer_mean = data['Clean_WER'].to_numpy().astype('float').mean()
    n_cer_mean = data['Noisy_CER'].to_numpy().astype('float').mean()
    c_cer_mean = data['Clean_CER'].to_numpy().astype('float').mean()

    with open(score_path, 'a') as f:
        f.write(','.join(('Average',str(n_wer_mean),str(c_wer_mean),str(n_cer_mean),str(c_cer_mean)))+'\n')