import os, sys, pdb,argparse
import whisper
import speech_recognition as sr
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
    parser.add_argument('--Noisy_path', type=str, default='/mnt/ASR_test/Data/TMHINTQI_V2_PCM')
    parser.add_argument('--text_path', type=str, default='/mnt/ASR_test/Data/320.csv') #transformerencoder
    parser.add_argument('--score_floder', type=str, default='./score') #transformerencoder
    parser.add_argument('--model', type=str, default='base')
    parser.add_argument('--task', type=str, default='TMHINT_QI_V2') #transformerencoder

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    # get parameter
    args = get_args()
    score_path = args.score_floder+'/'+args.task +'_'+ args.model +'_'+args.Noisy_path.split('/')[-2]+'_'+args.Noisy_path.split('/')[-1]+'.csv'

    print('model name =', args.model)
    print('task =', args.task)
    print('Noisy_path =', args.Noisy_path)
    print('score_path = ', score_path)
    
    print('Make Score floder...')
    check_folder(score_path)
    if os.path.exists(score_path):
        os.remove(score_path)
    with open(score_path, 'a') as f:
            f.write('Filename, google_CER, whisper_tiny_CER,whisper_base_CER,whisper_small_CER,whisper_medium_CER,whisper_large_CER\n')

    file_path = get_filepaths(args.Noisy_path, ftype='wav')
    model1 = whisper.load_model("tiny")
    model2 = whisper.load_model("base")
    model3 = whisper.load_model("small")
    model4 = whisper.load_model("medium")
    model5 = whisper.load_model("large")
    r = sr.Recognizer()
    
    for path in tqdm(file_path):

        wave_name = path.split('/')[-1].split('.')[0]
        num = 10*(int(wave_name.split('_')[-2])-1) + (int(wave_name.split('_')[-1])-1) 
              
        t_pred = model1.transcribe(path)
        b_pred = model2.transcribe(path)
        s_pred = model3.transcribe(path)
        m_pred = model4.transcribe(path)
        l_pred = model5.transcribe(path)
        
        # pdb.set_trace()
        with sr.WavFile(path) as source:
            audio = r.record(source)
            try:
                a_pred = r.recognize_google(audio, language='zh-tw')
            except:
                print("Google Speech Recognition could not understand audio")
                a_pred = 'Wrong Wrong Wrong Wrong Wrong Wrong Wrong'

        tt = open(args.text_path, 'r')
        txt = tt.read().split('\n')[int(num)]
        
        # pdb.set_trace()
        a_cer = cer(txt,a_pred)
        t_cer = cer(txt,t_pred['text'])
        b_cer = cer(txt,b_pred['text'])
        s_cer = cer(txt,s_pred['text'])
        m_cer = cer(txt,m_pred['text'])
        l_cer = cer(txt,l_pred['text'])
        tt.close()

        with open(score_path, 'a') as f:
            f.write(f'{wave_name},{a_cer},{t_cer},{b_cer},{s_cer},{m_cer},{l_cer}\n')
            

    data = pd.read_csv(score_path)
    a_cer_mean = data['google_CER'].to_numpy().astype('float').mean()
    t_cer_mean = data['whisper_tiny_CER'].to_numpy().astype('float').mean()
    b_cer_mean = data['whisper_base_CER'].to_numpy().astype('float').mean()
    s_cer_mean = data['whisper_small_CER'].to_numpy().astype('float').mean()
    m_cer_mean = data['whisper_base_CER'].to_numpy().astype('float').mean()
    l_cer_mean = data['whisper_medium_CER'].to_numpy().astype('float').mean()

    with open(score_path, 'a') as f:
        f.write(','.join(('Average',str(a_cer_mean),str(t_cer_mean),str(b_cer_mean),str(s_cer_mean),str(m_cer_mean),str(l_cer_mean)))+'\n')