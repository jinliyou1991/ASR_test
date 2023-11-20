import os, sys, pdb, argparse
import jiwer, re
import pandas as pd
import speech_recognition as sr
r = sr.Recognizer()

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

def google_asr(path, language):
    with sr.WavFile(path) as source:
        audio = r.record(source)
        try:
            a_pred = r.recognize_google(audio, language=language)
        except:
            a_pred = ''
    return a_pred

def normal_sen(sentence):
    sentence = sentence.lower()
    # remove all punctuation except words and space
    sentence = re.sub(r'[^\w\s]','', sentence)
    sentence = sentence.strip()
    return sentence

def nwer(ans, pred):
    speech_wer = round(jiwer.wer(normal_sen(ans), normal_sen(pred)),5) 
    # speech_wer = jiwer.wer(jiwer.RemovePunctuation()(ans),jiwer.RemovePunctuation()(pred),jiwer.wer_standardize,jiwer.wer_standardize)
    return speech_wer
def ncer(ans, pred):
    speech_cer = round(jiwer.cer(normal_sen(ans.replace(' ', '')), normal_sen(pred.replace(' ', ''))),5)
    # speech_wer = jiwer.cer(jiwer.RemovePunctuation()(ans.replace(' ', '')),jiwer.RemovePunctuation()(pred.replace(' ', '')),jiwer.wer_standardize,jiwer.wer_standardize)
    return speech_cer
