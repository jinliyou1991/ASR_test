import numpy as np
import soundfile as sf

from pathlib import Path
from tqdm import tqdm

# main
if __name__ == '__main__':
    audio_files = list(Path.cwd().joinpath('TMHINTQI_V2').glob('*.wav'))

    for audio_file in tqdm(audio_files):
        data, samplerate = sf.read(str(audio_file), dtype='float32')
        data = np.int16(data * 32767)
        sf.write(f"TMHINTQI_V2_PCM/{audio_file.stem}.wav", data, samplerate, 'PCM_16')
