from speechbrain.pretrained import EncoderDecoderASR
import torch
import numpy as np
import array
from pydub.silence import detect_nonsilent
class ASR:
    def __init__(self):
        self.asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-aishell", savedir="weights/asr-transformer-aishell") #中文transformer
        self.wav_lens=torch.tensor([1.0])

    def recognize_wav(self,audio):
        result=[]
        audio=audio.set_frame_rate(16000)
        audio=audio.set_channels(1)
        audio=audio.set_sample_width(2)
        # chunks = split_on_silence(audio,min_silence_len=300,silence_thresh=-45)#min_silence_len: 拆分语句时，静默满1秒则拆分。silence_thresh：小于-70dBFS以下的为静默。
        chunks = detect_nonsilent(audio,min_silence_len=300,silence_thresh=-45,seek_step=10) #结果数组起止点单位为ms
        for i,item in enumerate(chunks):
            wave_data=audio[item[0]:item[1]] #ms
            wave_data=np.array(array.array(wave_data.array_type, wave_data._data))
            wave_data=torch.from_numpy(wave_data).unsqueeze(0)
            wave_data=wave_data/32768
            try:
                res=self.asr_model.transcribe_batch(wave_data, self.wav_lens)
            except:
                print(str(i)+"   "+'error')
                continue
            print(str(i)+"   "+str(res[0][0]))
            result.append([item[0],item[1],res[0][0]])
          
        return result
      