import numpy as np
import python_speech_features
import torch
from transformers import Wav2Vec2Processor, HubertModel
from moviepy.editor import AudioFileClip
import glob
import imageio
import soundfile as sf
import pickle
import os

wav2vec2_processor = Wav2Vec2Processor.from_pretrained("./checkpoint/hubert")
hubert_model = HubertModel.from_pretrained("./checkpoint/hubert")


def process_audio(video_dir):
    r'''输入视频目录路径，处理所有视频与之对应的audio结果
    1.打算从video中拿出audio信息保证同步，并分析
    '''
    filenames = glob.glob(f'{video_dir}/*.mp4')
    for video_file in filenames:
        # fps指定帧数，保证音频片段和视频片段同步
        fps = 25
        audio_file=video_file.replace('.mp4','.wav')

        my_audio_clip = AudioFileClip(video_file)
        # 16kHz对应25帧的换算方式
        my_audio_clip.write_audiofile(audio_file,fps=int(fps/25*16000))

        # 获取音频信息
        mfcc=process_train_audio(audio_file)
        hugebert=process_syncNet_audio(audio_file)
        
        # 写入数据
        info={}
        info['input_audio_mfcc']=mfcc
        info['syncNet_audio']=hugebert
        with open(video_file.replace('.mp4','_audio.pkl'),'wb') as f:
            info =  pickle.dumps(info)
            f.write(info)

        # 删除不必要的数据
        os.remove(audio_file)
    return

def process_train_audio(audio_file):
    '''将音频转为MFCC'''
    speech, samplerate = sf.read(audio_file)
    # 16kHz对应25帧的换算方式
    fps=samplerate*25/16000
    speech=speech[:,0]
    speech = np.insert(speech, 0, np.zeros(1920))
    speech = np.append(speech, np.zeros(1920))
    mfcc = python_speech_features.mfcc(speech,samplerate,winstep=1/(fps*4))

    ind = 3
    input_mfcc = []
    while ind <= int(mfcc.shape[0]/4) - 4:
        t_mfcc =mfcc[( ind - 3)*4: (ind + 4)*4, 1:]
        input_mfcc.append(t_mfcc)
        ind += 1
    return np.array(input_mfcc)

def process_syncNet_audio(audio_file):
    # 获得hugebert
    speech, _ = sf.read(audio_file)
    hubert_hidden = get_hubert_from_speech(speech)
    return hubert_hidden.detach().numpy()

@torch.no_grad()
def get_hubert_from_speech(speech, device="cuda:1"):
    global hubert_model
    hubert_model = hubert_model.to(device)
    if speech.ndim ==2:
        speech = speech[:, 0] # [T, 2] ==> [T,]
    input_values_all = wav2vec2_processor(speech, return_tensors="pt", sampling_rate=16000).input_values # [1, T]
    input_values_all = input_values_all.to(device)
    # For long audio sequence, due to the memory limitation, we cannot process them in one run
    # HuBERT process the wav with a CNN of stride [5,2,2,2,2,2], making a stride of 320
    # Besides, the kernel is [10,3,3,3,3,2,2], making 400 a fundamental unit to get 1 time step.
    # So the CNN is euqal to a big Conv1D with kernel k=400 and stride s=320
    # We have the equation to calculate out time step: T = floor((t-k)/s)
    # To prevent overlap, we set each clip length of (K+S*(N-1)), where N is the expected length T of this clip
    # The start point of next clip should roll back with a length of (kernel-stride) so it is stride * N
    kernel = 400
    stride = 320
    clip_length = stride * 1000
    num_iter = input_values_all.shape[1] // clip_length
    expected_T = (input_values_all.shape[1] - (kernel-stride)) // stride
    res_lst = []
    for i in range(num_iter):
        if i == 0:
            start_idx = 0
            end_idx = clip_length - stride + kernel
        else:
            start_idx = clip_length * i
            end_idx = start_idx + (clip_length - stride + kernel)
        input_values = input_values_all[:, start_idx: end_idx]
        hidden_states = hubert_model.forward(input_values).last_hidden_state # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    if num_iter > 0:
        input_values = input_values_all[:, clip_length * num_iter:]
    else:
        input_values = input_values_all
    # if input_values.shape[1] != 0:
    if input_values.shape[1] >= kernel: # if the last batch is shorter than kernel_size, skip it            
        hidden_states = hubert_model(input_values).last_hidden_state # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    ret = torch.cat(res_lst, dim=0).cpu() # [T, 1024]
    # assert ret.shape[0] == expected_T
    assert abs(ret.shape[0] - expected_T) <= 1
    if ret.shape[0] < expected_T:
        ret = torch.nn.functional.pad(ret, (0,0,0,expected_T-ret.shape[0]))
    else:
        ret = ret[:expected_T]
    return ret

if __name__=='__main__':
    process_audio('data')