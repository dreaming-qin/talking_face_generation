import glob
import math
import pickle
import os
import json
import numpy as np



def process_audio(video_dir):
    r'''输入视频目录路径，处理所有视频与之对应的audio结果
    1.打算从video中拿出audio信息保证同步，并分析
    '''
    filenames = glob.glob(f'{video_dir}/*.mp4')
    for video_file in filenames:
        audio_file=video_file.replace('.mp4','.wav')
        # 转为16kHz音频
        cmd = 'ffmpeg -i {} -loglevel error -y -f wav -ac 1 -acodec pcm_s16le -ar 16000 {}'.format(
                video_file, audio_file)
        os.system(cmd)

        # 获取音素信息
        word=process_train_audio(audio_file)
        
        # 写入数据
        info={}
        info['audio_word']=np.array(word)
        with open(video_file.replace('.mp4','_audio.pkl'),'wb') as f:
            info =  pickle.dumps(info)
            f.write(info)

        # 删除不必要的数据
        os.remove(audio_file)
    return

with open("phindex.json",'r', encoding='UTF-8') as f:
    word_dict = json.load(f)
def process_train_audio(audio_file):
    '''将音频转为音素'''
    audio_command = f'cd /workspace/code/ez-phones-master && bash ps_shortcut.sh {os.path.abspath(audio_file)}'

    r = os.popen(audio_command)
    text = r.read()
    r.close()  
    text_list=text.split('\n')

    ans=[]
    for text in text_list:
        try:
            word,_,e_time,_=text.split(' ')
            e_time=float(e_time)
        except:
            continue
        length=math.floor((e_time*1000-1)/40)
        if word in word_dict:
            temp=[word_dict[word] for _ in range(length-len(ans))]
        else:
            print(f'无法识别的音素{word}，长度是{length-len(ans)}，在{audio_file}')
            if len(ans)!=0:
                temp=[ans[-1] for _ in range(length-len(ans))]
        ans+=temp
	
    return ans

if __name__=='__main__':
    process_audio('/workspace/dataset/MEAD/M003/video/front/angry/level_1')