import os
import imageio
import pickle
import numpy as np
import zlib




# with open('data/001.pkl','rb') as f:
#     info=f.read()
# info=zlib.decompress(info)
# info= pickle.loads(info)
# process_video=info['face_video']
# video_array=process_video
# imageio.mimsave('001.mp4',video_array,fps=30)


import os
import moviepy.video.io.ImageSequenceClip
from moviepy.editor import VideoFileClip

def add_mp3(video_src1, video_src2, video_dst):
    ' 将video_src1的音频嵌入video_src2视频中'
    video_src1 = VideoFileClip(video_src1)
    video_src2 = VideoFileClip(video_src2)
    audio = video_src1.audio
    videoclip2 = video_src2.set_audio(audio)
    videoclip2.write_videofile(video_dst, codec='libx264')

video_src1 = '001.mp4'
video_src2 = 'temp.mp4'
video_dst = '002.mp4'
add_mp3(video_src1, video_src2, video_dst)