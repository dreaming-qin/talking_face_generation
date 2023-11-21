
def merge_data2(dir_name):
    # 获得所有信息后，开始合并所需要的信息
    # 从dormat_data入手
    filenames = glob.glob('data2/format_data/*/*/*.pkl')
    # 存到这
    for file in filenames:
        info={}
        # test,解压测试
        with open(file,'rb') as f:
            data=f.read()
        data=zlib.decompress(data)
        data= pickle.loads(data)
        info.update(data)
        info.pop('frame_index')
        info.pop('path')

        # 再从视频中拿东西
        video_path=data['path']
        with open(video_path.replace('.mp4','_temp1.pkl'),'rb') as f:
            process_video= pickle.load(f)
        info['face_video']=process_video['face_video']
        info['mouth_mask']=process_video['mouth_mask']
        # 完了之后，删除没有必要的数据
        os.remove(video_path.replace('.mp4','_temp1.pkl'))

        
        # 没有视频信息的，直接弃用=
        frame=info['face_video'][0]
        if np.sum(frame==0)>3*256*256/4:
            continue

        # 获得最终数据，使用压缩操作
        save_file=file.replace('data2','data')
        os.makedirs(os.path.dirname(save_file),exist_ok=True)
        info = pickle.dumps(info)
        info=zlib.compress(info)
        with open(save_file,'wb') as f:
            f.write(info)




a=2
while True:
    if a%24==0:
        break
    a*=2
print(a)