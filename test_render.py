import os
import torch
from tqdm import tqdm
import torchvision
import glob
import zlib
import pickle
import imageio
import numpy as np
import shutil


# test
if __name__=='__main__':
    import os
    import sys
    path=sys.path[0]
    sys.path.append(os.path.join(path,'Deep3DFaceRecon_pytorch'))
    sys.path.append(os.path.join(path,'syncnet_python'))
    sys.path.append(os.path.join(path,'TDDFA_V2_master'))


from src.model.render.render import Render
from src.util.model_util import freeze_params
from src.util.util import get_window
from src.util.util import remove_file

from src.metrics.AED import AED_by_dir 
from src.metrics.APD import APD_by_dir
from src.metrics.F_LMD import F_LMD_by_dir
from src.metrics.SyncNet import sync_net_by_dir
from src.metrics.M_LMD import M_LMD_by_dir
from src.metrics.SSIM import ssim_by_dir 

'''对render，主要是得知道效果如何，只需要输入是同一个人的情况就行'''

@torch.no_grad()
def get_metrices(config):
    real_dir='{}/result/real'.format(config['result_dir'])
    fake_dir='{}/result/fake'.format(config['result_dir'])

    print('获得结果...')
    metrices_list=['AED','APD','F_LMD','LSE-C','LSE-D','M_LMD','SSIM']
    metrices_dict={}
    metrices_dict['AED']=AED_by_dir(fake_dir,real_dir,is_get_3dmm=True,is_delete_3dmm_file=True)
    metrices_dict['APD']=APD_by_dir(fake_dir,real_dir)
    metrices_dict['F_LMD']=F_LMD_by_dir(fake_dir,real_dir)
    metrices_dict['LSE-C'],metrices_dict['LSE-D'],metrices_dict['real_LSE-C'],metrices_dict['real_LSE-D']=\
        sync_net_by_dir(fake_dir,real_dir)
    metrices_dict['M_LMD']=M_LMD_by_dir(fake_dir,real_dir)
    metrices_dict['SSIM']=ssim_by_dir(fake_dir,real_dir)

    # gt版本
    metrices_dict['real_AED']=0
    metrices_dict['real_APD']=0
    metrices_dict['real_F_LMD']=0
    metrices_dict['real_M_LMD']=0
    metrices_dict['real_SSIM']=1

    # 写入结果
    lines=['','predict','GT']
    for metrices in metrices_list:
        lines[0]+=f',{metrices}'
        lines[1]+=f',{metrices_dict[metrices]}'
        lines[2]+=',{}'.format(metrices_dict[f'real_{metrices}'])
    save_file='{}/result.csv'.format(os.path.dirname(fake_dir))
    with open(save_file,'w',encoding='utf8') as f:
        for line in lines:
            line+='\n'
            f.writelines(line)

    # 删除fake和real文件夹
    # remove_file(real_dir)
    # remove_file(fake_dir)

@torch.no_grad()
def generate_video(config):

    # 加载device
    device = torch.device(config['device_id'][0] if torch.cuda.is_available() else "cpu")

    #render model
    render=Render(config['mapping_net'],config['warpping_net'],
                         config['editing_net'],config['common'])
    render=render.to(device)
    render= torch.nn.DataParallel(render, device_ids=config['device_id'])
    # 必须得有预训练文件
    print('render加载预训练模型{}'.format(config['render_pre_train']))
    state_dict=torch.load(config['render_pre_train'],map_location=torch.device('cpu'))
    render.load_state_dict(state_dict)
    freeze_params(render)
    render.eval()

    # 拿数据
    file_list=sorted(glob.glob('{}/test/*/*.pkl'.format(config['format_output_path'])))
    file_list=np.array(file_list)[:config['video_num']]
    print('生成视频中...')
    for file in tqdm(file_list):
        # 解压pkl文件
        with open(file,'rb') as f:
            byte_file=f.read()
        byte_file=zlib.decompress(byte_file)
        data= pickle.loads(byte_file)
        # 把数据放到device中
        data['face_video']=torch.tensor((data['face_video']/255*2)-1).float().to(device)
        data['face_coeff']['coeff']=torch.tensor(data['face_coeff']['coeff']).float().to(device)
        data['face_coeff']['transform_params']= \
            torch.tensor(data['face_coeff']['transform_params']).float().to(device)

        #[B,3,H,W]
        real_video=data['face_video'].permute(0,3,1,2)
        # 选第一张作为源图片
        input_img=real_video[0].expand(len(real_video),-1,-1,-1)
        # 获得驱动源(B,73,win size)
        driving_src=get_drving(data,config['render_win_size'])

        # 送入render
        fake_video=[]
        # 由于GPU限制，10张10张送
        frame_num=config['frame_num']
        i=-frame_num
        for i in range(0,len(input_img)-frame_num,frame_num):
            output_dict = render(input_img[i:i+frame_num], driving_src[i:i+frame_num])
            # 得到结果(B,3,H,W)
            fake_video.append(output_dict['fake_image'])
        output_dict = render(input_img[i+frame_num:], driving_src[i+frame_num:])
        fake_video.append(output_dict['fake_image'])
        fake_video=torch.cat(fake_video)

        real_save_file='{}/result/real/{}'.format(config['result_dir'],os.path.basename(file).replace('.pkl','.mp4'))
        real_video=data['face_video']
        save_video(real_video,data,real_save_file)

        fake_save_file='{}/result/fake/{}'.format(config['result_dir'],os.path.basename(file).replace('.pkl','.mp4'))
        fake_video=fake_video.permute(0,2,3,1)
        save_video(fake_video,data,fake_save_file)

        
        save_file='{}/result/merge/{}'.format(config['result_dir'],os.path.basename(file).replace('.pkl','.mp4'))
        video=torch.cat((real_video,fake_video),dim=2)
        save_video(video,data,save_file)

def get_drving(data,win_size):
    '''输入是字典数据，value是torch类型
    输出是驱动3dmm，形状是[B,73(exp 3dmm+pose),2*win_size+1]'''
    # 表情3dmm信息
    data_3DMM=data['face_coeff']['coeff']
    # [len,64]
    face3d_exp = data_3DMM[:, 80:144]  # expression 3DMM range
    
    # 姿势信息
    mat_dict = data['face_coeff']
    np_3dmm = mat_dict["coeff"]
    angles = np_3dmm[:, 224:227]
    translations = np_3dmm[:, 254:257]
    np_trans_params = mat_dict["transform_params"]
    crop = np_trans_params[:, -3:]
    # [len,9]
    pose_params = torch.cat((angles, translations, crop), dim=1)

    driving_src=torch.cat((face3d_exp,pose_params),dim=1).unsqueeze(0)

    driving_src=get_window(driving_src,win_size)
    driving_src=driving_src.squeeze().permute(0,2,1)
    return driving_src

def save_video(video,data,save_file):
    '''对保存视频的封装
    video是要保存的视频，torch类型数据，形状(B,H,W,3)，取值范围(-1,1)
    data是字典数据
    save_file是保存的文件路径'''
    os.makedirs(os.path.dirname(save_file),exist_ok=True)

    # 先把视频存好
    torchvision.io.write_video('temp.mp4', ((video+1)/2*255).cpu(), fps=25)

    # 将data['path']的音频嵌入temp.mp4视频中，保存在save_file中
    command ='ffmpeg -i {} -i {} -loglevel error -c copy -map 0:0 -map 1:1 -y -shortest {}'.format(
        'temp.mp4',data['path'],save_file
    ) 
    os.system(command)
    # 删除temp.mp4
    remove_file('temp.mp4')


if __name__ == '__main__':
    import os,sys
    import yaml

    config={}
    yaml_file=['config/dataset/common.yaml','config/model/render.yaml']
    for a in yaml_file:
        with open(a,'r',encoding='utf8') as f:
            config.update(yaml.safe_load(f))
    
    # 由于GPU限制，得10张10张的往GPU送
    config['frame_num']=10
    # 设置生成的视频数量最大值
    config['video_num']=50


    generate_video(config)
    get_metrices(config)
