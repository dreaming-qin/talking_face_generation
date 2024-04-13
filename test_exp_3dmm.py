import os
import torch
from tqdm import tqdm
import glob
import zlib
import pickle
import numpy as np


# test
if __name__=='__main__':
    import os
    import sys
    path=sys.path[0]
    sys.path.append(os.path.join(path,'Deep3DFaceRecon_pytorch'))
    sys.path.append(os.path.join(path,'syncnet_python'))
    sys.path.append(os.path.join(path,'TDDFA_V2_master'))


from src.model.render.render import Render
from src.model.exp3DMM.exp3DMM import Exp3DMM
from src.util.model_util import freeze_params
from src.util.util import get_window
from test_render import save_video
from test_render import get_metrices

'''对exp 3dmm，需要知道输入来源不同的情况，因此得补全'''

@torch.no_grad()
def generate_video(config):
    '''输入是auido mfcc,style_clip ,video, img。
    输出是视频'''

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
    
    # 3dmm model
    exp_model=Exp3DMM(config)
    exp_model=exp_model.to(device)
    exp_model= torch.nn.DataParallel(exp_model, device_ids=config['device_id'])
    print('exp 3dmm加载预训练模型{}'.format(config['exp_3dmm_pre_train']))
    # 必须得有预训练文件
    state_dict=torch.load(config['exp_3dmm_pre_train'],map_location=torch.device('cpu'))
    exp_model.load_state_dict(state_dict)
    freeze_params(exp_model)
    exp_model.eval()

    # 拿数据
    file_list=sorted(glob.glob('{}/test/*/*.pkl'.format(config['format_output_path'])))
    file_list=np.array(file_list)[:config['video_num']].tolist()
    print('生成视频中...')
    for file in tqdm(file_list):
        # 解压pkl文件
        with open(file,'rb') as f:
            byte_file=f.read()
        byte_file=zlib.decompress(byte_file)
        data= pickle.loads(byte_file)

        data=to_device(data,device,config)

        # import imageio
        # imageio.mimsave('test.mp4',data['video'][0].cpu().numpy())

        # 生成exp 3dmm
        exp_result=exp_model(data['video'].permute(0,1,4,2,3),data['audio'])
        driving_src=torch.cat((exp_result,data['pose']),dim=2)
        driving_src=get_window(driving_src,config['render_win_size'])
        # (len,dim,win)
        driving_src=driving_src.squeeze().permute(0,2,1)
        # 送入render
        fake_video=[]
        # 由于GPU限制，10张10张送
        frame_num=config['frame_num']
        i=-frame_num
        for i in range(0,len(data['img'])-frame_num,frame_num):
            output_dict = render(data['img'][i:i+frame_num], driving_src[i:i+frame_num])
            # 得到结果(B,3,H,W)
            fake_video.append(output_dict['fake_image'])
        output_dict = render(data['img'][i+frame_num:], driving_src[i+frame_num:])
        fake_video.append(output_dict['fake_image'])
        fake_video=torch.cat(fake_video)

        real_save_file='{}/result/real/{}'.format(config['result_dir'],os.path.basename(file).replace('.pkl','.mp4'))
        real_video=data['real_video']
        save_video(real_video,data,real_save_file)

        fake_save_file='{}/result/fake/{}'.format(config['result_dir'],os.path.basename(file).replace('.pkl','.mp4'))
        fake_video=fake_video.permute(0,2,3,1)
        save_video(fake_video,data,fake_save_file)

        save_file='{}/result/merge/{}'.format(config['result_dir'],os.path.basename(file).replace('.pkl','.mp4'))
        video=torch.cat((real_video,fake_video),dim=2)
        save_video(video,data,save_file)

def to_device(data,device,config):
    '''将相关数据放到data中，并返回需要的数据'''
    # 把数据放到device中
    ans={}
    # gt视频获得
    ans['real_video']=torch.tensor((data['face_video']/255*2)-1).float().to(device)

    # 选第一张为源图片
    # (3,h,w)
    ans['img']=ans['real_video'][0].expand(len(ans['real_video']),-1,-1,-1).permute(0,3,1,2)

    # audio获得
    # (len,28,dim)
    ans['audio']=torch.tensor(data['audio_mfcc']).float().to(device)

    # 可能长度无法对齐，调整
    if len(ans['audio'])<len(ans['real_video']):
        ans['real_video']=ans['real_video'][:len(ans['audio'])]
    else:
        ans['audio']=ans['audio'][:len(ans['real_video'])]
    # 因为audio窗口问题，需要扩展
    temp_first=ans['audio'][0].expand(config['audio_win_size'],-1,-1)
    temp_last=ans['audio'][-1].expand(config['audio_win_size'],-1,-1)
    ans['audio']=torch.cat((temp_first,ans['audio'],temp_last))
    
    # 获得输入video
    video=data['face_video']
    # mask_list=data['mouth_mask']
    # for img,mask in zip(video,mask_list):
    #     if mask[1]-mask[0]>0 and mask[3]-mask[2]>0:
    #         mask_width=config['mask_width']
    #         mask_height=config['mask_height']
    #         top,bottom,left,right=mask
    #         top_temp=max(0,min(top,top-((mask_height-bottom+top)//2)))
    #         bottom_temp=min(img.shape[0],max(bottom,bottom+((mask_height-bottom+top)//2)))
    #         left_temp=max(0,min(left,left-((mask_width-right+left)//2)))
    #         right_temp=min(img.shape[1],max(right,right+((mask_width-right+left)//2)))
    #         noise=np.random.randint(0,256,(bottom_temp-top_temp,right_temp-left_temp,3))
    #         img[top_temp:bottom_temp,left_temp:right_temp]=noise
    ans['video']=torch.tensor((video/255*2)-1).to(device).float()

    # 因为video窗口问题，需要扩展
    temp_first=ans['video'][0].expand(config['audio_win_size'],-1,-1,-1)
    temp_last=ans['video'][-1].expand(config['audio_win_size'],-1,-1,-1)
    ans['video']=torch.cat((temp_first,ans['video'],temp_last))
    
    # pose信息获得
    mat_dict = data['face_coeff']
    np_3dmm = mat_dict["coeff"]
    angles = np_3dmm[:, 224:227]
    translations = np_3dmm[:, 254:257]
    np_trans_params = mat_dict["transform_params"]
    crop = np_trans_params[:, -3:]
    # [len,9]
    pose_params = np.concatenate((angles, translations, crop), axis=1)
    ans['pose']=torch.tensor(pose_params).float().to(device)


    # 最后的一些处理
    for key,value in ans.items():
        ans[key]=value.unsqueeze(0)
    ans['img']=ans['img'].squeeze(0)
    ans['path']=data['path']
    ans['real_video']=ans['real_video'].squeeze(0)

    # test，拿出表情3DMM
    data_3DMM=data['face_coeff']['coeff']
    face3d_exp = data_3DMM[:, 80:144]  # expression 3DMM range
    ans['face3d_exp']=face3d_exp

    return ans


if __name__ == '__main__':
    import os,sys
    import yaml

    config={}
    yaml_file=['config/dataset/common.yaml','config/model/render.yaml'
               ,'config/model/exp3DMM.yaml']
    for a in yaml_file:
        with open(a,'r',encoding='utf8') as f:
            config.update(yaml.safe_load(f))
    
    # 由于GPU限制，得10张10张的往GPU送
    config['frame_num']=10
    # 设置生成的视频数量最大值
    config['video_num']=500


    generate_video(config)
    # get_metrices(config)

    # # test
    # real_dir='{}/result/real'.format(config['result_dir'])
    # fake_dir='{}/result/fake'.format(config['result_dir'])
    # from src.metrics.SyncNet import sync_net_by_dir
    # print('获得结果...')
    # metrices_dict={}
    # c,d,gt_c,gt_d=\
    #     sync_net_by_dir(fake_dir,real_dir)
    # print(f'{c} {d} {gt_c} {gt_d}')

    # test，获得其它方法的结果
    # method=['make','pc_avs','exp3DMM']
    # for b in method:
    #     config['result_dir']=f'result/{b}'
    #     get_metrices(config)
