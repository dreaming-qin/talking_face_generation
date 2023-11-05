import os
import torch
from tqdm import tqdm
import imageio
import uuid
import numpy as np
from skimage import img_as_ubyte
import cv2
import torchvision.transforms as transforms
from PIL import Image
import torchvision

# 测试代码
if __name__=='__main__':
    import os
    import sys
    path=sys.path[0]
    for _ in range(0):
        path=os.path.dirname(path)
    sys.path.append(path)
    sys.path.append(os.path.join(path,'Deep3DFaceRecon_pytorch'))

from src.util.logger import logger_config
from src.dataset.renderDtaset import RenderDataset
from src.model.exp3DMM.exp3DMM import Exp3DMM
from src.model.render.render import FaceGenerator
from src.loss.renderLoss import RenderLoss
from src.util.model_util import freeze_params
from src.util.util import get_windows_by_repeat,get_window


# 训练时，固定住exp3DMM模块，然后进行训练


@torch.no_grad()
def save_result(exp_3dmm_model,render,dataloader,save_dir,save_video_num):
    '''保存结果
    save_video_num指的是保存的视频个数'''
    os.makedirs(save_dir,exist_ok=True)

    for data in dataloader:
        if save_video_num==0:
            break
        # 把数据放到device中
        for key,value in data.items():
            if value.shape[0]>save_video_num:
                value=value[:save_video_num]
            data[key]=value.to(next(render.parameters()).device)
        transformer_video=data['video_input']
        save_video_num-=transformer_video.shape[0]
        transformer_video=transformer_video.permute(0,1,4,2,3)
        # 先获得exp 3dmm
        exp_3dmm=exp_3dmm_model(transformer_video,data['audio_input'])

        # 然后处理数据，方便作为render输入
        # [B,len,73]
        driving_source=torch.cat((exp_3dmm,data['pose']),dim=-1)
        np.save('temp.npy',driving_source.cpu().numpy())
        # [B,len,win size,73]
        driving_source=get_window(driving_source,config['render_win_size'])
        driving_source=driving_source.permute(0,1,3,2)


        input_image=[]
        for img in data['img']:
            # imageio.imsave('temp.png',(img.cpu().numpy()*255).astype(np.uint8))
            # frame = cv2.imread('temp.png')
            # os.remove('temp.png')
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # src_img_raw = Image.fromarray(frame)
            # image_transform = transforms.Compose(
            #     [
            #         transforms.ToTensor(),
            #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            #     ]
            # )
            # src_img = image_transform(src_img_raw)

            # [3,H,W]
            # tmp_img=src_img
            tmp_img=img.permute(2,0,1)
            # [len,3,H,W]
            tmp_img=tmp_img.expand(driving_source.shape[1],  -1, -1, -1)
            input_image.append(tmp_img)
        # [B,len,3,H,W]
        input_image=torch.stack(input_image)


        output_imgs=[]
        warp_img=[]
        driving_source=driving_source[0]
        driving_source = torch.split(driving_source, 10, dim=0)
        for win_exp in driving_source:
            cur_src_img = input_image[0][0].expand(win_exp.shape[0], -1, -1, -1).to(win_exp.device)
            output_dict = render(cur_src_img, win_exp)
            output_imgs.append(output_dict["fake_image"])
            warp_img.append(output_dict['warp_image'])
        output_imgs = torch.cat(output_imgs, 0).unsqueeze(0)
        warp_img = torch.cat(warp_img, 0).unsqueeze(0)


        # output_imgs = (output_imgs[0]* 255).to(torch.uint8).permute(0, 2, 3, 1)
        # torchvision.io.write_video('temp.mp4', output_imgs.cpu(), fps=25)

        # 输出为视频
        for real_video,video_input,img,warp,fake_video in zip(data['raw_video'],data['video_input'],data['img'],warp_img,output_imgs):
            # [len,H,W,3]
            real_video=real_video.detach().cpu().numpy()
            # [len,H,W,3]
            video_input=video_input.detach().cpu().numpy()
            # [len,H,W,3]
            fake_video=fake_video.permute(0,2,3,1)
            fake_video=fake_video.detach().cpu().numpy()
            # [len,H,W,3]
            warp=warp.permute(0,2,3,1)
            warp=warp.detach().cpu().numpy()
            # [len,H,W,3]
            img=img.expand(fake_video.shape[0],  -1, -1, -1)
            img=img.detach().cpu().numpy()
            save_path=os.path.join(save_dir,'{}.mp4'.format(str(uuid.uuid1())))
            # [len,H,4*W,3]
            video=np.concatenate((video_input,real_video,img,warp,fake_video),axis=2)
            imageio.mimsave(save_path,(video*255.0).astype('uint8') )

    return


def run(config):
    # 创建一些文件夹
    os.makedirs(config['result_dir'],exist_ok=True)

    # 加载device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 测试集
    test_dataset=RenderDataset(config,type='test')
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        # batch_size必须为1
        batch_size=1, 
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=test_dataset.collater
    )     

    #exp 3dmm model
    exp_model=Exp3DMM(config)
    exp_model=exp_model.to(device)
    exp_model= torch.nn.DataParallel(exp_model, device_ids=config['device_id'])
    # # 必须要有预训练模型
    state_dict=torch.load(config['exp_3dmm_pre_train'])
    exp_model.load_state_dict(state_dict)
    # exp 3dmm模块不会改变参数，设置为eval模式且固定参数
    exp_model.eval()
    freeze_params(exp_model)


    #render model
    render=FaceGenerator(config['mapping_net'],config['warpping_net'],
                         config['editing_net'],config['common'])
    render=render.to(device)
    render= torch.nn.DataParallel(render, device_ids=config['device_id'])
    state_dict=torch.load(config['render_pre_train'])
    render.load_state_dict(state_dict)
    render.eval()
    freeze_params(render)


    save_result(exp_model,render,test_dataloader,
                os.path.join(config['result_dir'],'test'),
                save_video_num=10)


if __name__ == '__main__':
    import os,sys
    import yaml

    # checkpoint = torch.load('checkpoint/render/render.pth', map_location=lambda storage, loc: storage)
    # renser=checkpoint["net_G_ema"]
    # torch.save(renser,'checkpoint/render/render.pth')

    # checkpoint = torch.load('checkpoint/render/render.pth')
    # render={}
    # for key,value in checkpoint.items():
    #     render[f'module.{key}']=value
    # torch.save(render,'checkpoint/render/render2.pth')

    config={}
    yaml_file=['config/data_process/common.yaml','config/dataset/common.yaml','config/model/render.yaml']
    for a in yaml_file:
        with open(a,'r',encoding='utf8') as f:
            config.update(yaml.safe_load(f))

    run(config)
