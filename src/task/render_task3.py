import os
import torch
from tqdm import tqdm
import torchvision
import numpy as np
import torch.nn.functional as F

# 测试代码
if __name__=='__main__':
    import os
    import sys
    path=sys.path[0]
    for _ in range(2):
        path=os.path.dirname(path)
    sys.path.append(path)
    sys.path.append(os.path.join(path,'Deep3DFaceRecon_pytorch'))

from src.util.logger import logger_config
from src.dataset.renderDtaset import RenderDataset
from src.model.render.render import Render
from src.loss.renderLoss import RenderLoss

from src.metrics.SSIM import ssim as eval_ssim

from Deep3DFaceRecon_pytorch.models.bfm import ParametricFaceModel
from Deep3DFaceRecon_pytorch.util.nvdiffrast import MeshRenderer


'''经过训练，最好的训练参数是：
train_dataset: frame_num=2 workers=3 batch_size=7
test_dataset: frame_num=2 workers=2 batch_size=5
eval_dataset: frame_num=2 workers=2 batch_size=5
optimizer: betas=(0.5, 0.999)
scheduler: gamma=0.2

epoch: 80以及更大
warp_epoch: 40（固定不变）
lr: 0.0001
lr_scheduler_step: [60]（edit阶段剩下那一半）

rec_low_weight: [60,0]
vgg_weight: [1,1,1,1,1]
num_scales: 4

使用预训练的模型的话，最好的参数是：
epoch: 17
warp_epoch: 2
lr: 0.0001
lr_scheduler_step: [5,10]（edit阶段剩下那一半）
'''

@torch.no_grad()
def eval(render,dataloader,checkpoint=None,stage='warp'):
    '''返回结果，这里是loss值'''
    if checkpoint is not None and '' != checkpoint:
        state_dict=torch.load(checkpoint)
        render.load_state_dict(state_dict)

    render.eval()
    metrices=[]
    for data in tqdm(dataloader):
        # 把数据放到device中
        for key,value in data.items():
            data[key]=value.to(next(render.parameters()).device)
    
        # [B,73,win size]
        driving_source=data['driving_src']
        # [B,3,H,W]
        input_image=data['src']
        # [B,3,H,W]
        output_dict = render(input_image, driving_source)
        
        if stage=='warp':
            temp=output_dict['warp_image']
        else:
            temp=output_dict['fake_image']
        temp=eval_ssim(((temp.permute(0,2,3,1).cpu().numpy()+1)/2*255).astype(np.uint8),
                        ((data['target'].permute(0,2,3,1).cpu().numpy()+1)/2*255).astype(np.uint8))
        metrices.append(temp)

    render.train()

    return sum(metrices)/len(metrices)

@torch.no_grad()
def save_result(render,dataloader,save_dir,save_video_num):
    '''保存结果
    save_video_num指的是保存的视频个数'''
    os.makedirs(save_dir,exist_ok=True)

    render.eval()
    for it,data in enumerate(dataloader):
        if save_video_num==it:
            break
        # 把数据放到device中
        for key,value in data.items():
            data[key]=value.to(next(render.parameters()).device)
        data=deal_data(data)

        # [B,73,win size]
        driving_source=data['driving_src']
        # [B,3,H,W]
        input_image=data['src']

        # 渲染成面部图片

        # [B,3,H,W]
        output_dict = render(input_image, driving_source)
    
        # 输出为视频，顺序是（源图片，目标，warp，生成图片）
        # [len,H,W,3]
        real_video=data['target'].permute(0,2,3,1)
        # [len,H,W,3]
        img=data['src'].permute(0,2,3,1)
        # [len,H,W,3]
        warp=output_dict['warp_image'].permute(0,2,3,1)
        fake=output_dict['fake_image'].permute(0,2,3,1)
        # [len,H,4*W,3]
        video=torch.cat((img,real_video,warp,fake),dim=2)
        save_path=os.path.join(save_dir,'{}.mp4'.format(it))
        torchvision.io.write_video(save_path, ((video+1)/2*255).cpu(), fps=1)
    render.train()

    return


class FaceImgFormat():
    def format(self,pred_face,data):
        # 将data中的pose信息对齐
        pose_src = data['src_pose']
        temp_pose=data['tgt_pose'].permute(0,2,1).reshape(-1,9)
        pose=torch.cat((pose_src,temp_pose),dim=0).cpu().numpy()

        out_images=[]
        t=np.zeros((2,1))
        for i in range(len(pred_face)):
            t[0,0]=pose[i][-2]
            t[1,0]=pose[i][-1]
            s=pose[i][-3]
            out_img= self.image_transform(pred_face[i],s,t)
            out_images.append(out_img[None])
        return torch.cat(out_images, 0)

    def image_transform(self, images,s,t):
        img= self.align_img(images,s,t)        
        return img    


    # utils for face reconstruction
    def align_img(self,img, s,t,target_size=224.):
        """
        Return:
            transparams        --numpy.array  (raw_W, raw_H, scale, tx, ty)
            img_new            --PIL.Image  (target_size, target_size, 3)
            lm_new             --numpy.array  (68, 2), y direction is opposite to v direction
            mask_new           --PIL.Image  (target_size, target_size)
        
        Parameters:
            img                --PIL.Image  (raw_H, raw_W, 3)
            lm                 --numpy.array  (68, 2), y direction is opposite to v direction
            lm3D               --numpy.array  (5, 3)
            mask               --PIL.Image  (raw_H, raw_W, 3)
        """

        # processing the image
        img_new = self.resize_n_crop_img(img, t, s, target_size=target_size)

        return img_new

    # resize and crop images for face reconstruction
    def resize_n_crop_img(self,img, t, s, target_size=224.):
        w0, h0 = 256,256
        w = int(w0*s)
        h = int(h0*s)
        left = int(w/2 - target_size/2 + float((t[0][0] - w0/2)*s))
        right_left=max(0,left)
        right = int(left + target_size)
        right_right=min(w,right)
        up = int(h/2 - target_size/2 + float((h0/2 - t[1])*s))
        right_up=max(0,up)
        below = int(up + target_size)
        right_below=min(h,below)

        new_mask=torch.zeros((3,h,w)).to(img)
        new_mask[:,right_up:right_below,right_left:right_right]=\
            img[:,right_up-up:224-below+right_below,right_left-left:224-right+right_right]
        new_mask = F.interpolate(new_mask.reshape(1,3,h,w), size = (h0, w0), mode='bilinear')
        new_mask=new_mask.reshape(3,h0,w0)

        return new_mask


@torch.no_grad()
def deal_data(data):
    '''处理从dataset中获得的数据
    1. 将其变成3d人脸
    2. 将其变成drving_src'''

    pred_vertex, pred_color = face_model.compute_for_render_id_cross(data)
    _, _, pred_face = face_render(pred_vertex, face_model.face_buf, feat=pred_color)
    face_format=FaceImgFormat()
    # [12*b,3,256,256]
    pred_face=face_format.format(pred_face,data)
    src_face=pred_face[:len(data['src'])]
    # [b,11,3,256,256]
    drving_face=pred_face[len(data['src']):]
    drving_face=drving_face.reshape(len(data['src']),-1,3,256,256)
    data['src_face']= ((src_face*2)-1).clamp( min=-1, max=1)
    data['tgt_face']= ((drving_face*2)-1).clamp( min=-1, max=1)
    
    # key_dict={'src','target','tgt_pose','tgt_exp','src_inf'}
    driving_src=torch.cat((data['tgt_exp'],data['tgt_pose']),dim=1)
    data['driving_src']= driving_src

    # 剔除无用数据
    data.pop('src_inf')
    data.pop('src_pose')
    data.pop('tgt_pose')
    data.pop('tgt_exp')

    # # test测试获得的面部结构是否有效
    # # 顺序是（源图片，源3D人脸，目标图片，目标3D人脸）
    # import imageio
    # for i in range(len(data['src_face'])):
    #     src=data['src'][i]
    #     src_face=data['src_face'][i]
    #     tgt=data['target'][i]
    #     tgt_face=data['tgt_face'][i][5]
    #     img=torch.cat((src,src_face,tgt,tgt_face),dim=2)
    #     imageio.imsave(f'temp/{i}.png',((img+1)/2*255).permute(1,2,0).cpu().numpy().astype(np.uint8))

    return data


def run(config):
    # 创建一些文件夹
    os.makedirs(config['result_dir'],exist_ok=True)
    os.makedirs(config['checkpoint_dir'],exist_ok=True)

    # 加载device
    device = torch.device(config['device_id'][0] if torch.cuda.is_available() else "cpu")

    # logger
    # 记录日常事务的log（包括训练信息）
    train_logger = logger_config(log_path='train_render.log', logging_name='train_render')
    # 记录test结果的log
    test_logger = logger_config(log_path='test_render.log', logging_name='test_render')

    # 数据集loader
    # 训练集
    # 训练时，获得的数据大小是batch_size*frame_num，frame_num必须不小于2
    train_dataset=RenderDataset(config,type='train',frame_num=2)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=7, 
        shuffle=True,
        drop_last=False,
        num_workers=4,
        collate_fn=train_dataset.collater
    )     
    # 验证集
    eval_dataset=RenderDataset(config,type='eval',frame_num=2)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=5, 
        shuffle=True,
        drop_last=False,
        num_workers=2,
        collate_fn=eval_dataset.collater
    )     
    # 测试集
    test_dataset=RenderDataset(config,type='test',frame_num=2)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=5, 
        shuffle=True,
        drop_last=False,
        num_workers=2,
        collate_fn=test_dataset.collater
    )     


    #render model
    render=Render(config['mapping_net'],config['warpping_net'],
                         config['editing_net'],config['common'])
    render=render.to(device)
    render= torch.nn.DataParallel(render, device_ids=config['device_id'])
    if 'render_pre_train' in config:
        train_logger.info('render模块加载预训练模型{}'.format(config['render_pre_train']))
        state_dict=torch.load(config['render_pre_train'],map_location=torch.device('cpu'))
        render.load_state_dict(state_dict,strict=False)
        # # test
        # mapping_state_dict={}
        # for key,val in state_dict.items():
        #     mapping_state_dict[key.replace('module.mapping_net.','')]=val
        # render.module.mapping_net.load_state_dict(mapping_state_dict,strict=False)
        # warpping_state_dict={}
        # for key,val in state_dict.items():
        #     warpping_state_dict[key.replace('module.warpping_net.','')]=val
        # render.module.warpping_net.load_state_dict(warpping_state_dict,strict=False)
    

    # 人脸重建模型
    face_model.to(device)
    face_render.to(device)
    


    # 验证
    save_result(render,test_dataloader,os.path.join(config['result_dir'],'epoch_-1_warp'),save_video_num=10)

    # loss
    loss_function=RenderLoss(config)
    # 优化器
    optimizer = torch.optim.Adam(render.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=config['lr_scheduler_step']
                ,gamma=0.2,verbose=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,verbose=True,cooldown=0,
    #     patience=config['lr_scheduler_step'],mode='min', threshold_mode='rel', threshold=0, min_lr=1e-06)
    
    train_logger.info('准备完成，开始训练')
    # 以ssim为准
    metrices={'warp':-1,'edit':-1}
    best_checkpoint=None
    for epoch in range(config['epoch']):
        epoch_loss=0
        stage='warp' if epoch<config['warp_epoch'] else 'edit'
        for data in tqdm(train_dataloader):
            # 把数据放到device中
            for key,value in data.items():
                data[key]=value.to(device)
            
            # [B,73,win size]
            driving_source=data['driving_src']
            # [B,3,H,W]
            input_image=data['src']
            # [B,3,H,W]
            output_dict = render(input_image, driving_source)

            if stage=='warp':
                # 计算loss
                loss=loss_function(output_dict['warp_image'],data,stage='edit')
            else:
                loss=loss_function(output_dict['warp_image'],data,stage='warp')
                loss+=loss_function(output_dict['fake_image'],data)

            epoch_loss+=loss.item()
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_logger.info(f'第{epoch}次迭代获得的loss值为{epoch_loss}')
        train_logger.info('当前学习率为{}'.format(optimizer.param_groups[0]['lr']))
        scheduler.step()
        eval_metrices=eval(render,eval_dataloader,checkpoint=None,stage=stage)
        test_logger.info(f'第{epoch}次迭代后，验证集的指标值为{eval_metrices}')
        if eval_metrices>metrices[stage]:
            save_path=os.path.join(config['result_dir'],'epoch_{}_{}'.format(epoch,stage))
            save_result(render,eval_dataloader,save_path,save_video_num=10)
            pth_path= os.path.join(config['checkpoint_dir'],f'{stage}_epoch_{epoch}_metrices_{eval_metrices}.pth')
            torch.save(render.state_dict(),pth_path)
            metrices[stage]=eval_metrices
            best_checkpoint=pth_path
    
    # 测试模型
    test_logger.info(f'模型训练完毕，加载最好的模型{best_checkpoint}进行测试')
    test_metrices=eval(render,test_dataloader,checkpoint=best_checkpoint,stage='edit')
    test_logger.info(f'测试结果为{test_metrices}')
    save_path=os.path.join(config['result_dir'],'test')
    save_result(render,test_dataloader,save_path,save_video_num=10)


if __name__ == '__main__':
    import os,sys
    import yaml

    config={}
    yaml_file=['config/dataset/common.yaml','config/model/render.yaml']
    for a in yaml_file:
        with open(a,'r',encoding='utf8') as f:
            config.update(yaml.safe_load(f))


    # reader = imageio.get_reader('data/001.mp4')
    # driving_video = []
    # try:
    #     # 信息表示为整型
    #     driving_video=[im for im in reader]
    # except RuntimeError:
    #     pass
    # reader.close()
    # # driving_video=np.array(driving_video)[:,:520,:,:]
    # from skimage.transform import resize
    # driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    # imageio.mimsave('tmp.mp4',driving_video)
    
    # 3dmm人脸重建模块
    face_model=ParametricFaceModel(
                bfm_folder='BFM', camera_distance=10.0, focal=1015.0, center=112.0,
                is_train=False, default_name='BFM_model_front.mat'
            )
    face_render=MeshRenderer(
                rasterize_fov=12.59363743796881, znear=5.0, zfar=15.0, 
                rasterize_size=224, use_opengl=False
            )


    run(config)

    # # 测试scheduler
    # model=Exp3DMM(config)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,verbose=True,cooldown=0,
    #     patience=config['lr_scheduler_step'],mode='min', threshold_mode='rel', threshold=0, min_lr=5e-05)
    # loss=999
    # while True:
    #     loss+=1
    #     scheduler.step(loss)