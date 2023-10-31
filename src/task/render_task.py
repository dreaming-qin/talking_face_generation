import os
import torch
from tqdm import tqdm
import imageio
import uuid
import numpy as np
from skimage import img_as_ubyte

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
from src.model.exp3DMM.exp3DMM import Exp3DMM
from src.model.render.render import FaceGenerator
from src.loss.renderLoss import RenderLoss
from src.util.model_util import freeze_params
from src.util.util import get_window


# 训练时，固定住exp3DMM模块，然后进行训练



@torch.no_grad()
def eval(exp_3dmm_model,render,dataloader,loss_fun,checkpoint=None):
    '''返回结果，这里是loss值
    save_video_num指的是在该过程中保存多少个视频'''
    if checkpoint is not None:
        state_dict=torch.load(checkpoint)
        render.load_state_dict(state_dict)

    exp_3dmm_model.eval()
    render.eval()
    total_loss=0
    for data in tqdm(dataloader):
        # 把数据放到device中
        for key,value in data.items():
            data[key]=value.to(next(render.parameters()).device)
        transformer_video=data['video_input']
        transformer_video=transformer_video.permute(0,1,4,2,3)
        # 先获得exp 3dmm
        exp_3dmm=exp_3dmm_model(transformer_video,data['audio_input'])

        # 然后处理数据，方便作为render输入
        # [B,len,73]
        driving_source=torch.cat((exp_3dmm,data['pose']),dim=-1)
        # [B,len,win size,73]
        driving_source=get_window(driving_source,config['win_size'])
        driving_source=driving_source.permute(0,1,3,2)
        input_image=[]
        for img in data['img']:
            # [3,H,W]
            tmp_img=img.permute(2,0,1)
            # [len,3,H,W]
            tmp_img=tmp_img.expand(driving_source.shape[1],  -1, -1, -1)
            input_image.append(tmp_img)
        # [B,len,3,H,W]
        input_image=torch.stack(input_image)


        output_imgs=[]
        for drving,img in zip(driving_source,input_image):
            output_dict = render(img, drving)
            output_imgs.append(output_dict['fake_image'])
        # [B,len,3,H,W]
        output_imgs=torch.stack(output_imgs)

        # 计算loss
        loss=loss_fun(output_imgs.permute(0,1,3,4,2),data)
        total_loss+=loss.item()
    render.train()

    return total_loss

@torch.no_grad()
def save_result(exp_3dmm_model,render,dataloader,save_dir,save_video_num):
    '''保存结果
    save_video_num指的是保存的视频个数'''
    os.makedirs(save_dir,exist_ok=True)

    exp_3dmm_model.eval()
    render.eval()
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
        # [B,len,win size,73]
        driving_source=get_window(driving_source,config['win_size'])
        driving_source=driving_source.permute(0,1,3,2)
        input_image=[]
        for img in data['img']:
            # [3,H,W]
            tmp_img=img.permute(2,0,1)
            # [len,3,H,W]
            tmp_img=tmp_img.expand(driving_source.shape[1],  -1, -1, -1)
            input_image.append(tmp_img)
        # [B,len,3,H,W]
        input_image=torch.stack(input_image)


        output_imgs=[]
        for drving,img in zip(driving_source,input_image):
            output_dict = render(img, drving)
            output_imgs.append(output_dict['fake_image'])
        # [B,len,3,H,W]
        output_imgs=torch.stack(output_imgs)

        # 输出为视频
        for real_video,video_input,img,fake_video in zip(data['raw_video'],data['video_input'],data['img'],output_imgs):
            # [len,H,W,3]
            real_video=real_video.detach().cpu().numpy()
            # [len,H,W,3]
            video_input=video_input.detach().cpu().numpy()
            # [len,H,W,3]
            fake_video=fake_video.permute(0,2,3,1)
            fake_video=fake_video.detach().cpu().numpy()
            # [len,H,W,3]
            img=img.expand(fake_video.shape[0],  -1, -1, -1)
            img=img.detach().cpu().numpy()
            save_path=os.path.join(save_dir,'{}.mp4'.format(str(uuid.uuid1())))
            # [len,H,4*W,3]
            video=np.concatenate((video_input,img,real_video,fake_video),axis=2)
            imageio.mimsave(save_path,(video*255.0).astype('uint8') )

    render.train()
    return


def run(config):
    # 创建一些文件夹
    os.makedirs(config['result_dir'],exist_ok=True)
    os.makedirs(config['checkpoint_dir'],exist_ok=True)

    # 加载device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # logger
    # 记录日常事务的log（包括训练信息）
    train_logger = logger_config(log_path='train_render.log', logging_name='train_render')
    # 记录test结果的log
    test_logger = logger_config(log_path='test_render.log', logging_name='test_render')

    # 数据集loader
    # 训练集
    # 训练时，gpu显存不够，因此设定训练集的最大长度
    train_dataset=RenderDataset(config,type='train',max_len=2)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1, 
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=train_dataset.collater
    )     
    # 验证集
    eval_dataset=RenderDataset(config,type='eval',max_len=20)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1, 
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=eval_dataset.collater
    )     
    # 测试集
    test_dataset=RenderDataset(config,type='test',max_len=20)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
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
    train_logger.info('exp 3dmm 模块加载预训练模型{}'.format(config['exp_3dmm_pre_train']))
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
    if 'render_pre_train' in config:
        train_logger.info('render模块加载预训练模型{}'.format(config['render_pre_train']))
        state_dict=torch.load(config['render_pre_train'])
        render.load_state_dict(state_dict)

    # loss
    loss_function=RenderLoss(config,device)

    # 优化器
    optimizer = torch.optim.Adam(render.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,verbose=True,cooldown=0,
        patience=config['lr_scheduler_step'],mode='min', threshold_mode='rel', threshold=0, min_lr=1e-05)


    # 其它的一些参数
    best_loss=int(1e10)
    best_checkpoint=0

    train_logger.info('准备完成，开始训练')
    # 开始训练
    for epoch in range(config['epoch']):
        epoch_loss=0
        for data in tqdm(train_dataloader):
            # 把数据放到device中
            for key,value in data.items():
                data[key]=value.to(device)
            transformer_video=data['video_input']
            transformer_video=transformer_video.permute(0,1,4,2,3)
            # 先获得exp 3dmm
            exp_3dmm=exp_model(transformer_video,data['audio_input'])

            # 然后处理数据，方便作为render输入
            # [B,len,73]
            driving_source=torch.cat((exp_3dmm,data['pose']),dim=-1)
            # [B,len,win size,73]
            driving_source=get_window(driving_source,config['win_size'])
            driving_source=driving_source.permute(0,1,3,2)
            input_image=[]
            for img in data['img']:
                # [3,H,W]
                tmp_img=img.permute(2,0,1)
                # [len,3,H,W]
                tmp_img=tmp_img.expand(driving_source.shape[1],  -1, -1, -1)
                input_image.append(tmp_img)
            # [B,len,3,H,W]
            input_image=torch.stack(input_image)

            output_imgs=[]
            for drving,img in zip(driving_source,input_image):
                output_dict = render(img, drving)
                # output_dict['fake_image'] [len,3,H,W]
                output_imgs.append(output_dict['fake_image'])
            # [B,len,3,H,W]
            output_imgs=torch.stack(output_imgs)

            # 计算loss
            loss=loss_function(output_imgs.permute(0,1,3,4,2),data)
            epoch_loss+=loss.item()
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(render.parameters(), 0.5)
            optimizer.step()
        train_logger.info(f'第{epoch}次迭代获得的loss值为{epoch_loss}')
        eval_loss=eval(exp_model,render,eval_dataloader,loss_function,checkpoint=None)
        train_logger.info(f'对模型进行验证，验证获得的结果为{eval_loss}')
        # 如果验证结果好，保存训练模型
        if eval_loss<best_loss:
            best_loss=eval_loss
            pth_path= os.path.join(config['checkpoint_dir'],f'epoch_{epoch}_loss_{best_loss}.pth')
            best_checkpoint=pth_path
            torch.save(render.state_dict(),pth_path)
            train_logger.info(f'第{epoch}次结果较好，获得的loss为{best_loss},已将checkoint文件保存至{pth_path}')
            test_logger.info(f'第{epoch}次结果较好，获得的loss为{best_loss},已将checkoint文件保存至{pth_path}')
            save_result(exp_model,render,eval_dataloader,
                        os.path.join(config['result_dir'],'epoch_{}'.format(epoch)),
                        save_video_num=10)
        # 根据验证结果，调节学习率
        scheduler.step(best_loss)
        train_logger.info('当前学习率为{}'.format(optimizer.param_groups[0]['lr']))


    
    # 测试模型
    test_logger.info(f'模型训练完毕，加载最好的模型{best_checkpoint}进行测试')
    test_loss=eval(exp_model,render,test_dataloader,loss_function,checkpoint=best_checkpoint)
    test_logger.info(f'测试结果为{test_loss}')
    save_result(exp_model,render,eval_dataloader,
                os.path.join(config['result_dir'],'test'),
                save_video_num=10)


if __name__ == '__main__':
    import os,sys
    import yaml

    config={}
    yaml_file=['config/data_process/common.yaml','config/dataset/common.yaml','config/model/render.yaml']
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