import os
import torch
from tqdm import tqdm
import imageio
import numpy as np
import torchvision
import time


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
from src.dataset.exp3DMMdataset import Exp3DMMdataset
from src.model.exp3DMM.exp3DMM import Exp3DMM
from src.model.render.render import Render
from src.loss.exp3DMMLoss import Exp3DMMLoss
from src.util.model_util import freeze_params
from src.metrics.SSIM import ssim as eval_ssim

'''
经过训练，最好的训练参数是：
4090版本的
train_dataset: frame_num=1 workers=6 batch_size=6

train_dataset: frame_num=1 workers=1 batch_size=6
test_dataset: frame_num=1 workers=1 batch_size=5
eval_dataset: frame_num=1 workers=1 batch_size=5
optimizer: betas=(0.5, 0.999)
scheduler: gamma=0.2


lr_scheduler_step: [10,17]
epoch: 25
lr: 0.0001

landmark_weight: 1
exp_weight: 4
sync_weight: 1
rec_weight: 15
rec_low_weight: [1,0]

使用预训练的模型的话，最好的参数是：
lr_scheduler_step: [10,16]
epoch: 20
lr: 0.0001


'''


@torch.no_grad()
def eval(exp_model,render,dataloader,checkpoint=None):
    '''返回结果，这里是loss值'''
    if checkpoint is not None and '' != checkpoint:
        exp_model.load_state_dict(torch.load(checkpoint))

    exp_model.eval()
    ans=[]
    for data in tqdm(dataloader):
        # 把数据放到device中
        for key,value in data.items():
            data[key]=value.to(next(render.parameters()).device)

        exp_result=exp_model(data['video'].permute(0,1,4,2,3),data['audio'])
        drving_src=torch.cat((exp_result,data['pose']),dim=2).permute(0,2,1)
        imgs = render(data['img'],drving_src )['fake_image']

        temp=eval_ssim(((imgs.permute(0,2,3,1).cpu().numpy()+1)/2*255).astype(np.uint8),
                        ((data['gt_video'].permute(0,2,3,1).cpu().numpy()+1)/2*255).astype(np.uint8))
        ans.append(temp)

    exp_model.train()
    
    return sum(ans)/len(ans)


@torch.no_grad()
def save_result(exp_model,render,dataloader,save_dir,save_video_num=1):
    '''保存结果
    save_video_num指的是保存的视频个数'''
    os.makedirs(save_dir,exist_ok=True)

    exp_model.eval()
    for it,data in enumerate(dataloader):
        if save_video_num==it:
            break
        # 把数据放到device中
        for key,value in data.items():
            data[key]=value.to(next(render.parameters()).device)
        
        exp_result=exp_model(data['video'].permute(0,1,4,2,3),data['audio'])
        drving_src=torch.cat((exp_result,data['pose']),dim=2).permute(0,2,1)
        imgs = render(data['img'],drving_src )['fake_image']
    
        # 输出为视频，顺序是（输入视频，源图片，目标，生成图片）
        input_video=data[f'video'][:,18]
        real_video= data['gt_video']
        # [len,H,W,3]
        real_video=real_video.reshape(-1,3,256,256).permute(0,2,3,1)
        # [len,H,W,3]
        src_imgs= data['img']
        src_imgs=src_imgs.permute(0,2,3,1)
        fake=imgs
        fake=fake.reshape(-1,3,256,256).permute(0,2,3,1)
        # [len,H,3*W,3]
        video=torch.cat((input_video,src_imgs,real_video,fake),dim=2)
        save_path=os.path.join(save_dir,'{}.mp4'.format(it))
        torchvision.io.write_video(save_path, ((video+1)/2*255).cpu(), fps=1)
    exp_model.train()

    return


def run(config):
    # 创建一些文件夹
    os.makedirs(config['result_dir'],exist_ok=True)
    os.makedirs(config['checkpoint_dir'],exist_ok=True)

    # 加载device
    device = torch.device(config['device_id'][0] if torch.cuda.is_available() else "cpu")

    # logger
    # 记录日常事务的log（包括训练信息）
    train_logger = logger_config(log_path='train_exp3DMM.log', logging_name='train_exp3DMM')
    # 记录test结果的log
    test_logger = logger_config(log_path='test_exp3DMM.log', logging_name='test_exp3DMM')

    # 训练集
    # 训练时，获得的数据大小是batch_size*frame_num*3, 3是因为三元对loss
    train_dataset=Exp3DMMdataset(config,type='train',frame_num=1)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=7, 
        shuffle=True,
        drop_last=False,
        num_workers=6,
        collate_fn=train_dataset.collater
    )
    # 验证集
    eval_dataset=Exp3DMMdataset(config,type='eval',frame_num=1)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=5, 
        shuffle=True,
        drop_last=False,
        num_workers=1,
        collate_fn=eval_dataset.collater
    )     
    # 测试集
    test_dataset=Exp3DMMdataset(config,type='test',frame_num=1)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=5, 
        shuffle=True,
        drop_last=False,
        num_workers=1,
        collate_fn=test_dataset.collater
    )     

    # render model
    render=Render(config['mapping_net'],config['warpping_net'],
                         config['editing_net'],config['common'])
    render=render.to(device)
    render= torch.nn.DataParallel(render, device_ids=config['device_id'])
    # render_pre_train必须要有
    train_logger.info('render模块加载预训练模型{}'.format(config['render_pre_train']))
    state_dict=torch.load(config['render_pre_train'],map_location=torch.device('cpu'))
    render.load_state_dict(state_dict)
    freeze_params(render)
    render.eval()

    # 3dmm model
    exp_model=Exp3DMM(config)
    exp_model=exp_model.to(device)
    exp_model= torch.nn.DataParallel(exp_model, device_ids=config['device_id'])
    if 'exp_3dmm_pre_train' in config:
        train_logger.info('exp 3dmm模块加载预训练模型{}'.format(config['exp_3dmm_pre_train']))
        state_dict=torch.load(config['exp_3dmm_pre_train'],map_location=torch.device('cpu'))
        exp_model.load_state_dict(state_dict,strict=False)

    
    # 验证
    save_path=os.path.join(config['result_dir'],'epoch_-1')
    save_result(exp_model,render,test_dataloader,save_path,save_video_num=3)

    # loss
    loss_function=Exp3DMMLoss(config,device)

    # 优化器
    optimizer = torch.optim.Adam(exp_model.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=config['lr_scheduler_step']
                ,gamma=0.2,verbose=True)


    # 其它的一些参数
    best_metrices=-1
    best_checkpoint=None
    
    train_logger.info('准备完成，开始训练')
    # 开始训练
    for epoch in range(config['epoch']):
        epoch_loss=0
        for data in tqdm(train_dataloader):
            # 把数据放到device中
            for key,value in data.items():
                data[key]=value.to(device)

            # test
            # imageio.mimsave('test.mp4',data['video'][0].cpu().numpy())

            exp_result=exp_model(data['video'].permute(0,1,4,2,3),data['audio'])
            drving_src=torch.cat((exp_result,data['pose']),dim=2).permute(0,2,1)
            imgs = render(data['img'],drving_src )['fake_image']

            # 计算loss
            loss=loss_function(exp_result,imgs,data)

            epoch_loss+=loss.item()
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(exp_model.parameters(), 0.5)
            optimizer.step()
            # print(f'训练用时间{time.time()-s}')

        train_logger.info(f'第{epoch}次迭代获得的loss值为{epoch_loss}，平均值是{epoch_loss/len(train_dataloader)}')
        eval_metrices=eval(exp_model,render,eval_dataloader,checkpoint=None)
        test_logger.info(f'第{epoch}次后，对模型进行验证，验证获得的结果为{eval_metrices}')
        # 如果验证结果好，保存训练模型
        if eval_metrices>best_metrices:
        # if True:
            save_path=os.path.join(config['result_dir'],'epoch_{}'.format(epoch))
            save_result(exp_model,render,eval_dataloader,save_path,save_video_num=3)
            best_metrices=eval_metrices
            pth_path= os.path.join(config['checkpoint_dir'],f'epoch_{epoch}_metrices_{best_metrices}.pth')
            best_checkpoint=pth_path
            torch.save(exp_model.state_dict(),pth_path)
        # 根据验证结果，调节学习率
        scheduler.step()
        train_logger.info('当前学习率为{}'.format(optimizer.param_groups[0]['lr']))

    
    # 测试模型
    test_logger.info(f'模型训练完毕，加载最好的模型{best_checkpoint}进行测试')
    test_loss=eval(exp_model,render,test_dataloader,checkpoint=best_checkpoint)
    test_logger.info(f'测试结果为{test_loss}')
    save_path=os.path.join(config['result_dir'],'test')
    save_result(exp_model,render,test_dataloader,save_path,save_video_num=3)



if __name__ == '__main__':
    import os,sys
    import yaml,glob

    config={}
    yaml_file=['config/dataset/common.yaml','config/model/render.yaml'
               ,'config/model/exp3DMM.yaml']
    for a in yaml_file:
        with open(a,'r',encoding='utf8') as f:
            config.update(yaml.safe_load(f))

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