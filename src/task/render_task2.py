import os
import torch
from tqdm import tqdm
import torchvision
import numpy as np
import yaml
from argparse import Namespace

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
from src.model.render2.render import StyleGAN2Generator as Render
from src.loss.renderLoss import RenderLoss

from src.metrics.SSIM import ssim as eval_ssim

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
        fake_img,_ = render(input_image, driving_source)
        
        temp=eval_ssim(((fake_img.permute(0,2,3,1).cpu().numpy()+1)/2*255).astype(np.uint8),
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

        # [B,73,win size]
        driving_source=data['driving_src']
        # [B,3,H,W]
        input_image=data['src']
        # [B,3,H,W]
        fake_img,_ = render(input_image, driving_source)
    
        # 输出为视频，顺序是（源图片，目标，生成图片）
        # [len,H,W,3]
        real_video=data['target'].permute(0,2,3,1)
        # [len,H,W,3]
        img=data['src'].permute(0,2,3,1)
        # [len,H,W,3]
        fake=fake_img.permute(0,2,3,1)
        # [len,H,4*W,3]
        video=torch.cat((img,real_video,fake),dim=2)
        save_path=os.path.join(save_dir,'{}.mp4'.format(it))
        torchvision.io.write_video(save_path, ((video+1)/2*255).cpu(), fps=1)
    render.train()

    return


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
        batch_size=9, 
        shuffle=True,
        drop_last=False,
        num_workers=5,
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

    
    # 指定 YAML 文件路径
    yaml_file_path = 'test.yaml'
    # 读取 YAML 文件并加载为字典
    with open(yaml_file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    # 将字典转换为 Namespace 对象
    opt = Namespace(**yaml_data)
    style_dim=1024
    config['mapping_net']['descriptor_nc']=style_dim
    #render model
    render=Render(opt,config['mapping_net'],style_dim=style_dim)
    render=render.to(device)
    render= torch.nn.DataParallel(render, device_ids=config['device_id'])
    if 'render_pre_train' in config:
        train_logger.info('render模块加载预训练模型{}'.format(config['render_pre_train']))
        state_dict=torch.load(config['render_pre_train'],map_location=torch.device('cpu'))
        render.load_state_dict(state_dict,strict=False)

    # 验证
    save_result(render,test_dataloader,os.path.join(config['result_dir'],'epoch_-1'),save_video_num=10)

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
    metrices=-1
    best_checkpoint=None
    for epoch in range(config['epoch']):
        epoch_loss=0
        for data in tqdm(train_dataloader):
            # 把数据放到device中
            for key,value in data.items():
                data[key]=value.to(device)
            
            # [B,73,win size]
            driving_source=data['driving_src']
            # [B,3,H,W]
            input_image=data['src']
            # [B,3,H,W]
            fake_img,_ = render(input_image, driving_source)

            loss=loss_function(fake_img,data)

            epoch_loss+=loss.item()
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(render.parameters(), 0.5)
            optimizer.step()

        train_logger.info(f'第{epoch}次迭代获得的loss值为{epoch_loss}')
        train_logger.info('当前学习率为{}'.format(optimizer.param_groups[0]['lr']))
        scheduler.step()
        eval_metrices=eval(render,eval_dataloader,checkpoint=None)
        test_logger.info(f'第{epoch}次迭代后，验证集的指标值为{eval_metrices}')
        if eval_metrices>metrices:
            save_path=os.path.join(config['result_dir'],'epoch_{}'.format(epoch))
            save_result(render,eval_dataloader,save_path,save_video_num=10)
            pth_path= os.path.join(config['checkpoint_dir'],f'epoch_{epoch}_metrices_{eval_metrices}.pth')
            torch.save(render.state_dict(),pth_path)
            metrices=eval_metrices
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