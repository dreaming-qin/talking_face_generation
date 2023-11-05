import os
import torch
from tqdm import tqdm
import torchvision

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
from src.model.render.render import FaceGenerator
from src.loss.renderLoss import RenderLoss


# 训练时，固定住exp3DMM模块，然后进行训练



@torch.no_grad()
def eval(render,dataloader,loss_fun,checkpoint=None,stage='warp'):
    '''返回结果，这里是loss值'''
    if checkpoint is not None:
        state_dict=torch.load(checkpoint)
        render.load_state_dict(state_dict)

    render.eval()
    total_loss=0
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
            loss=loss_fun(output_dict['warp_image'],data,stage='warp')
        else:
            loss=loss_fun(output_dict['warp_image'],data,stage='warp')
            loss+=loss_fun(output_dict['fake_image'],data)

        # 计算loss
        total_loss+=loss.item()
    
    render.train()

    return total_loss

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
        output_dict = render(input_image, driving_source)
    
        # 输出为视频
        # [len,H,W,3]
        real_video=data['target'].permute(0,2,3,1)
        # [len,H,W,3]
        warp=output_dict['warp_image'].permute(0,2,3,1)
        # [len,H,W,3]
        img=data['src'].permute(0,2,3,1)
        fake=output_dict['fake_image'].permute(0,2,3,1)
        # [len,H,4*W,3]
        video=torch.concatenate((real_video,img,warp,fake),dim=2)
        save_path=os.path.join(save_dir,'{}.mp4'.format(it))
        torchvision.io.write_video(save_path, ((video+1)/2*255).cpu(), fps=5)
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
    # 训练时，获得的数据大小是batch_size*frame_num，frame_num必须不小于2
    train_dataset=RenderDataset(config,type='train',frame_num=2)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1, 
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=train_dataset.collater
    )     
    # 验证集
    eval_dataset=RenderDataset(config,type='eval',frame_num=2)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=5, 
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=eval_dataset.collater
    )     
    # 测试集
    test_dataset=RenderDataset(config,type='test',frame_num=2)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=5, 
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=test_dataset.collater
    )     


    #render model
    render=FaceGenerator(config['mapping_net'],config['warpping_net'],
                         config['editing_net'],config['common'])
    render=render.to(device)
    render= torch.nn.DataParallel(render, device_ids=config['device_id'])
    if 'render_pre_train' in config:
        train_logger.info('render模块加载预训练模型{}'.format(config['render_pre_train']))
        state_dict=torch.load(config['render_pre_train'])
        render.load_state_dict(state_dict)

    # test
    save_result(render,eval_dataloader,os.path.join(config['result_dir'],'epoch_-1_warp'),save_video_num=3)

    # loss
    loss_function=RenderLoss(config)
    # 优化器
    optimizer = torch.optim.Adam(render.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=config['lr_scheduler_step']
                ,gamma=0.2,verbose=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,verbose=True,cooldown=0,
    #     patience=config['lr_scheduler_step'],mode='min', threshold_mode='rel', threshold=0, min_lr=1e-06)
    
    train_logger.info('准备完成，开始训练')
    best_loss=int(1e10)
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
                loss=loss_function(output_dict['warp_image'],data,stage='warp')
            else:
                loss=loss_function(output_dict['warp_image'],data,stage='warp')
                loss+=loss_function(output_dict['fake_image'],data)

            epoch_loss+=loss.item()
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(render.parameters(), 0.5)
            optimizer.step()

        train_logger.info(f'第{epoch}次迭代获得的loss值为{epoch_loss}')
        train_logger.info('当前学习率为{}'.format(optimizer.param_groups[0]['lr']))
        scheduler.step()
        eval_loss=eval(render,eval_dataloader,loss_function,checkpoint=None,stage=stage)
        if eval_loss<best_loss:
            save_path=os.path.join(config['result_dir'],'epoch_{}_{}'.format(epoch,stage))
            save_result(render,eval_dataloader,save_path,save_video_num=3)
            pth_path= os.path.join(config['checkpoint_dir'],f'{stage}_epoch_{epoch}_loss_{epoch_loss}.pth')
            torch.save(render.state_dict(),pth_path)
            best_loss=epoch_loss
            best_checkpoint=pth_path
    
    # 测试模型
    test_logger.info(f'模型训练完毕，加载最好的模型{best_checkpoint}进行测试')
    test_loss=eval(render,test_dataloader,loss_function,checkpoint=best_checkpoint,stage='edit')
    test_logger.info(f'测试结果为{test_loss}')
    save_path=os.path.join(config['result_dir'],'test')
    save_result(render,eval_dataloader,save_path,save_video_num=3)


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