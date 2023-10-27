import os
import time
import numpy as np
import torch

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
from src.loss.exp3DMMLoss import Exp3DMMLoss


@torch.no_grad()
def eval(model,dataloader,loss_fun,checkpoint=None):
    '''返回结果，这里是loss值'''
    if checkpoint is not None:
        state_dict=torch.load(checkpoint)
        model.load_state_dict(state_dict)

    total_loss=0
    for data in dataloader:
        # 把数据放到device中
        for key,value in data.items():
            data[key]=value.to(next(model.parameters()).device)
        transformer_video=data['video_input']
        transformer_video=transformer_video.permute(0,1,4,2,3)
        exp_3dmm=model(transformer_video,data['audio_input'])
        # 计算loss
        loss=loss_fun(exp_3dmm,data)
        total_loss+=loss.item()
    return total_loss

        
def run(config):
    # 创建一些文件夹
    os.makedirs(config['result_dir'],exist_ok=True)
    os.makedirs(config['checkpoint_dir'],exist_ok=True)

    # 加载device
    device = torch.device(config['device'])

    # logger
    # 记录日常事务的log（包括训练信息）
    train_logger = logger_config(log_path='train_exp3DMM.log', logging_name='train_exp3DMM')
    # 记录test结果的log
    test_logger = logger_config(log_path=os.path.join(config['result_dir'],'test_exp3DMM.log'), logging_name='test_exp3DMM')

    # 数据集loader
    # 训练集
    # 训练时，gpu显存不够，因此设定训练集的最大长度
    train_dataset=Exp3DMMdataset(config,type='train',max_len=4)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1, 
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=train_dataset.collater
    )     
    # 验证集
    eval_dataset=Exp3DMMdataset(config,type='test')
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=128, 
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=eval_dataset.collater
    )     
    # 测试集
    test_dataset=Exp3DMMdataset(config,type='test')
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128, 
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=test_dataset.collater
    )     

    # model
    model=Exp3DMM(config)
    model=model.to(device)

    # loss
    loss_function=Exp3DMMLoss(config,device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.95,verbose=True,cooldown=0,
        patience=config['lr_scheduler_step'],mode='min', threshold_mode='rel', threshold=0, min_lr=5e-05)


    # 其它的一些参数
    best_loss=int(1e10)
    best_checkpoint=0

    train_logger.info('准备完成，开始训练')
    # 开始训练
    for epoch in range(config['epoch']):
        epoch_loss=0
        for data in train_dataloader:
            # 把数据放到device中
            for key,value in data.items():
                data[key]=value.to(device)
            transformer_video=data['video_input']
            transformer_video=transformer_video.permute(0,1,4,2,3)
            exp_3dmm=model(transformer_video,data['audio_input'])
            # 计算loss
            loss=loss_function(exp_3dmm,data)
            epoch_loss+=loss.item()
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        train_logger.info(f'第{epoch}次迭代获得的loss值为{epoch_loss}')
        scheduler.step(epoch_loss)
        train_logger.info('开始验证...')
        eval_loss=eval(model,eval_dataloader,loss_function,checkpoint=None)
        train_logger.info(f'验证获得的结果为{eval_loss}')
        # 如果验证结果好，保存训练模型
        if eval_loss<best_loss:
            best_loss=eval_loss
            pth_path= os.path.join(config['checkpoint_dir'],f'epoch_{epoch}_loss_{best_loss}.pth')
            best_checkpoint=pth_path
            torch.save(model.state_dict(),pth_path)
            test_logger.info(f'第{epoch}次结果较好，获得的loss为{best_loss},已将checkoint文件保存至{pth_path}')

    
    # 测试模型
    test_logger.info(f'模型训练完毕，加载最好的模型{best_checkpoint}进行测试')
    test_loss=eval(model,test_dataloader,loss_function,checkpoint=best_checkpoint)
    test_logger.info(f'测试结果为{test_loss}')


if __name__ == '__main__':
    import os,sys
    import yaml,glob

    config={}
    yaml_file=glob.glob(r'config/*/*.yaml')
    for a in yaml_file:
        with open(a,'r',encoding='utf8') as f:
            config.update(yaml.safe_load(f))

    run(config)

    # # 测试scheduler
    # model=Exp3DMM(config)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.95,verbose=True,cooldown=0,
    #     patience=config['lr_scheduler_step'],mode='min', threshold_mode='rel', threshold=0, min_lr=5e-05)
    # loss=999
    # while True:
    #     loss+=1
    #     scheduler.step(loss)