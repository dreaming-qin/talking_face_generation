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
from src.dataset.sync_net_dataset import SyncNetDataset
from src.model.syncNet.sync_net import SyncNet
from src.loss.sync_net_loss import SyncNetLoss
from src.util.model_util import freeze_params




@torch.no_grad()
def eval(syncnet,loss_function,dataloader,checkpoint=None):
    '''返回结果，这里是loss值'''
    if checkpoint is not None and '' != checkpoint:
        syncnet.load_state_dict(torch.load(checkpoint))

    syncnet.eval()
    ans=[]
    for data in tqdm(dataloader):
        # 把数据放到device中
        for key,value in data.items():
            data[key]=value.to(next(syncnet.parameters()).device)

        audio_e,mouth_e=syncnet(data['hubert'],data['mouth_landmark'])
        temp=loss_function(audio_e,mouth_e,data['label'])
        ans.append(temp)

    syncnet.train()
    
    return sum(ans)/len(ans)



def run(config):
    # 创建一些文件夹
    os.makedirs(config['checkpoint_dir'],exist_ok=True)

    # 加载device
    device = torch.device(config['device_id'][0] if torch.cuda.is_available() else "cpu")

    # logger
    # 记录日常事务的log（包括训练信息）
    train_logger = logger_config(log_path='train_sync_net.log', logging_name='train_sync_net')
    # 记录test结果的log
    test_logger = logger_config(log_path='test_sync_net.log', logging_name='test_sync_net')

    # 训练集
    # 训练时，获得的数据大小是batch_size*frame_num*3, 3是因为三元对loss
    train_dataset=SyncNetDataset(config,type='train',sample_per_video=20)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=3, 
        shuffle=True,
        drop_last=False,
        num_workers=1,
        collate_fn=train_dataset.collater
    )     
    # 验证集
    eval_dataset=SyncNetDataset(config,type='eval',sample_per_video=20)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=5, 
        shuffle=True,
        drop_last=False,
        num_workers=1,
        collate_fn=eval_dataset.collater
    )     
    # 测试集
    test_dataset=SyncNetDataset(config,type='test',sample_per_video=20)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=5, 
        shuffle=True,
        drop_last=False,
        num_workers=1,
        collate_fn=test_dataset.collater
    )     


    # syncnet model
    syncnet=SyncNet(**config)
    syncnet=syncnet.to(device)
    syncnet= torch.nn.DataParallel(syncnet, device_ids=config['device_id'])
    if 'sync_net_pre_train' in config:
        train_logger.info('syncNet模块加载预训练模型{}'.format(config['sync_net_pre_train']))
        state_dict=torch.load(config['sync_net_pre_train'],map_location=torch.device('cpu'))
        syncnet.load_state_dict(state_dict)

    # loss
    loss_function=SyncNetLoss()

    # 优化器
    optimizer = torch.optim.Adam(syncnet.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=config['lr_scheduler_step']
                ,gamma=0.2,verbose=True)


    # 其它的一些参数
    best_metrices=1e4
    best_checkpoint=None

    train_logger.info('准备完成，开始训练')
    # 开始训练
    for epoch in range(config['epoch']):
        epoch_loss=0
        for data in tqdm(train_dataloader):
            # 把数据放到device中
            for key,value in data.items():
                data[key]=value.to(device)

            audio_e,mouth_e=syncnet(data['hubert'],data['mouth_landmark'])
            loss=loss_function(audio_e,mouth_e,data['label'])

            epoch_loss+=loss.item()
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(syncnet.parameters(), 0.5)
            optimizer.step()

        train_logger.info(f'第{epoch}次迭代获得的loss值为{epoch_loss}')
        eval_metrices=eval(syncnet,loss_function,eval_dataloader,checkpoint=None)
        test_logger.info(f'第{epoch}次后，对模型进行验证，验证获得的结果为{eval_metrices}')
        # 如果验证结果好，保存训练模型
        if eval_metrices<best_metrices:
            best_metrices=eval_metrices
            pth_path= os.path.join(config['checkpoint_dir'],f'epoch_{epoch}_metrices_{best_metrices}.pth')
            best_checkpoint=pth_path
            torch.save(syncnet.state_dict(),pth_path)
        # 根据验证结果，调节学习率
        scheduler.step()
        train_logger.info('当前学习率为{}'.format(optimizer.param_groups[0]['lr']))
    
    # 测试模型
    test_logger.info(f'模型训练完毕，加载最好的模型{best_checkpoint}进行测试')
    test_loss=eval(syncnet,loss_function,test_dataloader,checkpoint=best_checkpoint)
    test_logger.info(f'测试结果为{test_loss}')



if __name__ == '__main__':
    import os,sys
    import yaml,glob

    config={}
    yaml_file=['config/dataset/common.yaml','config/model/sync_net.yaml']
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