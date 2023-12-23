import glob
import shutil
import os

# test 
if __name__=='__main__':
    import sys
    path=sys.path[0]
    for _ in range(2):
        path=os.path.dirname(path)
    sys.path.append(path)
    sys.path.append(os.path.join(path,'syncnet_python'))



from syncnet_python.run_pipeline import run_pipeline_main
from syncnet_python.calculate_scores_real_videos import caculate_metrices_main



'''syync_net取值范围[0,正无穷]，越大越好'''

def sync_net_by_path(predict_video_file,gt_video_file):
    
    run_pipeline_main(predict_video_file,'wav2lip','temp')
    fake_dis,fake_conf=caculate_metrices_main(predict_video_file,'wav2lip','temp')
    shutil.rmtree('temp')
    
    run_pipeline_main(gt_video_file,'wav2lip','temp')
    real_dis,real_conf=caculate_metrices_main(gt_video_file,'wav2lip','temp')
    shutil.rmtree('temp')
    
    return fake_conf,fake_dis,real_conf,real_dis



def sync_net_by_dir(predict_video_dir,gt_video_dir):
    fake_conf_ans=[]
    real_conf_ans=[]
    fake_dis_ans=[]
    real_dis_ans=[]
    predict_video_dir=sorted(glob.glob(f'{predict_video_dir}/*.mp4'))
    gt_video_dir=sorted(glob.glob(f'{gt_video_dir}/*.mp4'))
    for predict_video_file,gt_video_file in zip(predict_video_dir,gt_video_dir):
        assert os.path.basename(predict_video_file)==os.path.basename(gt_video_file)
        fake_conf,fake_dis,real_conf,real_dis=sync_net_by_path(predict_video_file,gt_video_file)
        fake_conf_ans.append(fake_conf)
        real_conf_ans.append(real_conf)
        fake_dis_ans.append(fake_dis)
        real_dis_ans.append(real_dis)

    fake_conf = sum(fake_conf_ans) / len(fake_conf_ans)
    real_conf = sum(real_conf_ans) / len(real_conf_ans)
    fake_dis = sum(fake_dis_ans) / len(fake_dis_ans)
    real_dis = sum(real_dis_ans) / len(real_dis_ans)
    return fake_conf,fake_dis,real_conf,real_dis



if __name__=='__main__':
    print(sync_net_by_dir('result/atvg/result/real',
                        'result/atvg/result/fake'))