import torch
import numpy as np

def get_window(feature, win_size):
    """

    Args:
        feature (torch.tensor): (B,Len,feature dim)

    Returns:
        feature_wins (torch.tensor): (B,Len,win_size,feature dim)
    """
    B,L,_ = feature.shape
    ans=[]
    for batch in range(B):
        batch_ans=[]
        for num in range(L):
            num_ans=[]
            for i in range(num - win_size, num + win_size + 1):
                if i < 0:
                    num_ans.append(feature[batch,0])
                elif i >= num:
                    num_ans.append(feature[batch,-1])
                else:
                    num_ans.append(feature[batch,i])
            num_ans=torch.stack(num_ans)
            batch_ans.append(num_ans)
        batch_ans=torch.stack(batch_ans)
        ans.append(batch_ans)
    ans=torch.stack(ans)
    return ans