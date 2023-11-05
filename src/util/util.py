import torch

def get_window(feature, win_size):
    """

    Args:
        feature (torch.tensor): (B,Len,feature dim)

    Returns:
        feature_wins (torch.tensor): (B,Len,2*win_size+1,feature dim)
    """
    B,L,_ = feature.shape
    feature_wins=[]
    for batch in range(B):
        batch_ans=[]
        for num in range(L):
            num_ans=[]
            for i in range(num - win_size, num + win_size + 1):
                if i < 0:
                    num_ans.append(feature[batch,0])
                elif i >= L:
                    num_ans.append(feature[batch,-1])
                else:
                    num_ans.append(feature[batch,i])
            num_ans=torch.stack(num_ans)
            batch_ans.append(num_ans)
        batch_ans=torch.stack(batch_ans)
        feature_wins.append(batch_ans)
    feature_wins=torch.stack(feature_wins)
    return feature_wins


def get_windows_by_repeat(feature, win_size):
    """与get_window不同，get_windows_by_repeat中windows的元素是自己

    Args:
        feature (torch.tensor): (B,Len,feature dim)

    Returns:
        feature_wins (torch.tensor): (B,Len,2*win_size+1,feature dim)
    """
    B,L,_ = feature.shape
    feature_wins=[]
    for batch in range(B):
        batch_ans=[]
        for num in range(L):
            num_ans=feature[batch,-1].expand(2*win_size+1,-1)
            batch_ans.append(num_ans)
        batch_ans=torch.stack(batch_ans)
        feature_wins.append(batch_ans)
    feature_wins=torch.stack(feature_wins)
    return feature_wins
