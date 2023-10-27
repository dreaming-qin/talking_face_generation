import torch


def get_window(feature, win_size):
    """

    Args:
        feature (torch.tensor): (B,Len,video dim)

    Returns:
        feature_wins (torch.tensor): (B,Len,win_size,video dim)
    """
    num_frames = len(feature)
    ph_frames = []
    for rid in range(0, num_frames):
        ph = []
        for i in range(rid - win_size, rid + win_size + 1):
            if i < 0:
                ph.append(31)
            elif i >= num_frames:
                ph.append(31)
            else:
                ph.append(feature[i])

        ph_frames.append(ph)

    feature_wins =torch.Tensor(ph_frames)
    feature_wins=feature_wins.permute(0,1,3,2)
    return feature_wins
