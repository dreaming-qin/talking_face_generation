import torch

def reconstruct_idexp_lm3d(id_coeff, exp_coeff,key_id_base, key_exp_base ):
    """
    通过身份3DMM和表情3DMM生成3D keypoints，输入全为tensor
    Generate 3D landmark with keypoint base!
    id_coeff: Tensor[T, c=80]
    exp_coeff: Tensor[T, c=64]
    """
    id_coeff = id_coeff
    exp_coeff = exp_coeff
    id_base, exp_base = key_id_base, key_exp_base # [3*68, C]
    identity_diff_face = torch.matmul(id_coeff, id_base.transpose(0,1)) # [t,c],[c,3*68] ==> [t,3*68]
    expression_diff_face = torch.matmul(exp_coeff, exp_base.transpose(0,1)) # [t,c],[c,3*68] ==> [t,3*68]
    
    face = identity_diff_face + expression_diff_face # [t,3N]
    face = face.reshape([face.shape[0], -1,68, 3]) # [t,N,3]
    lm3d = face * 10
    return lm3d
    