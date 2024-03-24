from torch import nn

from src.model.exp3DMM.self_attention_pooling import SelfAttentionPooling
from src.model.exp3DMM.fusion import PositionalEncoding



class AudioEncoder(nn.Module):
    r'''音频编码器，输入一串音频的MFCC，输出音频特征，长度为256'''
    def __init__(self,feature_dim,num_encoder_layers,
                 nhead,feedforward_dim,seq_len,**_):
        super(AudioEncoder, self).__init__()


        self.mapping_net=nn.Sequential(
            nn.Linear(12,64),
            nn.Linear(64,128),
            nn.ReLU(True),
            nn.Linear(128,256),
        )

        
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=nhead,
                        dim_feedforward=feedforward_dim,batch_first=True)
        encoder_norm = nn.LayerNorm(feature_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # 音频长度28
        self.pos=PositionalEncoding(feature_dim, n_position=seq_len)

        self.self_atten=SelfAttentionPooling(input_dim=feature_dim)



    def forward(self,audio):
        '''audio输入维度[B,len,28,mfcc dim]
        输出维度[B,len,256]'''

        out=self.mapping_net(audio)

        B,L,_,dim=out.shape
        out=out.reshape(B*L,-1,dim)
        pos_embd=self.pos(out.shape[1])
        # pytorch中的transformer未进行pos操作，在这进行操作
        out=out+pos_embd
        out=self.encoder(out)

        out=self.self_atten(out)
        out=out.reshape(B,L,dim)

        return out
    


# if __name__=='__main__':
#     # # 定义编码器，词典大小为10，要把token编码成128维的向量
#     # embedding = nn.Embedding(10, 128)
#     # # 定义transformer，模型维度为128（也就是词向量的维度）
#     # transformer = nn.Transformer(d_model=128, batch_first=True) # batch_first一定不要忘记
#     # # 定义源句子，可以想想成是 <bos> 我 爱 吃 肉 和 菜 <eos> <pad> <pad>
#     # src = torch.LongTensor([[0, 3, 4, 5, 6, 7, 8, 1, 2, 2]])
#     # # 定义目标句子，可以想想是 <bos> I like eat meat and vegetables <eos> <pad>
#     # tgt = torch.LongTensor([[0, 3, 4, 5, 6, 7, 8, 1, 2]])
#     # # 将token编码后送给transformer（这里暂时不加Positional Encoding）
#     # outputs = transformer(embedding(src), embedding(tgt))
#     # outputs.size()


#     input=torch.ones((1,37,28,12))

#     # 初始化模型
#     num_encoder_layers=7
#     feature_dim=256

#     mapping_net=nn.Sequential(
#         nn.Linear(12,64),
#         nn.Linear(64,128),
#         nn.ReLU(True),
#         nn.Linear(128,256),
#     )

    
#     encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8,
#                     dim_feedforward=512,batch_first=True)
#     encoder_norm = nn.LayerNorm(feature_dim)
#     encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

#     self_atten=SelfAttentionPooling(input_dim=feature_dim)

#     # 音频长度28
#     pos=PositionalEncoding(feature_dim, n_position=28)

#     # test
#     temp=0
#     temp+=cnt_params(encoder)[0]
#     temp+=cnt_params(mapping_net)[0]
#     temp+=cnt_params(self_atten)[0]
#     temp+=cnt_params(pos)[0]
#     print(temp)

#     # 前向传播
#     out=mapping_net(input)

#     B,L,_,dim=out.shape
#     out=out.reshape(B*L,-1,dim)
#     pos_embd=pos(out.shape[1])
#     # pytorch中的transformer未进行pos操作，在这进行操作
#     out=out+pos_embd
#     out=encoder(out)

#     out=self_atten(out)
#     out=out.reshape(B,L,dim)
#     print(out.shape)

    







