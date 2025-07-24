import os 
import torch
import torch.nn as nn
import numpy as np
import math

"""
QKV注意力计算
"""
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, n_heads):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.n_heads = n_heads


    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q=1, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context


"""
多头注意力
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, len_q, len_k):
        super(MultiHeadAttention, self).__init__()

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ScaledDotProductAttention = ScaledDotProductAttention(self.d_k, n_heads)
        self.len_q = len_q
        self.len_k = len_k


    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # Q: [batch_size, n_heads, len_q, d_k]
        #print("Q.SIZE",Q.size())
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]

        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context= self.ScaledDotProductAttention(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).cuda()(output + residual)


"""
FeedForward
"""
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(), 
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.d_model = d_model

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).cuda()(output + residual)


"""
Transformer Encoder 子结构
"""
class Encoder(nn.Module):
    def __init__(self,d_model=512,d_ff=2048, d_k=64, d_v=64, n_heads=8, len_q=1):
        super(Encoder,self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, 1, len_q)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)
    def forward(self,enc_inputs):
        enc_outputs= self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs

"""
余弦位置编码
"""
class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=10000):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        

        self.register_buffer('pe', pe)

    def forward(self, x):
        pos = self.pe[: ,:x.size(1), :] + x
        return pos
    
    
"""
门控渐进融合网络ALBERT
"""
class ALGatedProgressiveFusionLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, len_q, len_k, shared_attn=None, shared_ffn=None):
        super(ALGatedProgressiveFusionLayer, self).__init__()
        
        if shared_attn is not None:
            self.self_attn = shared_attn
        else:
            self.self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, len_q, len_k)
        
        if shared_ffn is not None:
            self.feed_forward = shared_ffn
        else:
            self.feed_forward = PoswiseFeedForwardNet(d_model, 2048)  
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.gate_generator = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x, image_feature, text_feature):  
        gate_weights = self.gate_generator(image_feature)
        
        gated_image_feature = (1 - gate_weights) * image_feature
        gated_text_feature = gate_weights * text_feature
        
        combined_feature = gated_image_feature + gated_text_feature
        combined_feature = torch.cat([image_feature, combined_feature], dim=1)
        combined_feature = self.norm3(combined_feature)
        
        attn_output = self.self_attn(combined_feature, combined_feature, combined_feature)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x



# best version layer 4 4
class Transformer(nn.Module):
    def __init__(self, class_num=31, input_dim=512, n_layers=4, d_k=64, d_v=64, n_heads=4, len_q=100, len_k=100):
        super(Transformer, self).__init__()
        self.PE = FixedPositionalEncoding(embedding_dim=input_dim)
        self.shared_attn = MultiHeadAttention(d_model=input_dim, d_k=d_k, d_v=d_v, 
                               n_heads=n_heads + 4, len_q=len_q, len_k=len_k)
        self.shared_ffn = PoswiseFeedForwardNet(d_model=input_dim, d_ff=2048)
        self.stages = nn.ModuleList([
            ALGatedProgressiveFusionLayer(d_model=input_dim, d_k=d_k, d_v=d_v, n_heads=n_heads + 4, len_q=len_q, len_k=len_k,shared_attn=self.shared_attn,shared_ffn=self.shared_ffn)
            for _ in range(n_layers)
        ])
        
        self.stages_2nd =nn.ModuleList([
            Encoder(d_model=input_dim, d_ff= 2 * input_dim, d_k=int(input_dim / 4),
                               d_v=int(input_dim / 4), n_heads=4)
            for i in range(n_layers)
        ])
        
        self.lin_down = nn.Linear(768, input_dim)
        self.lin_cla = nn.Linear(input_dim, class_num)
        self.d_model = input_dim
        
        
    def forward(self, image_feature, init_text_feature):
        text_feature = self.lin_down(init_text_feature)
        
        initial_fusion = torch.concat([image_feature, text_feature], dim=1)
        
        input_feature = self.PE(initial_fusion)
        out = input_feature
        
        for layer in self.stages:
            out = layer(out, image_feature, text_feature)

        # out 24 512
        out = out[:, 0, :]
        
        for layer_2nd in self.stages_2nd:
            out = layer_2nd(out)
        
        out_final = out[:, 0, :]
        # out final 24 512
        classes_out = self.lin_cla(out_final)
        return classes_out, out_final

if __name__ == '__main__':
    model=Transformer(input_dim=512,n_layers=4).cuda()
    x=torch.ones((2,1,512)).cuda()
    z=torch.ones((2,291,768)).cuda()
    y=model(x,z)
    #net=ft_net_swin(class_num=31,circle=True)





