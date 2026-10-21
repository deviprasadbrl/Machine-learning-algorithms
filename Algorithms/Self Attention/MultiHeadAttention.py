import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()

        self.d_model=d_model
        self.num_heads=num_heads
        self.heads_dim=d_model//num_heads

        self.q_proj=nn.Linear(d_model,d_model)
        self.k_proj=nn.Linear(d_model,d_model)
        self.v_proj=nn.Linear(d_model,d_model)

        self.output=nn.Linear(d_model,d_model)
        self.scale=math.sqrt(self.heads_dim)

    def forward(self,x,mask=None,is_causal=False,context=None):
        batch_size,sequence,_=x.shape

        kv_input=context if context is not None else x

        q=self.q_proj(x)
        k=self.k_proj(kv_input)
        v=self.v_proj(kv_input)

        q_h=q.view(batch_size,-1,self.num_heads,self.heads_dim).transpose(1,2)
        k_h=k.view(batch_size,-1,self.num_heads,self.heads_dim).transpose(1,2)
        v_h=v.view(batch_size,-1,self.num_heads,self.heads_dim).transpose(1,2)

        score=torch.matmul(q_h,k_h.transpose(-2,-1))/self.scale

        if mask is not None:
            if mask.dim()==2:
                mask=mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim()==3:
                mask=mask.unsqueeze(1)
            score=score.masked_fill(mask==0,torch.finfo(score.dtype).min)

        if is_causal:
            assert context is None, "is_causal=True is only valid for self-attention"

            Lq,Lk=score.size(-2),score.size(-1)

            causal=torch.ones(Lq,Lk,dtype=torch.bool,device=x.device).tril()
            score = score.masked_fill(~causal,torch.finfo(score.dtype).min)

        attention_weights=F.softmax(score,dim=-1)

        context=torch.matmul(attention_weights,v_h)

        context=context.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)

        out=self.output(context)

        return out



if __name__ == "__main__":
  
    attention_module=MultiHeadAttention(d_model=64,num_heads=4)
    dummy_input=torch.randn(1, 3, 64)
    
    output=attention_module(dummy_input,is_causal=False,mask=None)
        
    print("--- Shapes Verification ---")
    print(f"Input Shape:   {dummy_input.shape}")               
    print(f"Output Shape:  {output.shape}")
        
    print("--- Shapes Verification ---")
    print(f"Input Shape:   {dummy_input.shape}")      
    print(f"Attention_Weights Shape: {attention_weights.shape}")          
    print(f"Output Shape:  {output.shape}")
