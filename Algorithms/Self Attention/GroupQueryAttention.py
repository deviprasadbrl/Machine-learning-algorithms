import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    def __init__(self,d_model,num_q,num_kv):
        super().__init__()

        assert d_model%num_q==0 ,"d_model must be divisible by num_q"
        assert num_q%num_kv==0,"num_q must be divisible by num_kv"

        self.d_model=d_model
        self.num_q=num_q
        self.num_kv=num_kv

        self.head_dim=self.d_model//self.num_q
        self.num_q_per_kv=self.num_q//self.num_kv

        self.scale=math.sqrt(self.head_dim)

        self.q_proj=nn.Linear(self.d_model,self.num_q*self.head_dim)
        self.k_proj=nn.Linear(self.d_model,self.num_kv*self.head_dim)
        self.v_proj=nn.Linear(self.d_model,self.num_kv*self.head_dim)
        self.out_proj=nn.Linear(self.d_model,self.d_model)

    def forward(self,x,mask=None,is_causal=False):
        batch,_,_=x.shape

        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)

        q=q.view(batch,-1,self.num_q,self.head_dim).transpose(1,2)
        k=k.view(batch,-1,self.num_kv,self.head_dim).transpose(1,2)
        v=v.view(batch,-1,self.num_kv,self.head_dim).transpose(1,2)

        if self.num_q_per_kv>1:
            k=k.repeat_interleave(self.num_q_per_kv,dim=1)
            v=v.repeat_interleave(self.num_q_per_kv,dim=1)

        score=torch.matmul(q,k.transpose(-2,-1))/self.scale

        if mask is not None:
            if mask.dim()==2:
                mask=mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim()==3:
                mask=mask.unsqueeze(1)
            score=score.masked_fill(mask==0,torch.finfo(score.dtype).min)


        if is_causal:
            Lq,Lk=score.size(-2),score.size(-1)
            causal=torch.ones(Lq,Lk,dtype=torch.bool,device=x.device).tril()
            score=score.masked_fill(~causal,torch.finfo(score.dtype).min)

        attention_weights=F.softmax(score,dim=-1)

        context=torch.matmul(attention_weights,v)
        context=context.transpose(1,2).contiguous().view(batch,-1,self.d_model)

        out=self.out_proj(context)
        
        return out



if __name__ == "__main__":
    # 1. Initialize your specific module
    attention_module=GroupedQueryAttention(d_model=64, num_q=8, num_kv=2)
    
    # 2. Create dummy input tensor
    dummy_input=torch.randn(1, 3, 64)
    
    # 3. Run the forward pass (Returns only 'out')
    output=attention_module(dummy_input, is_causal=False, mask=None)
        
    print("--- Shapes Verification ---")
    print(f"Input Shape:   {dummy_input.shape}")      
    print(f"Output Shape:  {output.shape}")
