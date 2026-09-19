import torch
import torch.nn as nn
import math
import torch.nn.functional as F

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

    def forward(self,x,is_causal=False,mask=None):
        batch_size,sequence,_=x.shape

        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)

        q_h=q.view(batch_size,sequence,self.num_heads,self.heads_dim).transpose(1,2)
        k_h=k.view(batch_size,sequence,self.num_heads,self.heads_dim).transpose(1,2)
        v_h=v.view(batch_size,sequence,self.num_heads,self.heads_dim).transpose(1,2)

        score=torch.matmul(q_h,k_h.transpose(-2,-1))/self.scale

        if mask is not None:
            if mask.dim()==2:
                mask=mask.unsqueeze(1).unsqueeze(2)
            score=score.masked_fill(mask==0,-1e9)

        if is_causal:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(sz=sequence, device=x.device,dtype=x.dtype)
            score = score + causal_mask

        attention_weights=F.softmax(score,dim=-1)

        context=torch.matmul(attention_weights,v_h)

        context=context.transpose(1,2).contiguous().view(batch_size,sequence,self.d_model)

        out=self.output(context)

        return attention_weights,out
        

if __name__ == "__main__":
  
    attention_module=MultiHeadAttention(d_model=64,num_heads=4)
    dummy_input=torch.randn(1, 3, 64)
    
    attention_weights,output=attention_module(dummy_input,is_causal=False,mask=None)
        
    print("--- Shapes Verification ---")
    print(f"Input Shape:   {dummy_input.shape}")      
    print(f"Attention_Weights Shape: {attention_weights.shape}")          
    print(f"Output Shape:  {output.shape}")
