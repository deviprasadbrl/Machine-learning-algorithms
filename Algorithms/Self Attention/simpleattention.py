import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class SimpleSelfAttenttion(nn.Module):
    def __init__(self,d_model):
        super().__init__()

        self.q_proj=nn.Linear(d_model,d_model)
        self.k_proj=nn.Linear(d_model,d_model)
        self.v_proj=nn.Linear(d_model,d_model)

        self.scale=d_model**0.5

    def forward(self,x,is_causal=False,mask=None):
        batch_size,sequence,_=x.shape
        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)

        score=torch.matmul(q,k.transpose(-2,-1))

        if mask is not None:
            if mask.dim()==2:
                mask=mask.unsqueeze(1).unsqueeze(2)
            score=score.masked_fill(mask==0,-1e9)

        if is_causal:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(sz=sequence, device=x.device,dtype=x.dtype)
            score = score + causal_mask

        score=score/self.scale

        attention_weights=F.softmax(score,dim=-1)

        out=torch.matmul(attention_weights,v)

        return attention_weights,out

if __name__ == "__main__":

    attention_module=SimpleSelfAttenttion(d_model=4)
    
    dummy_input=torch.randn(1, 3, 4)

    attention_weights,output=attention_module(dummy_input,is_causal=False,mask=None)
    
    print("--- Shapes Verification ---")
    print(f"Input Shape:   {dummy_input.shape}")      
    print(f"Attention_Weights Shape: {attention_weights.shape}")          
    print(f"Output Shape:  {output.shape}")
