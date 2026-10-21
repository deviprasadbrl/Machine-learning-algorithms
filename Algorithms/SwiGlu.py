import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self,d_model,intermediate_size):
        super().__init__()
        #The gate and vlaue weights are combined on the same projection to reduce computation
        self.gate_value=nn.Linear(d_model,2*intermediate_size,bias=False)

        self.out_gate=nn.Linear(intermediate_size,d_model,bias=False)

    def forward(self,x):
        combined=self.gate_value(x)
        gate,value=torch.chunk(combined,chunks=2,dim=-1)

        sigmoid_out=F.silu(gate)*value

        return self.out_gate(sigmoid_out)


if __name__=="__main__":
    # 1. Initialize the SwiGLU module
    # intermediate_size is often set to around (4 * d_model * 2/3) in models like LLaMA
    attention_module=SwiGLU(d_model=64, intermediate_size=128)
    
    # 2. Create dummy input tensor
    dummy_input=torch.randn(1, 3, 64)
    
    # 3. Run the forward pass
    output=attention_module(dummy_input)
        
    print("--- Shapes Verification ---")
    print(f"Input Shape:   {dummy_input.shape}")      
    print(f"Output Shape:  {output.shape}")
