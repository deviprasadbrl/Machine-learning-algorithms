import torch
import torch.nn as nn

class RMSNORM(nn.Module):
    def __init__(self,d_model,eps=1e-6):
        super().__init__()
        self.eps=eps
        self.gamma=nn.Parameter(torch.ones(d_model))

    def forward(self,x):
        x_dtype=x.dtype

        x=x.to(torch.float32)

        square_mean=x.pow(2).mean(dim=-1,keepdim=True)
        rms=x*torch.rsqrt(square_mean+self.eps)

        weighted_rms=(self.gamma*rms).to(x_dtype)

        return weighted_rms


if __name__ == "__main__":
    # 1. Initialize the RMSNorm module
    norm_module=RMSNORM(d_model=64)
    
    # 2. Create dummy input tensor
    dummy_input=torch.randn(1, 3, 64)
    
    # 3. Run the forward pass
    output=norm_module(dummy_input)
        
    print("--- Shapes Verification ---")
    print(f"Input Shape:   {dummy_input.shape}")      
    print(f"Output Shape:  {output.shape}")
