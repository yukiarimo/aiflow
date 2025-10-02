import math
import mlx.core as mx
import mlx.nn as nn

class LoRALinear(nn.Module):
    @staticmethod
    def from_linear(linear: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        output_dims, input_dims = linear.weight.shape
        lora_lin = LoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )
        lora_lin.linear = linear
        return lora_lin

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = alpha / r

        scale_lora_a = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-scale_lora_a,
            high=scale_lora_a,
            shape=(input_dims, r),
        )
        self.lora_b = mx.zeros(shape=(r, output_dims))

    def __call__(self, x):
        y = self.linear(x)
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        return y + (self.scale * z).astype(x.dtype)