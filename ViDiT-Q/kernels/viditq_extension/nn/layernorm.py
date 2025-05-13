import torch
import torch.nn as nn
import viditq_extension.fused as fused_kernels
from viditq_extension.nn.base import QuantParams

class LayerNormGeneral(nn.Module):
    def __init__(
        self,
        hidden_size : int,
        act_sum : bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=torch.float16),
        )
        self.variance_eps = eps

        self.act_sum = act_sum

    @classmethod
    def from_layer_norm(cls, layer_norm: nn.LayerNorm):
        ln = cls(
            hidden_size=layer_norm.weight.shape[0],
            eps=layer_norm.eps,
        )
        ln.weight = layer_norm.weight.clone().to(torch.float16)
        return ln
    
    def forward(
        self,
        input: torch.Tensor,
        shift_msa: torch.Tensor,
        scale_msa: torch.Tensor,
        quant_params: QuantParams,
    ):
        input = input.contiguous()
        shape = input.shape
        hidden_dim = shape[-1]
        input = input.view(-1, hidden_dim)
        output = torch.empty_like(input, dtype=torch.int8)
        # TODO: implement the complete forward pass
        fused_kernels.layernorm_nobias_t2i_quant_sum_fuse(
            output,
            input,
            self.weight,
            shift_msa.view(-1, hidden_dim),
            scale_msa.view(-1, hidden_dim),
            quant_params.sum_input,
            quant_params.scale_input,
            self.variance_eps,
        )

        return output.view(shape)