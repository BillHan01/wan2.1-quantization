import torch
import viditq_extension.fused as fused_kernels
from icecream import ic
import argparse

def test_layer_norm_kernel(batch_size, token_len, in_feat):
    x = torch.randn((1, token_len, in_feat), dtype=torch.float16, device="cuda")
    x = torch.cat([x for _ in range(batch_size)], dim=0)
    weight = torch.randn((in_feat,), dtype=torch.float16, device="cuda")
    shift_msa = torch.randn((1, in_feat), dtype=torch.float16, device="cuda")
    shift_msa = torch.cat([shift_msa for _ in range(batch_size)], dim=0)
    scale_msa = torch.randn((1, in_feat), dtype=torch.float16, device="cuda")
    scale_msa = torch.cat([scale_msa for _ in range(batch_size)], dim=0)

    gt_y = torch.nn.functional.layer_norm(x, (in_feat,), weight=weight, eps=1e-5)
    gt_y_t2i = gt_y * (1 + scale_msa.view(batch_size, -1, in_feat)) + shift_msa.view(batch_size, -1, in_feat)

    out_fp16 = torch.zeros_like(x.view(-1, in_feat), dtype=torch.float16, device="cuda")

    input_sum = torch.zeros((batch_size * token_len, ), dtype=torch.float16, device="cuda")


    fused_kernels.layernorm_nobias_t2i_noquant_nosum_fuse(out_fp16, x.view(-1, in_feat), weight, shift_msa, scale_msa, 1e-5)
    ic(torch.max(torch.abs(gt_y_t2i.view(-1, in_feat) - out_fp16)))

    fused_kernels.layernorm_nobias(out_fp16, x.view(-1, in_feat), weight, 1e-5)
    ic(torch.max(torch.abs(gt_y.view(-1, in_feat) - out_fp16)))

    out = torch.zeros_like(x.view(-1, in_feat), dtype=torch.int8, device="cuda")

    scaling = torch.zeros((batch_size * token_len, ), dtype=torch.float16, device="cuda")

    fused_kernels.layernorm_nobias_quant_sum_fuse(out, x.view(-1, in_feat), weight, input_sum, scaling, 1e-5)
    scale_gt_y = gt_y.view(-1, in_feat).abs().max(dim=1, keepdim=True).values.to(torch.float32) / 127.0
    quant_gt_y = torch.round(gt_y.view(-1, in_feat).to(torch.float32) / scale_gt_y)
    sum_gt_y = quant_gt_y.view(-1, in_feat).sum(dim=1) * scale_gt_y.view(-1)

    ic(torch.max(torch.abs(quant_gt_y - out)))
    ic(torch.max(torch.abs(scale_gt_y.to(torch.float16) - scaling.view(-1, 1))))
    ic(torch.max(torch.abs(sum_gt_y.to(torch.float16).view(-1) - input_sum)))

    breakpoint()


    fused_kernels.layernorm_nobias_t2i_quant_sum_fuse(out, x.view(-1, in_feat), weight, shift_msa, scale_msa, input_sum, scaling, 1e-5)

    scale_gt_y = gt_y_t2i.view(-1, in_feat).abs().max(dim=1, keepdim=True).values.to(torch.float32) / 127.0
    quant_gt_y = torch.round(gt_y_t2i.view(-1, in_feat).to(torch.float32) / scale_gt_y)
    sum_gt = quant_gt_y.view(-1, in_feat).sum(dim=1) * scale_gt_y.view(-1)

    ic(torch.max(torch.abs(quant_gt_y - out)))
    ic(torch.max(torch.abs(scale_gt_y.to(torch.float16) - scaling.view(-1, 1))))
    ic(torch.max(torch.abs(sum_gt.to(torch.float16).view(-1) - input_sum)))

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--token_len", type=int, default=1024)
parser.add_argument("--in_feat", type=int, default=4096)

args = parser.parse_args()


num_iter = 10
num_warmup_iter = 5

batch_size = args.batch_size
token_len = args.token_len
in_feat = args.in_feat

print(f"batch_size: {batch_size} token len: {token_len} in feat: {in_feat}")

# test_layer_norm_kernel(batch_size, token_len, in_feat)

layernorm_nobias_args = []
layer_norm_fuse_quant_args = []
layer_norm_fuse_t2i_modulate_args = []
layer_norm_fuse_t2i_modulate_fuse_quant_args = []
layernorm_nobias_t2i_quant_sum_fuse_args = []
gt_args = []

for _ in range(num_iter + 1):
    x = torch.randn((token_len * batch_size, in_feat), dtype=torch.float16, device="cuda")
    weight = torch.randn((in_feat,), dtype=torch.float16, device="cuda")

    out = torch.zeros_like(x, dtype=torch.int8, device="cuda")
    fp16_out = torch.zeros_like(x, dtype=torch.float16, device="cuda")
    scaling = torch.zeros((token_len * batch_size, ), dtype=torch.float16, device="cuda")
    sum_output = torch.zeros((batch_size * token_len, ), dtype=torch.float16, device="cuda")
    shift_msa = torch.randn((batch_size, in_feat), dtype=torch.float16, device="cuda")
    scale_msa = torch.randn((batch_size, in_feat), dtype=torch.float16, device="cuda")
    layernorm_nobias_args.append((fp16_out, x, weight, 1e-5))
    layer_norm_fuse_quant_args.append((out, x, weight, scaling, 1e-5))
    layer_norm_fuse_t2i_modulate_args.append((fp16_out, x, weight, shift_msa, scale_msa, 1e-5))
    layer_norm_fuse_t2i_modulate_fuse_quant_args.append((out, x, weight, shift_msa, scale_msa, scaling, 1e-5))
    layernorm_nobias_t2i_quant_sum_fuse_args.append((out, x, weight, shift_msa, scale_msa, sum_output, scaling, 1e-5))
    gt_args.append((x, weight, shift_msa, scale_msa))

for _ in range(num_warmup_iter):
    fused_kernels.layernorm_nobias(*layernorm_nobias_args[-1])
    fused_kernels.layernorm_nobias_t2i_quant_sum_fuse(*layernorm_nobias_t2i_quant_sum_fuse_args[-1])
    fused_kernels.layernorm_nobias_t2i_fuse(*layer_norm_fuse_t2i_modulate_args[-1])
    torch.nn.functional.layer_norm(gt_args[-1][0], (in_feat,), weight=gt_args[-1][1], eps=1e-5)

torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)


start.record()
for i in range(num_iter):
    fused_kernels.layernorm_nobias(*layernorm_nobias_args[i])

end.record()
torch.cuda.synchronize()
print(
    f"layernorm_nobias average inference time: {start.elapsed_time(end) / num_iter:.4f} ms")


start.record()
for i in range(num_iter):
    fused_kernels.layernorm_nobias_t2i_fuse(*layer_norm_fuse_t2i_modulate_args[i])
end.record()
torch.cuda.synchronize()
print(
    f"layernorm_nobias_t2i_fuse average inference time: {start.elapsed_time(end) / num_iter:.4f} ms")


start.record()
for i in range(num_iter):
    fused_kernels.layernorm_nobias_t2i_quant_sum_fuse(*layernorm_nobias_t2i_quant_sum_fuse_args[i])
end.record()
torch.cuda.synchronize()
print(
    f"layernorm_nobias_t2i_quant_sum_fuse average inference time: {start.elapsed_time(end) / num_iter:.4f} ms")

start.record()
for i in range(num_iter):
    torch.nn.functional.layer_norm(gt_args[i][0], (in_feat,), weight=gt_args[i][1], eps=1e-5)

end.record()
torch.cuda.synchronize()
print(
    f"torch layer norm inference time: {start.elapsed_time(end) / num_iter:.4f} ms")  # Add missing parentheses here


start.record()
for i in range(num_iter):
    torch.nn.functional.layer_norm(gt_args[i][0], (in_feat,), weight=gt_args[i][1], eps=1e-5) * (1 + gt_args[i][3].view(batch_size, -1, in_feat)) + gt_args[i][2].view(batch_size, -1, in_feat)

end.record()
torch.cuda.synchronize()
print(
    f"torch layer norm and t2i modulate inference time: {start.elapsed_time(end) / num_iter:.4f} ms")  # Add missing parentheses here
