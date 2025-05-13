from torch.nn.modules.module import Module
from torch.autograd import Function, Variable
import torch
#import resample2d_cuda

class Resample2dFunction(Function):

    @staticmethod
    def forward(ctx, input1, input2, kernel_size=1, bilinear= True):
        assert input1.is_contiguous()
        assert input2.is_contiguous()

        ctx.save_for_backward(input1, input2)
        ctx.kernel_size = kernel_size
        ctx.bilinear = bilinear

        _, d, _, _ = input1.size()
        b, _, h, w = input2.size()
        output = input1.new(b, d, h, w).zero_()
        
        #待修改
        output = torch.nn.functional.grid_sample(input1, input2.permute(0, 2, 3, 1), mode='bilinear', align_corners=False)
        
        #print("######")
        #print(input1.size(),input2.size(),kernel_size)

        #resample2d_cuda.forward(input1, input2, output, kernel_size, bilinear)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        assert grad_output.is_contiguous()

        input1, input2 = ctx.saved_tensors

        grad_input1 = Variable(input1.new(input1.size()).zero_())
        grad_input2 = Variable(input1.new(input2.size()).zero_())
        
        #待修改,疑似有问题
        #backward resample
        grad_output = torch.nn.functional.grid_sample(grad_output, input2.permute(0, 2, 3, 1), mode='bilinear', align_corners=False)

        """
        resample2d_cuda.backward(input1, input2, grad_output.data,
                                 grad_input1.data, grad_input2.data,
                                 ctx.kernel_size, ctx.bilinear)
        """

        return grad_input1, grad_input2, None, None

class Resample2d(Module):

    def __init__(self, kernel_size=1, bilinear = True):
        super(Resample2d, self).__init__()
        self.kernel_size = kernel_size
        self.bilinear = bilinear

    def forward(self, input1, input2):
        input1_c = input1.contiguous()
        return Resample2dFunction.apply(input1_c, input2, self.kernel_size, self.bilinear)
