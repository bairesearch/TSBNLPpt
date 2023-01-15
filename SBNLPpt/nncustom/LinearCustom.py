"""LinearCustom.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see XXXpt_globalDefs.py

# Usage:
see XXXpt_globalDefs.py

# Description:
Linear Custom

"""

from SBNLPpt_globalDefs import *

import math

import torch
from torch import nn
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function

if(useAutoResizeInput):
	from . import LinearCustomAutoResize
if(memoryTraceBias):
	from . import LinearCustomMTB
if(simulatedDendriticBranches):
	from . import LinearCustomSDB

if(useAutoResizeInput):
	#from https://pytorch.org/docs/master/notes/extending.html
	class LinearFunction(Function):
		@staticmethod
		# ctx is the first argument to forward
		def forward(ctx, input, weight, bias):
			# The forward pass can use ctx.
			ctx.save_for_backward(input, weight, bias)
			output = input.mm(weight.t())
			if bias is not None:
				output += bias.unsqueeze(0).expand_as(output)
			return output

		@staticmethod
		def backward(ctx, grad_output):
			input, weight, bias = ctx.saved_tensors
			grad_input = grad_weight = grad_bias = None

			if ctx.needs_input_grad[0]:
				grad_input = grad_output.mm(weight)
			if ctx.needs_input_grad[1]:
				grad_weight = grad_output.t().mm(input)
			if bias is not None and ctx.needs_input_grad[2]:
				grad_bias = grad_output.sum(0)

			return grad_input, grad_weight, grad_bias	
else:
	#from https://pytorch.org/docs/master/notes/extending.html
	class LinearFunction(Function):
		@staticmethod
		# ctx is the first argument to forward
		def forward(ctx, input, weight, bias):
			# The forward pass can use ctx.
			ctx.save_for_backward(input, weight, bias)
			output = input.matmul(weight.t())
			if bias is not None:
				output += bias.unsqueeze(0).expand_as(output)
			return output

		@staticmethod
		def backward(ctx, grad_output):
			input, weight, bias = ctx.saved_tensors
			grad_input = grad_weight = grad_bias = None

			if ctx.needs_input_grad[0]:
				grad_input = grad_output.matmul(weight)
			if ctx.needs_input_grad[1]:
				grad_weight = grad_output.swapaxes(-1, -2).matmul(input)
			if bias is not None and ctx.needs_input_grad[2]:
				grad_bias = grad_output.sum(0)

			return grad_input, grad_weight, grad_bias	


if(useModuleLinearTemplateCurrent):
	class Linear(nn.Module):
		#from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
		r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
		This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
		On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.
		Args:
			in_features: size of each input sample
			out_features: size of each output sample
			bias: If set to ``False``, the layer will not learn an additive bias.
				Default: ``True``
		Shape:
			- Input: :math:`(*, H_{in})` where :math:`*` means any number of
			  dimensions including none and :math:`H_{in} = \text{in\_features}`.
			- Output: :math:`(*, H_{out})` where all but the last dimension
			  are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
		Attributes:
			weight: the learnable weights of the module of shape
				:math:`(\text{out\_features}, \text{in\_features})`. The values are
				initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
				:math:`k = \frac{1}{\text{in\_features}}`
			bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
					If :attr:`bias` is ``True``, the values are initialized from
					:math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
					:math:`k = \frac{1}{\text{in\_features}}`
		Examples::
			>>> m = nn.Linear(20, 30)
			>>> input = torch.randn(128, 20)
			>>> output = m(input)
			>>> print(output.size())
			torch.Size([128, 30])
		"""
		__constants__ = ['in_features', 'out_features']
		in_features: int
		out_features: int
		weight: Tensor

		def __init__(self, in_features: int, out_features: int, bias: bool = True,
					 device=None, dtype=None) -> None:
			factory_kwargs = {'device': device, 'dtype': dtype}
			super(Linear, self).__init__()
			self.in_features = in_features
			self.out_features = out_features
			self.weight = Parameter(torch.empty((out_features*numberOfIndependentDendriticBranches, in_features), **factory_kwargs))
			if bias:
				self.bias = Parameter(torch.empty(out_features*numberOfIndependentDendriticBranches, **factory_kwargs))
			else:
				self.register_parameter('bias', None)
			self.reset_parameters()
			
			if(memoryTraceBias):
				device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
				self.memoryTrace = torch.empty(in_features, out_features*numberOfIndependentDendriticBranches).to(device)	#note memoryTrace.shape = [input_features, output_features] (transpose of weights)
				#self.memoryTrace = nn.Parameter(torch.empty(in_features, out_features*numberOfIndependentDendriticBranches), requires_grad=False)

		def reset_parameters(self) -> None:
			# Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
			# uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
			# https://github.com/pytorch/pytorch/issues/57109
			init.kaiming_uniform_(self.weight, a=math.sqrt(5))
			if self.bias is not None:
				fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
				bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
				init.uniform_(self.bias, -bound, bound)

		def forward(self, input: Tensor) -> Tensor:
			#orig: return F.linear(input, self.weight, self.bias)
			requireResizeInput = False
			shapeOutput = None
			memoryTraceUpdate = None

			if(useAutoResizeInput):
				input, requireResizeInput, shapeOutput = LinearCustomAutoResize.autoResizeInput(input, self.weight)
			if(memoryTraceBias):
				output, memoryTraceUpdate = LinearCustomMTB.executeAndCalculateMemoryTraceUpdate(input, self.weight, self.bias, self.memoryTrace)
			else:
				#See the autograd section for explanation of what happens here.
				output = LinearFunction.apply(input, self.weight, self.bias)

			if(simulatedDendriticBranches):
				output, shapeOutput, memoryTraceUpdate = LinearCustomSDB.selectDendriticBranchOutput(output, requireResizeInput, shapeOutput, memoryTraceUpdate)
			if(memoryTraceBias):
				self.memoryTrace = LinearCustomMTB.updateMemoryTrace(self.memoryTrace, memoryTraceUpdate)
			if(useAutoResizeInput):
				output = LinearCustomAutoResize.autoResizeOutput(requireResizeInput, output, shapeOutput)

			return output

		def extra_repr(self) -> str:
			return 'in_features={}, out_features={}, bias={}'.format(
				self.in_features, self.out_features, self.bias is not None
			)	
else:
	#from https://pytorch.org/docs/master/notes/extending.html
	class Linear(nn.Module):
		def __init__(self, input_features, output_features, bias=True):
			super(Linear, self).__init__()
			self.input_features = input_features
			self.output_features = output_features

			# nn.Parameter is a special kind of Tensor, that will get
			# automatically registered as Module's parameter once it's assigned
			# as an attribute. Parameters and buffers need to be registered, or
			# they won't appear in .parameters() (doesn't apply to buffers), and
			# won't be converted when e.g. .cuda() is called. You can use
			# .register_buffer() to register buffers.
			# nn.Parameters require gradients by default.
			self.weight = nn.Parameter(torch.empty(output_features*numberOfIndependentDendriticBranches, input_features))
			if bias:
				self.bias = nn.Parameter(torch.empty(output_features*numberOfIndependentDendriticBranches))
			else:
				# You should always register all possible parameters, but the
				# optional ones can be None if you want.
				self.register_parameter('bias', None)

			# Not a very smart way to initialize weights
			nn.init.uniform_(self.weight, -0.1, 0.1)
			if self.bias is not None:
				nn.init.uniform_(self.bias, -0.1, 0.1)

			if(memoryTraceBias):
				device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
				self.memoryTrace = torch.empty(input_features, output_features*numberOfIndependentDendriticBranches).to(device)	#note memoryTrace.shape = [input_features, output_features] (transpose of weights)
				#self.memoryTrace = nn.Parameter(torch.empty(input_features, output_features*numberOfIndependentDendriticBranches), requires_grad=False)

		def forward(self, input):
			#orig: return LinearFunction.apply(input, self.weight, self.bias)

			requireResizeInput = False
			shapeOutput = None
			memoryTraceUpdate = None

			if(useAutoResizeInput):
				input, requireResizeInput, shapeOutput = LinearCustomAutoResize.autoResizeInput(input, self.weight)
			if(memoryTraceBias):
				output, memoryTraceUpdate = LinearCustomMTB.executeAndCalculateMemoryTraceUpdate(input, self.weight, self.bias, self.memoryTrace)
			else:
				#See the autograd section for explanation of what happens here.
				output = LinearFunction.apply(input, self.weight, self.bias)

			if(simulatedDendriticBranches):
				output, shapeOutput, memoryTraceUpdate = LinearCustomSDB.selectDendriticBranchOutput(output, requireResizeInput, shapeOutput, memoryTraceUpdate)
			if(memoryTraceBias):
				self.memoryTrace = LinearCustomMTB.updateMemoryTrace(self.memoryTrace, memoryTraceUpdate)
			if(useAutoResizeInput):
				output = LinearCustomAutoResize.autoResizeOutput(requireResizeInput, output, shapeOutput)

			return output
			
		def extra_repr(self):
			# (Optional)Set the extra information about this module. You can test
			# it by printing an object of this class.
			return 'input_features={}, output_features={}, bias={}'.format(
				self.input_features, self.output_features, self.bias is not None
			)


