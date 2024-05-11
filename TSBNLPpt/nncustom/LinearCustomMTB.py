"""LinearCustomMTB.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see XXXpt_globalDefs.py

# Usage:
see XXXpt_globalDefs.py

# Description:
Linear Custom Memory Trace Bias - adjust training/inference based on network prior activations

"""

from TSBNLPpt_globalDefs import *

import torch
from torch import nn
from torch.autograd import Function

if(useAutoResizeInput):
	#from https://pytorch.org/docs/master/notes/extending.html
	class LinearCustomFunctionMTB(Function):
		@staticmethod
		# ctx is the first argument to forward
		def forward(ctx, input, weight, bias, memoryTrace):
			# The forward pass can use ctx.
			ctx.save_for_backward(input, weight, bias, memoryTrace)
			output = input.mm(calculateBiasedWeight(weight, memoryTrace).t())
			if bias is not None:
				output += bias.unsqueeze(0).expand_as(output)
			return output

		@staticmethod
		def backward(ctx, grad_output):
			input, weight, bias, memoryTrace = ctx.saved_tensors
			grad_input = grad_weight = grad_bias = grad_memoryTrace = None

			if ctx.needs_input_grad[0]:
				grad_input = grad_output.mm(calculateBiasedWeight(weight, memoryTrace))
			if ctx.needs_input_grad[1]:
				grad_weight = grad_output.t().mm(input)
			if bias is not None and ctx.needs_input_grad[2]:
				grad_bias = grad_output.sum(0)

			return grad_input, grad_weight, grad_bias, grad_memoryTrace
else:
	#from https://pytorch.org/docs/master/notes/extending.html
	class LinearCustomFunctionMTB(Function):
		@staticmethod
		# ctx is the first argument to forward
		def forward(ctx, input, weight, bias, memoryTrace):
			# The forward pass can use ctx.
			ctx.save_for_backward(input, weight, bias, memoryTrace)
			output = input.matmul(calculateBiasedWeight(weight, memoryTrace).t())
			if bias is not None:
				output += bias.unsqueeze(0).expand_as(output)
			return output

		@staticmethod
		def backward(ctx, grad_output):
			input, weight, bias, memoryTrace = ctx.saved_tensors
			grad_input = grad_weight = grad_bias = grad_memoryTrace = None

			if ctx.needs_input_grad[0]:
				grad_input = grad_output.matmul(calculateBiasedWeight(weight, memoryTrace))
			if ctx.needs_input_grad[1]:
				grad_weight = grad_output.swapaxes(-1, -2).matmul(input)
			if bias is not None and ctx.needs_input_grad[2]:
				grad_bias = grad_output.sum(0)

			return grad_input, grad_weight, grad_bias, grad_memoryTrace
			
def executeAndCalculateMemoryTraceUpdate(input, weight, bias, memoryTrace):
	memoryTraceUpdate = None
	output = LinearCustomFunctionMTB.apply(input, weight, bias, memoryTrace)
	if(normaliseActivationSparsity):
		output = nn.functional.layer_norm(output, output.shape[1:])	#CHECKTHIS: normalized_shape does not include batchSize
	memoryTraceUpdate = calculateMemoryTraceUpdate(input.detach(), output.detach(), weight.detach())
	return output, memoryTraceUpdate
			
def calculateBiasedWeight(weight, memoryTrace):
	biasedWeight = weight * memoryTrace.t()	#note weight.shape = [output_features, input_features]; therefore requires transpose
	return biasedWeight

def fadeMemoryTrace(memoryTrace):
	fade = calculateMemoryTraceFade()
	memoryTrace = torch.multiply(memoryTrace, fade)
	return memoryTrace
			
def calculateMemoryTraceUpdate(input, output, weight):
	#limitations: calculateMemoryTraceUpdate expects input/output to be batched (ie first dimension is batchSize)
	
	#memoryTraceBiasSigned - calculate positive/negative memory trace: positive memory trace calculated based on true positives and true negatives, negative memory trace calculated based on false positives and false negatives

	weight = weight.t()	#note weight.shape = [output_features, input_features]; therefore requires transpose
	
	outputActivation = torch.clamp(output, min=0.0)	#nn.functional.ReLU(output)	#CHECKTHIS - assume ReLU activation function (assumption does not always hold, but approximation might be valid)
	outputDeactivation = torch.clamp(output, max=0.0)
	outputDeactivation = torch.multiply(outputDeactivation, -1)	#make values positive or 0
	
	memoryTraceUpdateExcite = torch.matmul(input.swapaxes(-1, -2), outputActivation)	#coincidenceMatrixExcite
	memoryTraceUpdateInhibit = torch.matmul(input.swapaxes(-1, -2), outputDeactivation)	#coincidenceMatrixInhibit
	
	if(input.shape[0] > 1):
		pass
		#assume input/output is not batched (eg applyIOconversionLayers)
		#NO: if batchSize > 1, memoryTraceBias will expect batched sequences to be continuous/contiguous but implementation will not gain any additional inference ability for interpreting sequences within the batch"
			#memoryTraceUpdateExcite = torch.mean(memoryTraceUpdateExcite, dim=0)
			#memoryTraceUpdateInhibit = torch.mean(memoryTraceUpdateInhibit, dim=0)
		#print("input.shape = ", input.shape)
		#print("memoryTraceBias: calculateMemoryTraceUpdate warning: (input.shape[0] > 1)")
	else:
		memoryTraceUpdateExcite = memoryTraceUpdateExcite[0]	#take only first batch sample
		memoryTraceUpdateInhibit = memoryTraceUpdateInhibit[0]	#take only first batch sample

	if(memoryTraceBiasWeightDirectionDependent):
		weightPos = torch.gt(weight, 0).float()
		if(memoryTraceBiasWeightDependent):
			weightPos = weight * weightPos
		weightNeg = torch.lt(weight, 0).float()
		if(memoryTraceBiasWeightDependent):
			weightNeg = weight * weightNeg
			weightNeg = weightNeg * -1	#make values positive or 0

		memoryTraceUpdateTruePos = memoryTraceUpdateExcite*weightPos	#true positives
		memoryTraceUpdateTrueNeg = memoryTraceUpdateInhibit*weightNeg	#true negatives
		memoryTraceUpdate = memoryTraceUpdateTruePos + memoryTraceUpdateTrueNeg	#TruePos/TrueNeg/FalsePos/FalseNeg are mutually exclusive memoryTrace indices
		if(memoryTraceBiasSigned):
			memoryTraceUpdateFalsePos = memoryTraceUpdateExcite*weightNeg	#false positives
			memoryTraceUpdateFalseNeg = memoryTraceUpdateInhibit*weightPos	#false negatives
			memoryTraceUpdate = memoryTraceUpdate - memoryTraceUpdateFalsePos - memoryTraceUpdateFalseNeg	#TruePos/TrueNeg/FalsePos/FalseNeg are mutually exclusive memoryTrace indices
	else:
		memoryTraceUpdate = memoryTraceUpdateExcite
	
	return memoryTraceUpdate
	
def updateMemoryTrace(memoryTrace, memoryTraceUpdate):
	memoryTrace = fadeMemoryTrace(memoryTrace)
	memoryTrace = addMemoryTraceUpdate(memoryTrace, memoryTraceUpdate)
	return memoryTrace
	
def addMemoryTraceUpdate(memoryTrace, memoryTraceUpdate):
	memoryTrace = memoryTrace + (memoryTraceUpdate*calculateMemoryTraceFade())
	return memoryTrace
	
def calculateMemoryTraceFade():
	fade = ((0.5) * 1/memoryTraceBiasHalflife)
	return fade
