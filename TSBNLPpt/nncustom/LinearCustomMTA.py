"""LinearCustomMTA.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see XXXpt_globalDefs.py

# Usage:
see XXXpt_globalDefs.py

# Description:
Linear Custom Memory Trace Atrophy - deactivate unused connections

"""

from TSBNLPpt_globalDefs import *

import torch
from torch import nn
from torch.autograd import Function

def updateWeights(weight, input, output, memoryTraceAtrophyActive):
	if(memoryTraceAtrophyActive):
		input = input.float()
		input = torch.sum(input, dim=0)	#reduce along batch dim
		weight = weight.t()	#note weight.shape = [output_features, input_features]; therefore requires transpose
		inputInactive = torch.eq(input, 0)
		inputActive = torch.logical_not(inputInactive)
		inputActive = inputActive.float()
		inputInactive = inputInactive.float()
		
		if(memoryTraceAtrophyMultiplication):
			inputInactiveMod = torch.multiply(inputInactive, 1-memoryTraceAtrophyRate)
			inputMod = torch.add(inputActive, inputInactiveMod)
			inputMod = torch.unsqueeze(inputMod, dim=1)	#prepare for broadcast
			weight = torch.multiply(weight, inputMod)	#broadcasted
		else:
			inputInactiveMod = torch.multiply(inputInactive, -memoryTraceAtrophyRate)
			inputInactiveMod = torch.unsqueeze(inputInactiveMod, dim=1)	#prepare for broadcast
			weight = torch.add(weight, inputInactiveMod)	#broadcasted
	
		weight = weight.t()
		weight = torch.nn.parameter.Parameter(weight, requires_grad=True)
	return weight

