"""LinearCustomSDB.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see XXXpt_globalDefs.py

# Usage:
see XXXpt_globalDefs.py

# Description:
Linear Custom Simulated Dendritic Branches - simulates multiple independent fully connected weights per neuron

"""

from SBNLPpt_globalDefs import *

import torch
from torch import nn

def selectDendriticBranchOutput(output, requireResizeInput, shapeOutput, memoryTraceBias=False, memoryTraceUpdate=None):
	newShape = [i for i in output.shape[0:-1]] + [numberOfIndependentDendriticBranches] + [output.shape[-1]//numberOfIndependentDendriticBranches]
	output = torch.reshape(output, newShape)
	#note LinearCustom simulatedDendriticBranches implementation uses additional biases (*numberOfIndependentDendriticBranches) for simplicity, which are not theoretically necessary/useful
	outputMax = torch.max(output, dim=-2, keepdim=False)
	output = outputMax.values
	outputMaxIndices = outputMax.indices
	if(requireResizeInput):
		shapeOutput[-1] = shapeOutput[-1]//numberOfIndependentDendriticBranches
	if(memoryTraceBias):
		outputMaxOneHot = torch.nn.functional.one_hot(outputMaxIndices.detach())
		memoryTraceUpdate = memoryTraceUpdate*outputMaxOneHot
		#memoryTraceUpdate = torch.index_select(memoryTraceUpdate, 1, outputMaxIndices)
	if(normaliseActivationSparsity):
		output = nn.functional.layer_norm(output, output.shape[1:])	#CHECKTHIS: normalized_shape does not include batchSize
	return output, shapeOutput, memoryTraceUpdate
