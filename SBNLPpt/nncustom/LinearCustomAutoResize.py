"""LinearCustomAutoResize.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see XXXpt_globalDefs.py

# Usage:
see XXXpt_globalDefs.py

# Description:
Linear Auto Resize

"""

import torch

def autoResizeInput(input, weight):
	requireBroadcasting = False
	shapeOutput = None
	numberOfDimensionsToCollapse = len(input.shape) - len(weight.shape)
	if(numberOfDimensionsToCollapse > 0):
		requireBroadcasting = True
		sizeOfDimensionsToCollapse = 1
		shapeInput = []
		shapeOutput = []
		for inputDimensionIndex in range(len(input.shape)):
			inputDimensionSize = input.shape[inputDimensionIndex]
			if(inputDimensionIndex <= numberOfDimensionsToCollapse):
				sizeOfDimensionsToCollapse = sizeOfDimensionsToCollapse*inputDimensionSize
				if(inputDimensionIndex == numberOfDimensionsToCollapse):
					shapeInput.append(sizeOfDimensionsToCollapse)

				shapeOutput.append(inputDimensionSize)
			else:
				shapeInput.append(inputDimensionSize)
		for outputDimensionIndex in range(len(weight.shape)-1):	#CHECKTHIS for 3d weight matrices
			weightDimensionSize = weight.shape[outputDimensionIndex]
			shapeOutput.append(weightDimensionSize)
		input = torch.reshape(input, shapeInput)	#torch.broadcast_to(input, shapeInput)
	return input, requireBroadcasting, shapeOutput
	
def autoResizeOutput(requireBroadcasting, output, shapeOutput):
	if(requireBroadcasting):
		output = torch.reshape(output, shapeOutput)
	return output

