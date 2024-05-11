"""TSBNLPpt_transformerRecursiveLayers.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see TSBNLPpt_main.py

# Usage:
see TSBNLPpt_main.py

# Description:
TSBNLPpt transformer recursive layers

"""

import torch
from torch import nn
import nncustom

from TSBNLPpt_globalDefs import *

def declareRecursiveLayers(self, config, RobertaLayer):
	
	if(transformerSuperblocks or recursiveLayersEmulateOrigImplementation2):
		self.superblocksList = nn.ModuleList()
		if(transformerSuperblocksLayerNorm):
			if(transformerSuperblocksLayerNormList):
				self.superblockLayerNormList = nn.ModuleList()
			else:
				self.superblockLayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

	for superblockIndex in range(transformerSuperblocksNumber):
		if(recursiveLayersOrigImplementation):
			if(recursiveLayers):
				if(sharedLayerWeights):
					robertaSharedLayerModules = RobertaSharedLayerModules(config)
					layerList = nn.ModuleList([RobertaLayer(config, robertaSharedLayerModules) for _ in range(config.num_hidden_layers)])
				else:
					if(recursiveLayersEvalOverride):
						numberUniqueLayers = recursiveLayersNumberIterationsEvalOverride
					else:
						numberUniqueLayers = config.num_hidden_layers
					self.recursiveLayer = RobertaLayer(config)
					layerList = nn.ModuleList([self.recursiveLayer for _ in range(numberUniqueLayers)])
			else:
				layerList = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])
		else:
			if(recursiveLayersEmulateOrigImplementation):
				#depreciated (use recursiveLayersOrigImplementation instead)
				numberUniqueLayers = 1
			else:
				numberUniqueLayers = config.num_hidden_layers
			if(sharedLayerWeights):
				#depreciated (use recursiveLayersOrigImplementation instead)
				robertaSharedLayerModules = RobertaSharedLayerModules(config)
				layerList = nn.ModuleList([RobertaLayer(config, robertaSharedLayerModules) for layerIndex in range(numberUniqueLayers)])
			else:
				layerList = nn.ModuleList([RobertaLayer(config) for layerIndex in range(numberUniqueLayers)])

		if(transformerSuperblocks or recursiveLayersEmulateOrigImplementation2):
			self.superblocksList.append(layerList)
			if(transformerSuperblocksLayerNorm):
				if(transformerSuperblocksLayerNormList):
					superblockLayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
					self.superblockLayerNormList.append(superblockLayerNorm)
		else:
			self.layer = layerList
						
if(sharedLayerWeights):
	class RobertaSharedLayerModules:
		def __init__(self, config):
			#precalculate these parameters locally (temp);
			num_attention_heads = config.num_attention_heads
			attention_head_size = int(config.hidden_size / config.num_attention_heads)
			all_head_size = num_attention_heads * attention_head_size

			if(sharedLayerWeightsAttention):
				if(sharedLayerWeightsSelfAttention):
					self.robertaSelfAttentionSharedLayerQuery = nncustom.Linear(config.hidden_size, all_head_size)
					self.robertaSelfAttentionSharedLayerKey = nncustom.Linear(config.hidden_size, all_head_size)
					self.robertaSelfAttentionSharedLayerValue = nncustom.Linear(config.hidden_size, all_head_size)
				if(sharedLayerWeightsSelfOutput):
					self.robertaSelfOutputSharedLayerDense = nncustom.Linear(config.hidden_size, config.hidden_size) 
			if(sharedLayerWeightsMLP):
				if(sharedLayerWeightsIntermediate):
					self.robertaIntermediateSharedLayerDense = nncustom.Linear(config.hidden_size, config.intermediate_size)
				if(sharedLayerWeightsOutput):
					self.robertaOutputSharedLayerDense = nncustom.Linear(config.intermediate_size, config.hidden_size)

