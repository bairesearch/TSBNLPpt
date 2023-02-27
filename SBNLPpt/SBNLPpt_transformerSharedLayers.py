"""SBNLPpt_transformerSharedLayers.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SBNLPpt_main.py

# Usage:
see SBNLPpt_main.py

# Description:
SBNLPpt transformer shared layers

"""

from SBNLPpt_globalDefs import *

if(sharedLayerWeights):
	class RobertaSharedLayerModules:
		def __init__(self, config):
			#precalculate these parameters locally (temp);
			num_attention_heads = config.num_attention_heads
			attention_head_size = int(config.hidden_size / config.num_attention_heads)
			all_head_size = num_attention_heads * attention_head_size

			self.robertaSelfAttentionSharedLayerQuery = nncustom.Linear(config.hidden_size, all_head_size)
			self.robertaSelfAttentionSharedLayerKey = nncustom.Linear(config.hidden_size, all_head_size)
			self.robertaSelfAttentionSharedLayerValue = nncustom.Linear(config.hidden_size, all_head_size)
			self.robertaIntermediateSharedLayerDense = nncustom.Linear(config.hidden_size, config.intermediate_size)
			if(sharedLayerWeightsOutput):
				self.RobertaOutputSharedLayerOutput = nncustom.Linear(config.intermediate_size, config.hidden_size)
				self.RobertaSelfOutputSharedLayerOutput = nncustom.Linear(config.hidden_size, config.hidden_size) 

