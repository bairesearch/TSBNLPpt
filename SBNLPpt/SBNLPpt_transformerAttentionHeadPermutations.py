"""SBNLPpt_transformerAttentionHeadPermutations.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SBNLPpt_main.py

# Usage:
see SBNLPpt_main.py

# Description:
SBNLPpt transformer attention head permutations

"""

import torch
from torch import nn
from SBNLPpt_globalDefs import *

def applySelfAttention(self, hidden_states, attention_mask, head_mask, query_layer, key_layer, value_layer):

	if(transformerAttentionHeadPermutationsType=="independent"):
		query_layerP = query_layer
		key_layerP = key_layer

	if(transformerAttentionHeadPermutations):
		batchSize = query_layer.shape[0]
		sequenceLength = query_layer.shape[2]
		if(transformerAttentionHeadPermutationsType=="dependent"):
			#shape: batchSize, 1, sequenceLength*numberAttentionHeads, attention_head_size	#emulate single headed attention
			query_layer = query_layer.reshape(batchSize, 1, sequenceLength*self.num_attention_heads, self.attention_head_size)
			key_layer = key_layer.reshape(batchSize, 1, sequenceLength*self.num_attention_heads, self.attention_head_size)
			value_layer = value_layer.reshape(batchSize, 1, sequenceLength*self.num_attention_heads, self.attention_head_size)
		elif(transformerAttentionHeadPermutationsType=="independent"):
			#shape: batchSize, numberAttentionHeads*numberAttentionHeads, sequenceLength, attention_head_size
			query_layer = query_layer.repeat(1, self.num_attention_heads, 1, 1)	
			key_layer = torch.repeat_interleave(key_layer, repeats=self.num_attention_heads, dim=1)

	# Take the dot product between "query" and "key" to get the raw attention scores.
	attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

	if(transformerAttentionHeadPermutationsType=="independent"):
		relative_position_scores = self.calculateRelativePositionScores(hidden_states, query_layerP, key_layerP)
	else:
		relative_position_scores = self.calculateRelativePositionScores(hidden_states, query_layer, key_layer)

	if(transformerAttentionHeadPermutationsType=="independent"):
		relative_position_scores = relative_position_scores.repeat(1, self.num_attention_heads, 1, 1)
	attention_scores = attention_scores + relative_position_scores	

	attention_scores = attention_scores / math.sqrt(self.attention_head_size)

	if attention_mask is not None:
		if(transformerAttentionHeadPermutationsType=="dependent"):
			attention_mask = attention_mask.repeat(1, 1, 1, self.num_attention_heads)
		# Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
		attention_scores = attention_scores + attention_mask

	# Normalize the attention scores to probabilities.
	attention_probs = nn.functional.softmax(attention_scores, dim=-1)

	if(transformerAttentionHeadPermutationsType=="independent"):
		attention_probs = attention_probs.view(batchSize, self.num_attention_heads, self.num_attention_heads, sequenceLength, sequenceLength)
		attention_probs = torch.topk(attention_probs, k=1, dim=2).values
		attention_probs = torch.squeeze(attention_probs, dim=2)

	if(tokenMemoryBank):
		self.tokenMemoryBankClass.calculateAttentionProbsMaxIndex(attention_scores.detach())

	# This is actually dropping out entire tokens to attend to, which might
	# seem a bit unusual, but is taken from the original Transformer paper.
	attention_probs = self.dropout(attention_probs)

	# Mask heads if we want to
	if head_mask is not None:
		if(transformerAttentionHeadPermutationsType=="dependent"):
			head_mask = head_mask.repeat(1, 1, self.num_attention_heads, 1)
		attention_probs = attention_probs * head_mask

	context_layer = torch.matmul(attention_probs, value_layer)
	if(transformerAttentionHeadPermutationsType=="dependent"):
		context_layer = context_layer.view(batchSize, self.num_attention_heads, sequenceLength, self.attention_head_size)

	return attention_probs, context_layer
