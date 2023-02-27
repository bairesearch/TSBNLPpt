"""SBNLPpt_GIAembedding.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SBNLPpt_main.py

# Usage:
see SBNLPpt_main.py

# Description:
SBNLPpt GIA embedding (used by SBNLPpt_transformerModel to access pretrained GIA embedding encoder) 

"""

import torch as pt
from torch import nn

from SBNLPpt_globalDefs import *
import SBNLPpt_GIA
import SBNLPpt_GIAdefinePOSwordLists

class GIAwordEmbeddingEncoderClass(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.modelStoreList = SBNLPpt_GIAdefinePOSwordLists.vectorSpaceList
		for modelStoreIndex, modelStore in enumerate(self.modelStoreList):
			modelStore.model = SBNLPpt_GIA.loadModelIndex(modelStoreIndex)
			modelStore.model.input.requires_grad_(False)	#disable backprop for pretrained word embeddings
			modelStore.model.generateEmbeddingLayer()
			if(GIAuseOptimisedEmbeddingLayer):
				#note nn.Embedding weights are defined inverted with respect to nn.Linear weights?
				assert modelStore.model.embeddingEncoder.weight.shape[0] == config.vocab_size
				assert modelStore.model.embeddingEncoder.weight.shape[1] ==  config.hidden_size//embeddingListLen	
			else:
				assert modelStore.model.embeddingEncoder.weight.shape[0] == config.hidden_size//embeddingListLen
				assert modelStore.model.embeddingEncoder.weight.shape[1] == config.vocab_size
		if(GIAgenerateUniqueWordVectorsForRelationTypes):
			trainableHiddenSize = config.hidden_size//trainableEmbeddingSpaceFraction
			self.wordEmbeddingsNonRelationTypes = nn.Embedding(config.vocab_size, trainableHiddenSize, padding_idx=config.pad_token_id)	#trainable embedding layer (for non relation type tokens; e.g. nouns)
					
	def forward(self, x):
		with torch.no_grad(): 
			outputPretrainedList = []	#embeddingListLen of: batchSize * sequenceLength * embeddingLayerSize (hiddenLayerSizeTransformer/embeddingListLen)
			for modelStoreIndex, modelStore in enumerate(self.modelStoreList):
				embeddingEncoder = modelStore.model.embeddingEncoder #embedding layer	#shape: batchSize * sequenceLength * vocabSize
				#print("x.shape = ", x.shape)	
				output = embeddingEncoder(x)	#shape: batchSize * sequenceLength * embeddingLayerSize (hiddenLayerSizeTransformer/embeddingListLen)
				#print("output.shape = ", output.shape)	
				outputPretrainedList.append(output)
			outputPretrained = torch.stack(outputPretrainedList, dim=-2)	#shape: batchSize * sequenceLength * embeddingListLen * embeddingLayerSize
			#print("outputPretrained.shape = ", outputPretrained.shape)	
			outputPretrained = torch.flatten(outputPretrained, start_dim=-2, end_dim=-1)		#shape: batchSize * sequenceLength * pretrainedHiddenSize
			#print("outputPretrained.shape = ", outputPretrained.shape)	
		if(GIAgenerateUniqueWordVectorsForRelationTypes):
			outputTrainable = self.wordEmbeddingsNonRelationTypes(x)	#shape: batchSize * sequenceLength * trainableHiddenSize
		outputs = torch.cat([outputPretrained, outputTrainable], dim=-1)		#shape: batchSize * sequenceLength * hiddenLayerSizeTransformer
		#print("outputs.shape = ", outputs.shape)
		return outputs


	
