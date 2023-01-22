"""SBNLPpt_GIAmodel.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SBNLPpt_main.py

# Usage:
see SBNLPpt_main.py

# Description:
SBNLPpt GIA model

based on word2vec training model

"""

import torch as pt
from torch import nn

from SBNLPpt_globalDefs import *

#vocabSize -> embeddingLayerSize -> vocabSize

#EMBED_MAX_NORM = 1 	#CHECKTHIS

class GIAconfig():
	def __init__(self, vocabSize, embeddingLayerSize):
		self.vocabSize = vocabSize
		self.embeddingLayerSize = embeddingLayerSize

class GIAmodel(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.input = nn.Linear(in_features=config.vocabSize, out_features=config.embeddingLayerSize)	#nn.Embedding(num_embeddings=config.vocabSize, embedding_dim=config.embeddingLayerSize, max_norm=EMBED_MAX_NORM)  
		self.output = nn.Linear(in_features=config.embeddingLayerSize, out_features=config.vocabSize)
		self.lossFunction = nn.CrossEntropyLoss()
			
	def forward(self, x, y):
		embeddings = self.input(x)
		outputs = self.output(embeddings)
		loss = self.lossFunction(outputs, y)
		if(debugPrintModelPropagation):
			print("self.input.weight = ", self.input.weight)
			print("self.output.weight = ", self.output.weight)
			print("x = ", x)
			print("embeddings = ", embeddings)
			print("outputs = ", outputs)
		
		return loss, outputs

	if(useIndependentReverseRelationsModels):
		def getEmbeddingLayerF():
			return self.input
		def getEmbeddingLayerR():
			return torch.swapaxes(self.output)
	else:
		def getEmbeddingLayer():
			return self.input
