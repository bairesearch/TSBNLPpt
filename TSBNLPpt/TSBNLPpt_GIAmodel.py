"""TSBNLPpt_GIAmodel.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see TSBNLPpt_main.py

# Usage:
see TSBNLPpt_main.py

# Description:
TSBNLPpt GIA model

based on word2vec training model

"""

import torch as pt
from torch import nn

from TSBNLPpt_globalDefs import *
if(GIAmemoryTraceAtrophy):
	from nncustom.LinearCustomMTA import updateWeights

#model architecture: vocabSize -> embeddingLayerSize -> vocabSize

EMBED_MAX_NORM = 1 	#CHECKTHIS

class GIAwordEmbeddingConfig():
	def __init__(self, vocabSize, embeddingLayerSize):
		self.vocabSize = vocabSize
		self.embeddingLayerSize = embeddingLayerSize

class GIAwordEmbeddingModel(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		#print("config.vocabSize = ", config.vocabSize)
		#print("config.embeddingLayerSize = ", config.embeddingLayerSize)
		if(GIAuseOptimisedEmbeddingLayer1):
			self.input = nn.Embedding(num_embeddings=config.vocabSize, embedding_dim=config.embeddingLayerSize, max_norm=EMBED_MAX_NORM)
		else:
			self.input = nn.Linear(in_features=config.vocabSize, out_features=config.embeddingLayerSize)	
		self.output = nn.Linear(in_features=config.embeddingLayerSize, out_features=config.vocabSize)
		self.lossFunction = nn.CrossEntropyLoss()
		self.memoryTraceAtrophyActive = True
		if(GIAuseOptimisedEmbeddingLayer2):
			self.embeddingEncoder = None	#will be generated after training the GIA model

	def forward(self, x, y):
		#print("forward")
		if(GIAuseOptimisedEmbeddingLayer1):
			x = x.long()	#or use int32	#nn.Embedding requires integer input
			y = y.long()
		embeddings = self.input(x)
		if(GIAuseOptimisedEmbeddingLayer1):
			#incomplete (results in incorrect embedding shape)
			print("x.shape = ", x.shape)
			print("embeddings.shape = ", embeddings.shape)
		if(GIAmemoryTraceAtrophy):
			with torch.no_grad():
				self.input.weight = updateWeights(self.input.weight.detach(), x.detach(), embeddings.detach(), self.memoryTraceAtrophyActive)
		#printCUDAmemory("forward2")
		outputs = self.output(embeddings)
		loss = self.lossFunction(outputs, y)
		if(debugPrintModelPropagation):
			print("self.input.weight = ", self.input.weight)
			print("self.output.weight = ", self.output.weight)
			print("x = ", x)
			print("embeddings = ", embeddings)
			print("outputs = ", outputs)

		return loss, outputs

	if(useGIAwordEmbeddings):
		def generateEmbeddingLayer(self):
			if(GIAuseOptimisedEmbeddingLayer2):
				self.embeddingEncoder = nn.Embedding(num_embeddings=self.config.vocabSize, embedding_dim=self.config.embeddingLayerSize)
				self.embeddingEncoder.weight = torch.nn.Parameter(self.input.weight.transpose(1, 0))
			else:
				self.embeddingEncoder = self.input
	else:
		#old (not used);
		if(useIndependentReverseRelationsModels):
			def getEmbeddingLayerF(self):
				return self.input
			def getEmbeddingLayerR(self):
				return torch.swapaxes(self.output)
		else:
			def getEmbeddingLayer(self):
				return self.input
