"""TSBNLPpt_POSembedding.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see TSBNLPpt_main.py

# Usage:
see TSBNLPpt_main.py

# Description:
TSBNLPpt POS embedding
	
"""

import torch as pt
from torch import nn
import torch

from TSBNLPpt_globalDefs import *
import TSBNLPpt_POSwordLists

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def preparePOSdictionary():
	if(GIAuseVectorisedPOSidentification):
		global posVectorList
		global numberOfVectorSpaces
		posVectorList = TSBNLPpt_POSwordLists.loadPOSwordListVectors(TSBNLPpt_POSwordLists.wordListVectorsDictAll)
		numberOfVectorSpaces = len(TSBNLPpt_POSwordLists.wordListVectorsDictAll)
	else:
		printe("!GIAuseVectorisedPOSidentification not coded")
			
class POSwordEmbeddingEncoderClass(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.wordEmbeddingsTrainable = nn.Embedding(config.vocab_size, trainableHiddenSize, padding_idx=config.pad_token_id)	#trainable embedding layer (for non relation type tokens; e.g. nouns)
					
	def forward(self, x):
		with torch.no_grad(): 
			batchTokenIDs = x
			batchPOSflagsList = prepareBatchPOSflagsList(batchTokenIDs, posVectorList)
			batchPOSflags = torch.stack(batchPOSflagsList, dim=0).to(device)	#shape: POSembeddingSize * batchSize * sequenceLength 
			batchPOSflags = torch.permute(batchPOSflags, (1, 2, 0))		#shape: batchSize * sequenceLength * POSembeddingSize
			outputPretrained = batchPOSflags		#shape: batchSize * sequenceLength * pretrainedHiddenSize
		outputTrainable = self.wordEmbeddingsTrainable(x)	#shape: batchSize * sequenceLength * trainableHiddenSize
		outputs = torch.cat([outputPretrained, outputTrainable], dim=-1)		#shape: batchSize * sequenceLength * hiddenLayerSizeTransformer
		return outputs

def prepareBatchPOSflagsList(batchTokenIDs, posVectorList):
	batchPOSflagsList = []
	for POSindex, POSitem in enumerate(TSBNLPpt_POSwordLists.wordListVectorsDictAll.items()):
		posVector = posVectorList[POSindex]	
		batchPOSflags = torch.zeros(batchTokenIDs.shape)	#dtype=torch.bool
		for sampleIndex in range(batchSize):
			for tokenIndex in range(sequenceMaxNumTokens):
				sampleToken = batchTokenIDs[sampleIndex][tokenIndex]	#token ID
				posFlag = posVector[sampleToken]	#.bool()
				batchPOSflags[sampleIndex][tokenIndex] = posFlag
		batchPOSflagsList.append(batchPOSflags)
	return batchPOSflagsList
