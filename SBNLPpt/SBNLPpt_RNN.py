"""SBNLPpt_RNN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SBNLPpt_main.py

# Usage:
see SBNLPpt_main.py

# Description:
SBNLPpt RNN

"""

import torch as pt

from SBNLPpt_globalDefs import *
import SBNLPpt_data
import SBNLPpt_RNNmodel

hiddenLayerSize = 1024	#65536	#2^16 - large hidden size is required for recursive RNN as parameters are shared across a) sequence length and b) number of layers
if(SBNLPpt_RNNmodel.applyIOconversionLayers):
	embeddingLayerSize = 768
else:
	embeddingLayerSize = hiddenLayerSize

numberOfHiddenLayers = 6

modelPathName = modelFolderName + '/modelRNN.pt'

useBidirectionalRNN = False
if(useBidirectionalRNN):
	bidirectional = 2
else:
	bidirectional = 1

def createModel():
	print("creating new model")
	config = SBNLPpt_RNNmodel.RNNconfig(
		vocabularySize=vocabularySize,
		numberOfHiddenLayers=numberOfHiddenLayers,
		batchSize=batchSize,
		sequenceLength=sequenceMaxNumTokens,
		bidirectional=bidirectional,
		hiddenLayerSize=hiddenLayerSize,
		embeddingLayerSize=embeddingLayerSize
	)
	model = SBNLPpt_RNNmodel.RNNmodel(config)
	return model

def loadModel():
	print("loading existing model")
	model = pt.load(modelPathName)
	return model
	
def saveModel(model):
	pt.save(model, modelPathName)

def propagate(device, model, tokenizer, batch):
	inputIDs = batch['inputIDs'].to(device)
	attentionMask = batch['attentionMask'].to(device)
	labels = batch['labels'].to(device)
	
	loss, outputs = model(labels, device)	#incomplete (must predict next token)
	
	if(SBNLPpt_RNNmodel.calculateVocabPredictionHeadLoss):
		predictionMask = attentionMask	#CHECKTHIS #incomplete
		accuracy = SBNLPpt_data.getAccuracy(tokenizer, inputIDs, predictionMask, labels, outputs)
	else:
		accuracy = 0.0
	
	return loss, accuracy

