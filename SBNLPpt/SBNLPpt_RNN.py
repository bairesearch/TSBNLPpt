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

def createModel(vocabSize):
	print("creating new model")
	config = SBNLPpt_RNNmodel.RNNconfig(
		vocabSize=vocabSize,
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
	model = pt.load(modelPathNameFull)
	return model
	
def saveModel(model):
	pt.save(model, modelPathNameFull)

def propagate(device, model, tokenizer, batch):
	inputIDs = batch['inputIDs'].to(device)
	attentionMask = batch['attentionMask'].to(device)
	labels = batch['labels'].to(device)
	
	loss, outputs = model(labels, device)	#incomplete (must predict next token)
	
	if(SBNLPpt_RNNmodel.calculateVocabPredictionHeadLoss):
		predictionMask = attentionMask	#CHECKTHIS #incomplete
		accuracy = SBNLPpt_data.getAccuracy(inputIDs, predictionMask, labels, outputs)
	else:
		accuracy = 0.0
	
	return loss, accuracy

