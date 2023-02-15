"""SBNLPpt_SANI.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SBNLPpt_main.py

# Usage:
see SBNLPpt_main.py

# Description:
SBNLPpt SANI

Similar to WaveNet 

"""

import torch as pt

from SBNLPpt_globalDefs import *
import SBNLPpt_data
import SBNLPpt_SANImodel

def createModel(vocabSize):
	print("creating new model")
	config = SBNLPpt_SANImodel.SANIconfig(
		vocabSize=vocabSize,
		#numberOfHiddenLayers=numberOfHiddenLayers,
		batchSize=batchSize,
		sequenceLength=sequenceMaxNumTokens,
		#bidirectional=bidirectional,
		hiddenLayerSize=hiddenLayerSize,
		embeddingLayerSize=embeddingLayerSize
	)
	model = SBNLPpt_SANImodel.SANImodel(config)
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
	
	loss, outputs, predictionMask = model(labels, attentionMask, device)
	
	if(SBNLPpt_SANImodel.calculateVocabPredictionHeadLoss):
		accuracy = SBNLPpt_data.getAccuracy(tokenizer, inputIDs, predictionMask, labels, outputs)
	else:
		accuracy = 0.0
	
	return loss, accuracy

