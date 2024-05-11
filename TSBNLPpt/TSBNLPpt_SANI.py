"""TSBNLPpt_SANI.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see TSBNLPpt_main.py

# Usage:
see TSBNLPpt_main.py

# Description:
TSBNLPpt SANI

Similar to WaveNet 

"""

import torch as pt

from TSBNLPpt_globalDefs import *
import TSBNLPpt_data
import TSBNLPpt_SANImodel

def createModel(vocabSize):
	print("creating new model")
	config = TSBNLPpt_SANImodel.SANIconfig(
		vocabSize=vocabSize,
		#numberOfHiddenLayers=numberOfHiddenLayers,
		batchSize=batchSize,
		sequenceLength=sequenceMaxNumTokens,
		#bidirectional=bidirectional,
		hiddenLayerSize=hiddenLayerSize,
		embeddingLayerSize=embeddingLayerSize
	)
	model = TSBNLPpt_SANImodel.SANImodel(config)
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
	
	if(TSBNLPpt_SANImodel.calculateVocabPredictionHeadLoss):
		accuracy = TSBNLPpt_data.getAccuracy(inputIDs, predictionMask, labels, outputs)
	else:
		accuracy = 0.0
	
	return loss, accuracy

