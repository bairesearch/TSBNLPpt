"""SBNLPpt_GIA.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SBNLPpt_main.py

# Usage:
see SBNLPpt_main.py

# Description:
SBNLPpt GIA
	
"""

import torch as pt
import torch

from SBNLPpt_globalDefs import *
import SBNLPpt_GIAmodel
import torch.nn.functional as F

import SBNLPpt_GIAvectorSpaces
if(GIAuseVectorisedSemanticRelationIdentification):
	import SBNLPpt_GIAsemanticRelationVectorised
else:
	import SBNLPpt_GIAsemanticRelationStandard


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def preparePOSdictionary():
	if(GIAuseVectorisedSemanticRelationIdentification):
		global posVectorList
		posVectorList = SBNLPpt_GIAvectorSpaces.loadPOSwordListVectors()
		numberOfVectorSpaces = len(SBNLPpt_GIAvectorSpaces.vectorSpaceList)
	else:
		SBNLPpt_GIAsemanticRelationStandard.preparePOSdictionary()
	

#if(useMultipleModels):
def createModelIndex(vocabSize, modelStoreIndex):
	return createModel(vocabSize)
def loadModelIndex(modelStoreIndex):
	modelNameIndex = getModelNameIndex(modelStoreIndex)
	modelPathNameFull = getModelPathNameFull(modelPathName, modelNameIndex)
	return loadModel(modelPathNameFull)
def saveModelIndex(model, modelStoreIndex):
	modelNameIndex = getModelNameIndex(modelStoreIndex)
	modelPathNameFull = getModelPathNameFull(modelPathName, modelNameIndex)
	pt.save(model, modelPathNameFull)
def getModelNameIndex(modelStoreIndex):
	modelNameIndex = GIAmodelName + str(modelStoreIndex)
	return modelNameIndex
			
def createModel(vocabSize):
	print("creating new model")
	config = SBNLPpt_GIAmodel.GIAwordEmbeddingConfig(vocabSize, embeddingLayerSize)
	model = SBNLPpt_GIAmodel.GIAwordEmbeddingModel(config)
	return model

def loadModel(modelPathNameFull):
	print("loading existing model")
	model = pt.load(modelPathNameFull)
	return model

def saveAllModels():
	for modelStoreIndex in range(vectorSpaceListLen):
		saveModelIndex(model, modelStoreIndex)
			
def saveModel(model, modelPathNameFull):
	pt.save(model, modelPathNameFull)

def propagateIndex(device, model, tokenizer, batch, modelStoreIndex):
	loss, accuracy = propagate(device, model, tokenizer, batch)
	result = True
	return loss, accuracy, result
	 
def propagate(device, model, tokenizer, labels):
	(xLabels, yLabels) = labels
	loss, outputs = model(xLabels, yLabels)
	accuracy = 0
	return loss, accuracy


if(GIAuseVectorisedSemanticRelationIdentification):
	def calculateXYlabels(tokenizer, batch, vocabSize):
		return SBNLPpt_GIAsemanticRelationVectorised.calculateXYlabels(tokenizer, batch, vocabSize, posVectorList)
else:
	def calculateXYlabels(tokenizer, vectorSpace, vectorSpaceIndex, batch, vocabSize):
		return SBNLPpt_GIAsemanticRelationStandard.calculateXYlabels(tokenizer, vectorSpace, vectorSpaceIndex, batch, vocabSize)
	

