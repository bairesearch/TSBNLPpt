"""SBNLPpt_transformer.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SBNLPpt_main.py

# Usage:
see SBNLPpt_main.py

# Description:
SBNLPpt transformer

- recursiveLayers:numberOfHiddenLayers=6 supports approximately 2^6 tokens per sentence (contextual window = 512 tokens)

See RobertaForMaskedLM tutorial; 
	https://huggingface.co/blog/how-to-train
	https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6
	
"""

import torch as pt

from SBNLPpt_globalDefs import *
import SBNLPpt_data
from transformers import RobertaConfig
if(usePretrainedModelDebug):
	from transformers import RobertaForMaskedLM
else:
	from SBNLPpt_transformerModel import RobertaForMaskedLM
import SBNLPpt_transformerModel

if(not usePretrainedModelDebug):
	from SBNLPpt_transformerModel import getMaxPositionEmbedding
	if(relativeTimeEmbeddings):
		positionEmbeddingType = "relative_time"	#calculates relative time between layer tokens
		maxPositionEmbeddings = getMaxPositionEmbedding(sequenceRegisterMaxTokenTime)
	else:
		positionEmbeddingType = "relative_key"	#"absolute"	#default
		maxPositionEmbeddings = getMaxPositionEmbedding(sequenceMaxNumTokens)

if(useMultipleModels):
	def createModelIndex(vocabSize, modelStoreIndex):
		if(modelStoreIndex == 0):
			model = createModel(vocabSize)
		else:
			model = createTokenMemoryBankStorageSelectionModel(modelStoreIndex)
		return model
	def loadModelIndex(modelStoreIndex):
		if(modelStoreIndex == 0):
			model = loadModel()
		else:
			model = loadTokenMemoryBankStorageSelectionModel(modelStoreIndex)
		return model
	def saveModelIndex(model, modelStoreIndex):
		if(modelStoreIndex == 0):
			saveModel(model)
		else:		
			saveTokenMemoryBankStorageSelectionModel(model, modelStoreIndex)
		
def createModel(vocabularySize):	
	print("creating new model")	
	config = RobertaConfig(
		vocab_size=vocabularySize,  #sync with tokenizer vocab_size
		max_position_embeddings=maxPositionEmbeddings,
		hidden_size=hiddenLayerSize,
		num_attention_heads=numberOfAttentionHeads,
		num_hidden_layers=numberOfHiddenLayers,
		intermediate_size=intermediateSize,
		type_vocab_size=1,
		position_embedding_type=positionEmbeddingType,
	)
	model = RobertaForMaskedLM(config)
	return model

def loadModel():
	print("loading existing model")
	model = RobertaForMaskedLM.from_pretrained(modelPathName, local_files_only=True)
	return model
	
def saveModel(model):
	model.save_pretrained(modelPathName)

def propagateIndex(device, model, tokenizer, batch, modelStoreIndex):	
	if(modelStoreIndex == 0):
		loss, accuracy = propagate(device, model, tokenizer, batch)
		result = True
	else:
		loss, accuracy, result = propagateTokenMemoryBankStorageSelection(device, model, tokenizer, batch)
	return loss, accuracy, result
			
def propagate(device, model, tokenizer, batch):	
	inputIDs = batch['inputIDs'].to(device)
	attentionMask = batch['attentionMask'].to(device)
	labels = batch['labels'].to(device)
	
	if(tokenMemoryBank):
		attentionMaskMemoryBank = pt.ones((batchSize, sequenceRegisterMemoryBankLength)).to(device)
		attentionMask = pt.cat((attentionMask, attentionMaskMemoryBank), dim=1)
		
	outputs = model(inputIDs, attention_mask=attentionMask, labels=labels)

	predictionMask = pt.where(inputIDs==customMaskTokenID, 1.0, 0.0)	#maskTokenIndexFloat = maskTokenIndex.float()	
	accuracy = SBNLPpt_data.getAccuracy(tokenizer, inputIDs, predictionMask, labels, outputs.logits)
	loss = outputs.loss
	
	return loss, accuracy

if(tokenMemoryBankStorageSelectionAlgorithmAuto):

	def getTokenMemoryBankStorageSelectionModelBatch(layerIndex):
		return SBNLPpt_transformerModel.getTokenMemoryBankStorageSelectionModelBatch(layerIndex)
		
	def createTokenMemoryBankStorageSelectionModel(modelStoreIndex):	
		print("creating new tokenMemoryBankStorageSelection model")	
		layerIndex = getLayerIndex(modelStoreIndex)
		model = SBNLPpt_transformerModel.createModelTokenMemoryBankStorageSelection(layerIndex, tokenMemoryBankStorageSelectionConfig)
		return model

	def loadTokenMemoryBankStorageSelectionModel(modelStoreIndex):
		print("loading existing tokenMemoryBankStorageSelection model")
		modelPathNameFull = getTokenMemoryBankStorageSelectionModelPathName(modelStoreIndex)
		model = pt.load(modelPathNameFull)
		return model
		
	def saveTokenMemoryBankStorageSelectionModel(model, modelStoreIndex):
		#print("save tokenMemoryBankStorageSelection model")
		#modelStoreIndex = model.modelStoreIndex
		modelPathNameFull = getTokenMemoryBankStorageSelectionModelPathName(modelStoreIndex)
		pt.save(model, modelPathNameFull)
		
	def getTokenMemoryBankStorageSelectionModelName(modelStoreIndex):
		modelName = tokenMemoryBankStorageSelectionModelName + str(modelStoreIndex)
		return modelName

	def getTokenMemoryBankStorageSelectionModelPathName(modelStoreIndex):
		modelName = getTokenMemoryBankStorageSelectionModelName(modelStoreIndex)
		modelPathNameFull = getModelPathNameFull(modelPathName, modelName)
		return modelPathNameFull
		
	def getModelStoreIndex(layerIndex):
		modelStoreIndex = layerIndex+1	#0: modelTransformer, 1+: modelTokenMemoryBankStorageSelection*
		return modelStoreIndex

	def getLayerIndex(modelStoreIndex):
		layerIndex = modelStoreIndex-1	#0: modelTransformer, 1+: modelTokenMemoryBankStorageSelection*
		return layerIndex

	def propagateTokenMemoryBankStorageSelection(device, model, tokenizer, batch):	
		result = True
		if(batch['xLabels'].shape[0] > 0):	#ensure there are samples to propagate
			xLabels = batch['xLabels'].to(device)
			yLabels = batch['yLabels'].to(device)
			y = model(xLabels)
			loss = model.lossFunction(y, yLabels)
			accuracy = model.accuracyMetric(y, yLabels)	#CHECKTHIS: calculate top-1 accuracy	#threshold = 0.5
			accuracy = accuracy.cpu().numpy()
		else:
			loss = 0.0
			accuracy = 0.0
			result = False
		return loss, accuracy, result

	class modelStoreClass():
		def __init__(self, name):
			self.name = name
			#self.config = None
			self.model = None
			self.optim = None

	modelStoreList = [None]*(1+numberOfHiddenLayersTokenMemoryBankParameters)
	modelStoreList[0] = modelStoreClass(modelName)
	for layerIndex in range(numberOfHiddenLayersTokenMemoryBankParameters):
		modelStoreIndex = getModelStoreIndex(layerIndex)
		modelStoreList[modelStoreIndex] = modelStoreClass(getTokenMemoryBankStorageSelectionModelName(modelStoreIndex))

	tokenMemoryBankStorageSelectionConfig = SBNLPpt_transformerModel.tokenMemoryBankStorageSelectionConfig(
		numberOfHiddenLayers=numberOfHiddenLayers,	#or numberOfHiddenLayersTokenMemoryBankParameters?
		inputLayerSize=tokenMemoryBankStorageSelectionModelInputLayerSize,
		hiddenLayerSize=tokenMemoryBankStorageSelectionModelHiddenLayerSize,
		outputLayerSize=tokenMemoryBankStorageSelectionModelOutputLayerSize,
	)
		
