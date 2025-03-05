"""TSBNLPpt_transformer.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see TSBNLPpt_main.py

# Usage:
see TSBNLPpt_main.py

# Description:
TSBNLPpt transformer

- recursiveLayers:numberOfHiddenLayers=6 supports approximately 2^6 tokens per sentence (contextual window = 512 tokens)

See RobertaForMaskedLM tutorial; 
	https://huggingface.co/blog/how-to-train
	https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6
	
"""

import torch as pt

from TSBNLPpt_globalDefs import *
import TSBNLPpt_data
from transformers import RobertaConfig
if(useMaskedLM):
	if(usePretrainedModelDebug):
		from transformers import RobertaForMaskedLM as RobertaLM
	else:
		from TSBNLPpt_transformerModel import RobertaForMaskedLM as RobertaLM
else:
	if(usePretrainedModelDebug):
		from transformers import RobertaForCausalLM as RobertaLM
	else:
		from TSBNLPpt_transformerModel import RobertaForCausalLM as RobertaLM
import TSBNLPpt_transformerModel
import TSBNLPpt_transformerTokenMemoryBank
if(transformerPOSembeddings):
	import TSBNLPpt_POSwordLists
if(detectLocalConceptColumns):
	import TSBNLPpt_transformerConceptColumns
	
if(transformerPOSembeddings):
	def preparePOSdictionary():
		TSBNLPpt_transformerModel.preparePOSdictionary()
		
if(not usePretrainedModelDebug):
	from TSBNLPpt_transformerModel import getMaxPositionEmbedding
	if(relativeTimeEmbeddings):
		maxPositionEmbeddings = getMaxPositionEmbedding(sequenceRegisterMaxTokenTime)
	else:
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
	localConceptColumnExpertsTotal = None
	localConceptColumnExpertsIntermediateSize = None
	if(detectLocalConceptColumns):
		localConceptColumnExpertsTotal = TSBNLPpt_transformerConceptColumns.initialise_dictionary()
		
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
		is_decoder=True,
		localConceptColumnExpertsTotal=localConceptColumnExpertsTotal,
		expert_intermediate_size=localConceptColumnExpertsIntermediateSize,
	)
	model = RobertaLM(config)
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
	
	conceptColumnStartIndices = conceptColumnEndIndices = None
	conceptColumnIDsPrev = conceptColumnIDsNext = None
	if(detectLocalConceptColumns):
		offsets = batch['offsets']	#List of tuples (start, end), not tensor
		if(localConceptColumnExperts):
			conceptColumnStartIndicesPrev, conceptColumnEndIndicesPrev, conceptColumnIDsPrev = TSBNLPpt_transformerConceptColumns.generateConceptColumnIndices(device, tokenizer, inputIDs, offsets, identify_type="identify_previous_column")
			conceptColumnStartIndicesNext, conceptColumnEndIndicesNext, conceptColumnIDsNext = TSBNLPpt_transformerConceptColumns.generateConceptColumnIndices(device, tokenizer, inputIDs, offsets, identify_type="identify_next_column")
		elif(localConceptColumnAttention):
			#this is not a perfect implementation (will not strictly/technically attend to both column tokens as they are defined in the GIAANN specification but uses an offset rule instead); localConceptColumnAttention could be upgraded to use both identify_previous_column and identify_next_column in future
			conceptColumnStartIndices, conceptColumnEndIndices, conceptColumnIDs = TSBNLPpt_transformerConceptColumns.generateConceptColumnIndices(device, tokenizer, inputIDs, offsets, identify_type="identify_both_columns")
	
	outputs = model(inputIDs, attention_mask=attentionMask, labels=labels, conceptColumnStartIndices=conceptColumnStartIndices, conceptColumnEndIndices=conceptColumnEndIndices, conceptColumnIDsPrev=conceptColumnIDsPrev, conceptColumnIDsNext=conceptColumnIDsNext)

	accuracy = TSBNLPpt_data.getAccuracy(inputIDs, attentionMask, labels, outputs)
	loss = outputs.loss
	
	return loss, accuracy


if(tokenMemoryBankStorageSelectionAlgorithmAuto):

	def getTokenMemoryBankStorageSelectionModelBatch(layerIndex):
		return TSBNLPpt_transformerTokenMemoryBank.getTokenMemoryBankStorageSelectionModelBatch(layerIndex)
		
	def createTokenMemoryBankStorageSelectionModel(modelStoreIndex):	
		print("creating new tokenMemoryBankStorageSelection model")	
		layerIndex = getLayerIndex(modelStoreIndex)
		model = TSBNLPpt_transformerTokenMemoryBank.createModelTokenMemoryBankStorageSelection(layerIndex, tokenMemoryBankStorageSelectionConfig)
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

	tokenMemoryBankStorageSelectionConfig = TSBNLPpt_transformerTokenMemoryBank.tokenMemoryBankStorageSelectionConfig(
		numberOfHiddenLayers=numberOfHiddenLayers,	#or numberOfHiddenLayersTokenMemoryBankParameters?
		inputLayerSize=tokenMemoryBankStorageSelectionModelInputLayerSize,
		hiddenLayerSize=tokenMemoryBankStorageSelectionModelHiddenLayerSize,
		outputLayerSize=tokenMemoryBankStorageSelectionModelOutputLayerSize,
	)
		

#if(transformerPOSembeddings):
#	if(GIAuseVectorisedPOSidentification):
#		def calculateXYlabels(tokenizer, batch, vocabSize):
#			return TSBNLPpt_POSembedding.calculateXYlabels(tokenizer, batch, vocabSize, posVectorList)
#	else:
#		printe("!GIAuseVectorisedPOSidentification not coded")
		
