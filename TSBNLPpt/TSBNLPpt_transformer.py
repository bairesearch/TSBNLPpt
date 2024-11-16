"""TSBNLPpt_transformer.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2024 Baxter AI (baxterai.com)

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
if(attendToLocalConceptColumns):
	import spacy
	nlp = spacy.load('en_core_web_md')	#spacy.load('en_core_web_lg')
	
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
	if(attendToLocalConceptColumns):
		offsets = batch['offsets']	#List of tuples (start, end), not tensor
		conceptColumnStartIndices, conceptColumnEndIndices = generateConceptColumnIndices(device, tokenizer, inputIDs, offsets)
	
	outputs = model(inputIDs, attention_mask=attentionMask, labels=labels, conceptColumnStartIndices=conceptColumnStartIndices, conceptColumnEndIndices=conceptColumnEndIndices)

	accuracy = TSBNLPpt_data.getAccuracy(inputIDs, attentionMask, labels, outputs)
	loss = outputs.loss
	
	return loss, accuracy

def generateConceptColumnIndices(device, tokenizer, batch_input_ids, batch_offsets):
	noun_pos_tags = {'NOUN', 'PROPN'}
	non_noun_pos_tags = {'ADJ', 'ADV', 'VERB', 'ADP', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NUM', 'PART', 'PRON', 'SCONJ', 'SYM', 'X'}

	batch_concept_start_indices = []
	batch_concept_end_indices = []
	
	batch_size = len(batch_input_ids)
	for batch_index in range(batch_size):
		input_ids_sample = batch_input_ids[batch_index]
		offsets_sample = batch_offsets[batch_index]
		
		tokens = tokenizer.decode(input_ids_sample, skip_special_tokens=True)	
	
		  # Process the text with spaCy
		doc = nlp(tokens)

		# Create a mapping from character positions to spaCy tokens
		char_to_token = []
		for idx, (start_char, end_char) in enumerate(offsets_sample):
			if start_char == end_char:
				# Special tokens (e.g., <s>, </s>)
				char_to_token.append(None)
			else:
				# Find the corresponding spaCy token
				for token in doc:
					if token.idx <= start_char and token.idx + len(token) >= end_char:
						char_to_token.append(token)
						break
				else:
					char_to_token.append(None)

		# Initialize lists for concept columns
		conceptColumnStartIndexList = []
		conceptColumnEndIndexList = []

		# Initialize variables
		seq_length = input_ids_sample.shape[0]
		conceptColumnStartIndexList.append(0)
		concept_indices = []  # Indices of concept words in tokens

		for idx in range(seq_length):
			token = char_to_token[idx]
			if token is not None:
				pos = token.pos_
				if pos in noun_pos_tags:
					concept_indices.append(idx)
					if len(conceptColumnStartIndexList) > 1:
						conceptColumnEndIndexList.append(idx - 1)
					conceptColumnStartIndexList.append(idx + 1)

		# get first_pad_index;
		pad_token_id = tokenizer.pad_token_id	#default=1 #https://huggingface.co/transformers/v2.11.0/model_doc/roberta.html 
		pad_indices = (input_ids_sample == pad_token_id).nonzero(as_tuple=True)[0]
		if len(pad_indices) > 0:
			first_pad_index = pad_indices[0].item()
		else:
			first_pad_index = (seq_length - 1)
			
		conceptColumnEndIndexList.append(first_pad_index)

		# Remove the last start index as per the pseudocode
		if(len(conceptColumnStartIndexList) > 1):
			conceptColumnStartIndexList.pop()
		
		assert len(conceptColumnStartIndexList) == len(conceptColumnEndIndexList)
		#print("conceptColumnStartIndexList = ", conceptColumnStartIndexList)
		#print("conceptColumnEndIndexList = ", conceptColumnEndIndexList)
		
		# For each token, assign its concept column start and end indices
		token_concept_start_indices = pt.zeros(seq_length, dtype=pt.long)
		token_concept_end_indices = pt.zeros(seq_length, dtype=pt.long)

		# Assign concept columns to tokens
		current_concept_idx = 0
		for idx in range(seq_length):
			if current_concept_idx < len(conceptColumnStartIndexList):
				start_idx = conceptColumnStartIndexList[current_concept_idx]
				end_idx = conceptColumnEndIndexList[current_concept_idx]
				token_concept_start_indices[idx] = start_idx
				token_concept_end_indices[idx] = end_idx
				if idx == end_idx:
					current_concept_idx += 1
			else:
				# For tokens after the last concept column
				token_concept_start_indices[idx] = conceptColumnStartIndexList[-1]
				token_concept_end_indices[idx] = conceptColumnEndIndexList[-1]

		batch_concept_start_indices.append(token_concept_start_indices)
		batch_concept_end_indices.append(token_concept_end_indices)


	# Stack tensors to create batch tensors
	conceptColumnStartIndices = pt.stack(batch_concept_start_indices)  # Shape: [batch_size, seq_length]
	conceptColumnEndIndices = pt.stack(batch_concept_end_indices)	  # Shape: [batch_size, seq_length]

	conceptColumnStartIndices = conceptColumnStartIndices.to(device)
	conceptColumnEndIndices = conceptColumnEndIndices.to(device)

	'''
	print("batch_input_ids = ", batch_input_ids)
	print("conceptColumnStartIndices = ", conceptColumnStartIndices)
	print("conceptColumnEndIndices = ", conceptColumnEndIndices)
	'''
	
	return conceptColumnStartIndices, conceptColumnEndIndices


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
		
