"""SBNLPpt_data.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SBNLPpt_main.py

# Usage:
see SBNLPpt_main.py

# Description:
SBNLPpt data

"""

import torch

import SBNLPpt_dataPreprocess
import SBNLPpt_dataTokeniser
import SBNLPpt_dataLoader
import SBNLPpt_dataLoaderOrdered
from SBNLPpt_globalDefs import *

def initialiseDataLoader():
	dataset = None
	if(statePreprocessDataset or (not usePreprocessedDataset)):
		dataset = SBNLPpt_dataPreprocess.loadDataset()
	if(statePreprocessDataset):
		SBNLPpt_dataPreprocess.preprocessDataset(dataset)
	dataElements = SBNLPpt_dataPreprocess.prepareDataElements(dataset)
	
	if(usePretrainedModelDebug):
		tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
	else:
		if(stateTrainTokeniser):
			SBNLPpt_dataTokeniser.trainTokeniser(dataElements, vocabularySize)
		if(stateTrainDataset or stateTestDataset):
			tokenizer = SBNLPpt_dataTokeniser.loadTokeniser()
		if(debugCreateOrderedDatasetFiles):
			SBNLPpt_dataLoaderOrdered.createDatasetLargeDocuments(tokenizer, dataElements)
	
	return tokenizer, dataElements
	
def createDataLoader(useMLM, tokenizer, dataElements, trainNumberOfDataFiles, pathIndexMin, pathIndexMax):
	return SBNLPpt_dataLoader.createDataLoader(useMLM, tokenizer, dataElements, trainNumberOfDataFiles, pathIndexMin, pathIndexMax)

#can be moved to SBNLP algorithm common file:

def getAccuracy(tokenizer, input_ids, attention_mask, labels, outputs):
	if(useMaskedLM):
		return getAccuracyMaskedLM(tokenizer, input_ids, labels, outputs)
	else:
		return getAccuracyCausalLM(input_ids, outputs)
		
def getAccuracyMaskedLM(tokenizer, inputIDs, labels, outputs):
	predictionMask = torch.where(inputIDs==customMaskTokenID, 1.0, 0.0)	#maskTokenIndexFloat = maskTokenIndex.float()	#orig: maskTokenIndex

	tokenizerNumberTokens = SBNLPpt_dataTokeniser.getTokenizerLength(tokenizer)
	
	tokenLogits = (outputs.logits).detach()

	tokenLogitsTopIndex = torch.topk(tokenLogits, accuracyTopN).indices	#get highest n scored entries from dictionary	#tokenLogitsTopIndex.shape = batchSize, sequenceMaxNumTokens, accuracyTopN
	
	if(accuracyTopN == 1):
		tokenLogitsTopIndex = torch.squeeze(tokenLogitsTopIndex)	#tokenLogitsTopIndex[:, :, 1] -> #tokenLogitsTopIndex[:, :]
		comparison = (tokenLogitsTopIndex == labels).float()
		comparisonMasked = torch.multiply(comparison, predictionMask)
		accuracy = (torch.sum(comparisonMasked)/torch.sum(predictionMask)).cpu().numpy()
	else:
		labelsExpanded = torch.unsqueeze(labels, dim=2)
		labelsExpanded = labelsExpanded.expand(-1, -1, tokenLogitsTopIndex.shape[2])	#labels broadcasted to [batchSize, sequenceMaxNumTokens, accuracyTopN]
		comparison = (tokenLogitsTopIndex == labelsExpanded).float()
		predictionMaskExpanded = torch.unsqueeze(predictionMask, dim=2)
		predictionMaskExpanded = predictionMaskExpanded.expand(-1, -1, tokenLogitsTopIndex.shape[2])	#predictionMask broadcasted to [batchSize, sequenceMaxNumTokens, accuracyTopN]
		comparisonMasked = torch.multiply(comparison, predictionMaskExpanded)	#predictionMask broadcasted to [batchSize, sequenceMaxNumTokens, accuracyTopN]
		accuracy = (torch.sum(comparisonMasked)/torch.sum(predictionMask)).cpu().numpy() 	#or torch.sum(comparisonMasked)/(torch.sum(predictionMaskExpanded)/accuracyTopN)
	
	#accuracy2 = (torch.mean(comparisonMasked)).cpu().numpy()
	
	return accuracy

def getAccuracyCausalLM(inputs, outputs):	
	#based on SBNLPpt_data:getAccuracyMaskedLM
	logits = outputs.logits.detach()
	logits = outputs.logits.detach()
	# Shift so that tokens < n predict n
	shift_labels = inputs[..., 1:].contiguous()
	shift_logits = logits[..., :-1, :].contiguous()
	tokenLogitsTopIndex = torch.topk(shift_logits, accuracyTopN).indices	#get highest n scored entries from dictionary	#tokenLogitsTopIndex.shape = batchSize, sequenceMaxNumTokens, accuracyTopN
	if(accuracyTopN == 1):
		tokenLogitsTopIndex = torch.squeeze(tokenLogitsTopIndex)	#tokenLogitsTopIndex[:, :, 1] -> #tokenLogitsTopIndex[:, :] 	
		comparison = (tokenLogitsTopIndex == shift_labels).float()
		accuracy = torch.mean(comparison)
	else:
		labelsExpanded = torch.unsqueeze(shift_labels, dim=2)
		labelsExpanded = labelsExpanded.expand(-1, -1, tokenLogitsTopIndex.shape[2])	#labels broadcasted to [batchSize, sequenceMaxNumTokens, accuracyTopN]
		comparison = (tokenLogitsTopIndex == labelsExpanded).float()
		accuracy = torch.mean(comparison)
	accuracy = accuracy.item()
	#print("accuracy = ", accuracy)
	return accuracy
	
