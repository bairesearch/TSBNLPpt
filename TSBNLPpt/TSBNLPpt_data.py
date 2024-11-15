"""TSBNLPpt_data.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see TSBNLPpt_main.py

# Usage:
see TSBNLPpt_main.py

# Description:
TSBNLPpt data

"""

import torch

import TSBNLPpt_dataPreprocess
import TSBNLPpt_dataTokeniser
import TSBNLPpt_dataLoader
import TSBNLPpt_dataLoaderOrdered
from TSBNLPpt_globalDefs import *

def initialiseDataLoader():
	dataset = None
	if(statePreprocessDataset or (not usePreprocessedDataset)):
		dataset = TSBNLPpt_dataPreprocess.loadDataset()
	if(statePreprocessDataset):
		TSBNLPpt_dataPreprocess.preprocessDataset(dataset)
	dataElements = TSBNLPpt_dataPreprocess.prepareDataElements(dataset)
	
	if(usePretrainedModelDebug):
		tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
	else:
		if(stateTrainTokeniser):
			TSBNLPpt_dataTokeniser.trainTokeniser(dataElements, vocabularySize)
		if(stateTrainTokeniser or stateTrainDataset or stateTestDataset):
			tokenizer = TSBNLPpt_dataTokeniser.loadTokeniser()
		if(debugCreateOrderedDatasetFiles):
			TSBNLPpt_dataLoaderOrdered.createDatasetLargeDocuments(tokenizer, dataElements)
	
	return tokenizer, dataElements
	
def createDataLoader(useMLM, tokenizer, dataElements, trainNumberOfDataFiles, pathIndexMin, pathIndexMax):
	return TSBNLPpt_dataLoader.createDataLoader(useMLM, tokenizer, dataElements, trainNumberOfDataFiles, pathIndexMin, pathIndexMax)

#can be moved to TSBNLP algorithm common file:

def getAccuracy(input_ids, attention_mask, labels, outputs):
	if(useMaskedLM):
		return getAccuracyMaskedLM(input_ids, labels, outputs)
	else:
		return getAccuracyCausalLM(input_ids, outputs, attention_mask)

def generateProsodyExcludedTokenSet(tokenizer):
	global excluded_token_set
	excluded_token_set = None
	if(prosodyDelimitedData):
		if(prosodyDelimitedType=="controlTokens" or prosodyDelimitedType=="txtDebug"):
			excludedTokensIntList = []
		elif(prosodyDelimitedType=="uniqueTokens"):
			excludedTokensIntList = read_token_ids("prosodyTokenIDs.txt")
			print("excludedTokensIntList = ", excludedTokensIntList)
			#excludedTokensStringList = read_token_ids("prosodyTokens.txt")
			#excludedTokensIntList = [tokenizer.encode(token, add_special_tokens=False)[0] for token in excludedTokensStringList]
		elif(prosodyDelimitedType=="repeatTokens"):
			excludedTokensIntList = []	#TODO
		excluded_token_set = set(excludedTokensIntList)
	#return excluded_token_set
				
def removeProsodyTokensFromPredictionMask(predictionMask, labels, excluded_token_set):
	if(prosodyDelimitedData):
		if(prosodyDelimitedType=="uniqueTokens"):	#FUTURE: or prosodyDelimitedType=="repeatTokens"
			for token in excluded_token_set:
				predictionMask &= (labels != token)
	return predictionMask
				
def getAccuracyMaskedLM(inputIDs, labels, outputs):
	predictionMask = torch.where(inputIDs==customMaskTokenID, 1.0, 0.0)	#maskTokenIndexFloat = maskTokenIndex.float()	#orig: maskTokenIndex
	###predictionMask = removeProsodyTokensFromPredictionMask(predictionMask, labels, excluded_token_set)
	#tokenizerNumberTokens = TSBNLPpt_dataTokeniser.getTokenizerLength(tokenizer)
	tokenLogits = (outputs.logits).detach()
	accuracy = getAccuracyWithPredictionMask(labels, tokenLogits, predictionMask)
	return accuracy

def getAccuracyCausalLM(inputs, outputs, attention_mask):	
	#based on TSBNLPpt_data:getAccuracyMaskedLM
	predictionMask = attention_mask[:, 1:]
	logits = outputs.logits.detach()
	logits = outputs.logits.detach()
	# Shift so that tokens < n predict n
	shift_labels = inputs[..., 1:].contiguous()
	shift_logits = logits[..., :-1, :].contiguous()
	###predictionMask = removeProsodyTokensFromPredictionMask(predictionMask, shift_labels, excluded_token_set)
	accuracy = getAccuracyWithPredictionMask(shift_labels, shift_logits, predictionMask)
	accuracy = accuracy.item()
	#print("accuracy = ", accuracy)
	return accuracy
	
def getAccuracyWithPredictionMask(labels, tokenLogits, predictionMask):	
	tokenLogitsTopIndex = torch.topk(tokenLogits, accuracyTopN).indices	#get highest n scored entries from dictionary	#tokenLogitsTopIndex.shape = batchSize, sequenceMaxNumTokens, accuracyTopN
	if(accuracyTopN == 1):
		tokenLogitsTopIndex = torch.squeeze(tokenLogitsTopIndex)	#tokenLogitsTopIndex[:, :, 1] -> #tokenLogitsTopIndex[:, :]
		comparison = (tokenLogitsTopIndex == labels).float()
		comparisonMasked = torch.multiply(comparison, predictionMask)
		accuracy = (torch.sum(comparisonMasked)/torch.sum(predictionMask)).cpu().numpy()	#accuracy.item()
	else:
		labelsExpanded = torch.unsqueeze(labels, dim=2)
		labelsExpanded = labelsExpanded.expand(-1, -1, tokenLogitsTopIndex.shape[2])	#labels broadcasted to [batchSize, sequenceMaxNumTokens, accuracyTopN]
		comparison = (tokenLogitsTopIndex == labelsExpanded).float()
		predictionMaskExpanded = torch.unsqueeze(predictionMask, dim=2)
		predictionMaskExpanded = predictionMaskExpanded.expand(-1, -1, tokenLogitsTopIndex.shape[2])	#predictionMask broadcasted to [batchSize, sequenceMaxNumTokens, accuracyTopN]
		comparisonMasked = torch.multiply(comparison, predictionMaskExpanded)	#predictionMask broadcasted to [batchSize, sequenceMaxNumTokens, accuracyTopN]
		accuracy = (torch.sum(comparisonMasked)/torch.sum(predictionMask)).cpu().numpy() 	#or torch.sum(comparisonMasked)/(torch.sum(predictionMaskExpanded)/accuracyTopN)	#accuracy.item()
	return accuracy
