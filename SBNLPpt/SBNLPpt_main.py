"""SBNLPpt_main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
conda create -n transformersenv
source activate transformersenv
conda install python=3.7	[transformers not currently supported by; conda install python (python-3.10.6)]
pip install datasets
pip install transfomers==4.23.1
pip install torch
pip install lovely-tensors
pip install nltk

# Usage:
source activate transformersenv
python SBNLPpt_main.py

# Description:
SBNLPpt main - Syntactic Bias natural language processing (SBNLP): neural architectures with various syntactic inductive biases 
(recursiveLayers, simulatedDendriticBranches, memoryTraceBias, semanticRelationVectorSpaces, tokenMemoryBank)

"""


import torch
from tqdm.auto import tqdm

from transformers import AdamW
import math 
import os

from SBNLPpt_globalDefs import *
import SBNLPpt_data
if(useAlgorithmTransformer):
	from SBNLPpt_transformer import createModel, loadModel, saveModel, propagate
	import SBNLPpt_transformer
elif(useAlgorithmRNN):
	from SBNLPpt_RNN import createModel, loadModel, saveModel, propagate
	import SBNLPpt_RNN
elif(useAlgorithmSANI):
	from SBNLPpt_SANI import createModel, loadModel, saveModel, propagate
	import SBNLPpt_SANI
elif(useAlgorithmGIA):
	from SBNLPpt_GIA import createModel, loadModel, saveModel, propagate
	import SBNLPpt_GIA
	import SBNLPpt_GIAdefinePOSwordLists

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def main():
	tokenizer, dataElements = SBNLPpt_data.initialiseDataLoader()
	if(usePretrainedModelDebug):
		if(stateTrainDataset):
			print("usePretrainedModelDebug error: stateTestDataset required")
			exit()
		if(stateTestDataset):
			testDataset(tokenizer, dataElements)
	else:
		if(stateTrainDataset):
			trainDataset(tokenizer, dataElements)
		if(stateTestDataset):
			testDataset(tokenizer, dataElements)
			
def continueTrainingModel():
	continueTrain = False
	if((trainStartEpoch > 0) or (trainStartDataFile > 0)):
		continueTrain = True	#if trainStartEpoch=0 and trainStartDataFile=0 will recreate model, if trainStartEpoch>0 or trainStartDataFile>0 will load existing model
	return continueTrain	
	
def trainDataset(tokenizer, dataElements):

	#vocabSize = countNumberOfTokens(tokenizer)
	model, optim = prepareModelTrainWrapper()

	if(usePreprocessedDataset):
		pathIndexMin = trainStartDataFile
		pathIndexMax = pathIndexMin+int(trainNumberOfDataFiles*dataFilesFeedMultiplier)
	else:
		pathIndexMin = None
		pathIndexMax = None
	
	if(useAlgorithmTransformer):
		useMLM = True
	else:
		useMLM = False
	loader = SBNLPpt_data.createDataLoader(useMLM, tokenizer, dataElements, trainNumberOfDataFiles, pathIndexMin, pathIndexMax)
	
	for epoch in range(trainStartEpoch, trainStartEpoch+trainNumberOfEpochs):
		loop = tqdm(loader, leave=True)
		
		if(printAccuracyRunningAverage):
			(runningLoss, runningAccuracy) = (0.0, 0.0)
		
		for batchIndex, batch in enumerate(loop):			
			loss, accuracy = trainBatchWrapper(batchIndex, batch, tokenizer, model, optim)
			
			if(printAccuracyRunningAverage):
				(loss, accuracy) = (runningLoss, runningAccuracy) = (runningLoss/runningAverageBatches*(runningAverageBatches-1)+(loss/runningAverageBatches), runningAccuracy/runningAverageBatches*(runningAverageBatches-1)+(accuracy/runningAverageBatches))
				
			loop.set_description(f'Epoch {epoch}')
			loop.set_postfix(batchIndex=batchIndex, loss=loss, accuracy=accuracy)
		
		print("finished training model")
		saveModel(model)
		
def testDataset(tokenizer, dataElements):

	model = prepareModelTestWrapper()
	
	if(usePreprocessedDataset):
		if(reserveValidationSet):
			pathIndexMin = int(datasetNumberOfDataFiles*trainSplitFraction)
		else:
			pathIndexMin = 0
		pathIndexMax = pathIndexMin+int(testNumberOfDataFiles*dataFilesFeedMultiplier)
	else:
		pathIndexMin = None
		pathIndexMax = None
	
	if(useAlgorithmTransformer):
		useMLM = True
	else:
		useMLM = False
		
	loader = SBNLPpt_data.createDataLoader(useMLM, tokenizer, dataElements, testNumberOfDataFiles, pathIndexMin, pathIndexMax)
		
	for epoch in range(trainStartEpoch, trainStartEpoch+trainNumberOfEpochs):
		loop = tqdm(loader, leave=True)
		
		if(printAccuracyRunningAverage):
			(runningLoss, runningAccuracy) = (0.0, 0.0)
		(averageAccuracy, averageLoss, batchCount) = (0.0, 0.0, 0)
		
		for batchIndex, batch in enumerate(loop):
			loss, accuracy = testBatchWrapper(batchIndex, batch, tokenizer, model)
			
			if(not usePretrainedModelDebug or not math.isnan(accuracy)):
				(averageAccuracy, averageLoss, batchCount) = (averageAccuracy+accuracy, averageLoss+loss, batchCount+1)
			
			if(printAccuracyRunningAverage):
				(loss, accuracy) = (runningLoss, runningAccuracy) = (runningLoss/runningAverageBatches*(runningAverageBatches-1)+(loss/runningAverageBatches), runningAccuracy/runningAverageBatches*(runningAverageBatches-1)+(accuracy/runningAverageBatches))
				
			loop.set_description(f'Epoch {epoch}')
			loop.set_postfix(batchIndex=batchIndex, loss=loss, accuracy=accuracy)

		(averageAccuracy, averageLoss) = (averageAccuracy/batchCount, averageLoss/batchCount)
		print("averageAccuracy = ", averageAccuracy)
		print("averageLoss = ", averageLoss)


def prepareModelTrainWrapper():
	if(useAlgorithmGIA):
		SBNLPpt_GIA.preparePOSdictionary()	#required for SBNLPpt_getAllPossiblePosTags.getAllPossiblePosTags(word)
	if(useMultipleModels):
		for vectorSpaceIndex, vectorSpace in enumerate(SBNLPpt_GIAdefinePOSwordLists.vectorSpaceList):
			vocabSize = vocabularySize
			if(encode3tuples):
				if(vectorSpace.intermediateType == SBNLPpt_GIAdefinePOSwordLists.intermediateTypePOS):
					vocabSize = vocabularySize + vocabularySize	#size = referenceSetSuj/Obj (entity) + referenceSetDelimiter (semanticRelation)
			model, optim = prepareModelTrain(vocabSize)
			vectorSpace.model = model
			vectorSpace.optim = optim
	else:
		model, optim = prepareModelTrain(vocabularySize)
	return model, optim
	
def prepareModelTestWrapper():
	if(useAlgorithmGIA):
		SBNLPpt_GIA.preparePOSdictionary()	#required for SBNLPpt_getAllPossiblePosTags.getAllPossiblePosTags(word)
	if(useMultipleModels):
		for vectorSpaceIndex, vectorSpace in enumerate(SBNLPpt_GIAdefinePOSwordLists.vectorSpaceList):
			model = prepareModelTest()
			vectorSpace.model = model
	else:
		model = prepareModelTest()
	return model

		
def trainBatchWrapper(batchIndex, batch, tokenizer, model, optim):
	if(useMultipleModels):
		averageAccuracy = 0.0
		averageLoss = 0.0 
		spaceCount = 0
		if(useVectorisedSemanticRelationIdentification):
			labelsList, labelsFoundList = SBNLPpt_GIA.calculateXYlabels(tokenizer, batch, vocabularySize)
		for vectorSpaceIndex, vectorSpace in enumerate(SBNLPpt_GIAdefinePOSwordLists.vectorSpaceList):
			model = vectorSpace.model
			optim = vectorSpace.optim	
			if(useVectorisedSemanticRelationIdentification):
				(labels, labelsFound) = (labelsList[vectorSpaceIndex], labelsFoundList[vectorSpaceIndex])
			else:
				labels, labelsFound = SBNLPpt_GIA.calculateXYlabels(tokenizer, vectorSpace, vectorSpaceIndex, batch, vocabularySize)
			
			if(labelsFound):
				loss, accuracy = trainBatch(batchIndex, labels, tokenizer, model, optim)
			else:
				(loss, accuracy) = (0.0, 0.0)
			averageAccuracy = averageAccuracy + accuracy
			averageLoss = averageLoss + loss
			spaceCount = spaceCount + 1
			
		accuracy = averageAccuracy/spaceCount
		loss = averageLoss/spaceCount
	else:
		loss, accuracy = trainBatch(batchIndex, batch, tokenizer, model, optim)
	return loss, accuracy
		
def testBatchWrapper(batchIndex, batch, tokenizer, model):
	if(useMultipleModels):
		averageAccuracy = 0.0
		averageLoss = 0.0 
		spaceCount = 0
		if(useVectorisedSemanticRelationIdentification):
			labelsList, labelsFoundList = SBNLPpt_GIA.calculateXYlabels(tokenizer, batch, vocabularySize)	
		for vectorSpace, vectorSpaceIndex in enumerate(SBNLPpt_GIAdefinePOSwordLists.vectorSpaceList):
			model = vectorSpace.model
			if(useVectorisedSemanticRelationIdentification):
				(labels, labelsFound) = (labelsList[vectorSpaceIndex], labelsFoundList[vectorSpaceIndex])
			else:
				labels, labelsFound = SBNLPpt_GIA.calculateXYlabels(tokenizer, vectorSpace, vectorSpaceIndex, batch, vocabularySize)
			if(labelsFound):
				loss, accuracy = testBatch(batchIndex, labels, tokenizer, model)
			else:
				(loss, accuracy) = (0.0, 0.0)
			averageAccuracy = averageAccuracy + accuracy
			averageLoss = averageLoss + loss
			spaceCount = spaceCount + 1

		accuracy = averageAccuracy/spaceCount
		loss = averageLoss/spaceCount
	else:
		loss, accuracy = testBatch(batchIndex, batch, tokenizer, model)
	return loss, accuracy
	
def prepareModelTrain(vocabSize):

	if(debugDoNotTrainModel):
		model = None
		optim = None
	else:
		if(continueTrainingModel()):
			model = loadModel()
		else:
			model = createModel(vocabSize)

		model.to(device)

		model.train()
		optim = AdamW(model.parameters(), lr=learningRate)
	
	return model, optim
	
def prepareModelTest():

	if(usePretrainedModelDebug):
		if(useAlgorithmTransformer):
			model = RobertaForMaskedLM.from_pretrained("roberta-base")
		else:
			print("testDataset error: usePretrainedModelDebug requires useAlgorithmTransformer")
			exit()
	else:
		model = loadModel()

	model.to(device)

	model.eval()
	
	return model
	
def trainBatch(batchIndex, batch, tokenizer, model, optim):
	if(debugDoNotTrainModel):
		loss = 0.0
		accuracy = 0.0
	else:
		optim.zero_grad()

		loss, accuracy = propagate(device, model, tokenizer, batch)

		loss.backward()
		optim.step()

		if(batchIndex % modelSaveNumberOfBatches == 0):
			saveModel(model)

		loss = loss.item()
	return loss, accuracy
			
def testBatch(batchIndex, batch, tokenizer, model):

	loss, accuracy = propagate(device, model, tokenizer, batch)

	loss = loss.detach().cpu().numpy()
	
	return loss, accuracy


if(__name__ == '__main__'):
	main()

