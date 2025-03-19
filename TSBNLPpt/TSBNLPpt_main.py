"""TSBNLPpt_main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
conda create -n pytorchsenv
source activate pytorchsenv
conda install python=3.12
pip install networkx
pip install matplotlib
pip install yattag
pip install torch
pip install torch_geometric
pip install nltk 
pip install spacy
pip install benepar
pip install datasets
pip install transfomers
pip install lovely-tensors
pip install torchmetrics
pip install pynvml
pip install sortedcontainers
python3 -m spacy download en_core_web_md

# Usage:
source activate pytorchsenv
python TSBNLPpt_main.py

# Description:
TSBNLPpt main - Transformer Syntactic Bias natural language processing (TSBNLP): transformer with various syntactic inductive biases 
(recursiveLayers, simulatedDendriticBranches, memoryTraceBias, GIAsemanticRelationVectorSpaces, tokenMemoryBank, transformerAttentionHeadPermutations, transformerPOSembeddings, transformerSuperblocks)

"""


import torch
from tqdm.auto import tqdm

from transformers import AdamW
import math 
import os

from TSBNLPpt_globalDefs import *
import TSBNLPpt_data
if(useAlgorithmTransformer):
	from TSBNLPpt_transformer import createModel, loadModel, saveModel, propagate
	import TSBNLPpt_transformer
	if(useMultipleModels):
		from TSBNLPpt_transformer import loadModelIndex, createModelIndex, saveModelIndex, propagateIndex
elif(useAlgorithmRNN):
	from TSBNLPpt_RNN import createModel, loadModel, saveModel, propagate
	import TSBNLPpt_RNN
elif(useAlgorithmSANI):
	from TSBNLPpt_SANI import createModel, loadModel, saveModel, propagate
	import TSBNLPpt_SANI
elif(useAlgorithmGIA):
	from TSBNLPpt_GIA import createModel, loadModel, saveModel, propagate
	import TSBNLPpt_GIA
	import TSBNLPpt_GIAvectorSpaces
	if(useMultipleModels):
		from TSBNLPpt_GIA import loadModelIndex, createModelIndex, saveModelIndex, propagateIndex


if(useMultipleModels):
	if(useAlgorithmGIA):
		modelStoreList = TSBNLPpt_GIAvectorSpaces.vectorSpaceList
	elif(useAlgorithmTransformer):
		modelStoreList = TSBNLPpt_transformer.modelStoreList

if(useGPU):
	device = torch.device('cuda') #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
else:
	device = torch.device('cpu')
	
def main():
	tokenizer, dataElements = TSBNLPpt_data.initialiseDataLoader()
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

	TSBNLPpt_data.generateProsodyExcludedTokenSet(tokenizer)
	
	#vocabSize = countNumberOfTokens(tokenizer)
	model, optim = prepareModelTrainOrTestWrapper(True)

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

	if(useTrainWarmup):
		learningRateCurrent = warmupLearningRateStart
	
	for epoch in range(trainStartEpoch, trainStartEpoch+trainNumberOfEpochs):
		loader = TSBNLPpt_data.createDataLoader(useMLM, tokenizer, dataElements, trainNumberOfDataFiles, pathIndexMin, pathIndexMax)	#required to reset dataloader and still support tqdm modification
		loop = tqdm(loader, leave=True)
		
		if(printAccuracyRunningAverage):
			(runningLoss, runningAccuracy) = (0.0, 0.0)
		
		for batchIndex, batch in enumerate(loop):
			loss, accuracy = trainOrTestBatchWrapper(True, batchIndex, batch, tokenizer, model, optim)

			if(useTrainWarmup):
				if(epoch == trainStartEpoch):
					if(batchIndex < warmupSteps):
						learningRateCurrent += warmupLearningRateIncrement
						for param_group in optim.param_groups:
							param_group['lr'] = learningRateCurrent

			if(printAccuracyRunningAverage):
				(loss, accuracy) = (runningLoss, runningAccuracy) = (runningLoss/runningAverageBatches*(runningAverageBatches-1)+(loss/runningAverageBatches), runningAccuracy/runningAverageBatches*(runningAverageBatches-1)+(accuracy/runningAverageBatches))
				
			loop.set_description(f'Epoch {epoch}')
			if(debugPrintMultipleModelAccuracy):
				loop.set_postfix(batchIndex=batchIndex, accuracy2=loss, accuracy=accuracy)
			else:
				loop.set_postfix(batchIndex=batchIndex, loss=loss, accuracy=accuracy)

		print("finished training model")
		if(not useMultipleModels):	#limitation; useMultipleModels cannot save model after last train batch
			saveModel(model)
					
def testDataset(tokenizer, dataElements):

	TSBNLPpt_data.generateProsodyExcludedTokenSet(tokenizer)

	model, optim = prepareModelTrainOrTestWrapper(False)
	
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
				
	for epoch in range(trainStartEpoch, trainStartEpoch+trainNumberOfEpochs):
		loader = TSBNLPpt_data.createDataLoader(useMLM, tokenizer, dataElements, testNumberOfDataFiles, pathIndexMin, pathIndexMax)	#required to reset dataloader and still support tqdm modification
		loop = tqdm(loader, leave=True)
		
		if(printAccuracyRunningAverage):
			(runningLoss, runningAccuracy) = (0.0, 0.0)
		(averageAccuracy, averageLoss, batchCount) = (0.0, 0.0, 0)
		
		for batchIndex, batch in enumerate(loop):
			loss, accuracy = trainOrTestBatchWrapper(False, batchIndex, batch, tokenizer, model)
			
			if(not usePretrainedModelDebug and not math.isnan(accuracy)):
				(averageAccuracy, averageLoss, batchCount) = (averageAccuracy+accuracy, averageLoss+loss, batchCount+1)
				
				if(printAccuracyRunningAverage):
					(loss, accuracy) = (runningLoss, runningAccuracy) = (runningLoss/runningAverageBatches*(runningAverageBatches-1)+(loss/runningAverageBatches), runningAccuracy/runningAverageBatches*(runningAverageBatches-1)+(accuracy/runningAverageBatches))
				
			loop.set_description(f'Epoch {epoch}')
			loop.set_postfix(batchIndex=batchIndex, loss=loss, accuracy=accuracy)

		(averageAccuracy, averageLoss) = (averageAccuracy/batchCount, averageLoss/batchCount)
		print("averageAccuracy = ", averageAccuracy)
		print("averageLoss = ", averageLoss)

def prepareModelTrainOrTestWrapper(trainOrTest):
	if(transformerPOSembeddings):
		TSBNLPpt_transformer.preparePOSdictionary()
	if(useAlgorithmGIA):
		TSBNLPpt_GIA.preparePOSdictionary()	#required for TSBNLPpt_POSgetAllPossiblePosTags.getAllPossiblePosTags(word)
	if(useMultipleModels):
		for modelStoreIndex, modelStore in enumerate(modelStoreList):
			vocabSize = vocabularySize
			if(useAlgorithmGIA):
				if(encode3tuples):
					if(modelStore.intermediateType == TSBNLPpt_GIAvectorSpaces.intermediateTypePOS):
						vocabSize = vocabularySize + vocabularySize	#size = referenceSetSuj/Obj (entity) + referenceSetDelimiter (semanticRelation)
			model, optim = prepareModelTrainOrTest(trainOrTest, vocabSize, modelStoreIndex)
			if(useAlgorithmGIA):
				saveModelIndex(model, modelStoreIndex)	#save required to quickly generate GIA model files for useAlgorithmTransformer testing
			(modelStore.model, modelStore.optim) = (model, optim)
	else:
		model, optim = prepareModelTrainOrTest(trainOrTest, vocabularySize)
	return model, optim

def trainOrTestBatchWrapper(trainOrTest, batchIndex, batch, tokenizer, model, optim=None):
	if(useMultipleModels):
		averageAccuracy = 0.0
		averageAccuracy2 = 0.0
		averageLoss = 0.0 
		spaceCount = 0
		spaceCount2 = 0
		batchList, batchFoundList = multipleModelsGetBatchList(tokenizer, batch)
		for modelStoreIndex, modelStore in enumerate(modelStoreList):
			model = modelStore.model
			if(trainOrTest):
				optim = modelStore.optim
			labels, labelsFound = multipleModelsGetBatch(tokenizer, modelStoreIndex, batch, batchList, batchFoundList, model)
			if(labelsFound):
				loss, accuracy = trainOrTestBatch(trainOrTest, batchIndex, labels, tokenizer, model, optim, modelStoreIndex)
			else:
				(loss, accuracy) = (0.0, 0.0)
			if(debugPrintMultipleModelAccuracy):	
				if(modelStoreIndex == 0):
					averageAccuracy = averageAccuracy + accuracy
					spaceCount += 1
				else:
					averageAccuracy2 = averageAccuracy2 + accuracy
					spaceCount2 += 1
			else:
				averageAccuracy = averageAccuracy + accuracy
				averageLoss = averageLoss + loss
				spaceCount += 1
		
		if(debugPrintMultipleModelAccuracy):
			accuracy = averageAccuracy/spaceCount
			loss = averageAccuracy2/spaceCount2
		else:
			accuracy = averageAccuracy/spaceCount
			loss = averageLoss/spaceCount
	else:
		loss, accuracy = trainOrTestBatch(trainOrTest, batchIndex, batch, tokenizer, model, optim)
	return loss, accuracy

def multipleModelsGetBatchList(tokenizer, batch):
	(labelsList, labelsFoundList) = (None, None)
	if(useAlgorithmGIA):
		if(GIAuseVectorisedSemanticRelationIdentification):
			labelsList, labelsFoundList = TSBNLPpt_GIA.calculateXYlabels(tokenizer, batch, vocabularySize)
	return labelsList, labelsFoundList

def multipleModelsGetBatch(tokenizer, modelStoreIndex, batch, labelsList=None, labelsFoundList=None, model=None):
	if(useAlgorithmGIA):
		if(GIAuseVectorisedSemanticRelationIdentification):
			(labels, labelsFound) = (labelsList[modelStoreIndex], labelsFoundList[modelStoreIndex])
		else:
			labels, labelsFound = TSBNLPpt_GIA.calculateXYlabels(tokenizer, modelStoreList[modelStoreIndex], modelStoreIndex, batch, vocabularySize)
	elif(useAlgorithmTransformer):
		if(modelStoreIndex == 0):
			labels = batch
		else:
			layerIndex = TSBNLPpt_transformer.getLayerIndex(modelStoreIndex)
			labels = TSBNLPpt_transformer.getTokenMemoryBankStorageSelectionModelBatch(layerIndex)
		labelsFound = True
	return labels, labelsFound

def prepareModelTrainOrTest(trainOrTest, vocabSize, modelStoreIndex=None):
	if(trainOrTest):
		return prepareModelTrain(vocabSize, modelStoreIndex)
	else:
		return prepareModelTest(modelStoreIndex)

def prepareModelTrain(vocabSize, modelStoreIndex=None):
	if(debugDoNotTrainModel):
		model = None
		optim = None
	else:
		if(useMultipleModels):
			if(continueTrainingModel()):
				model = loadModelIndex(modelStoreIndex)
			else:
				model = createModelIndex(vocabSize, modelStoreIndex)
		else:
			if(continueTrainingModel()):
				model = loadModel()
			else:
				model = createModel(vocabSize)		

		model.to(device)

		model.train()
		if(useTrainWarmup):
			learningRateCurrent = warmupLearningRateStart
		else:
			learningRateCurrent = learningRate
		optim = AdamW(model.parameters(), lr=learningRateCurrent)
	
	return model, optim

def prepareModelTest(modelStoreIndex=None):
	if(usePretrainedModelDebug):
		if(useAlgorithmTransformer):
			model = RobertaForMaskedLM.from_pretrained("roberta-base")
		else:
			print("testDataset error: usePretrainedModelDebug requires useAlgorithmTransformer")
			exit()
	else:
		if(useMultipleModels):
			model = loadModelIndex(modelStoreIndex)
		else:
			model = loadModel()

	model.to(device)
	model.eval()
	
	optim = None
	return model, optim

def trainOrTestBatch(trainOrTest, batchIndex, batch, tokenizer, model, optim=None, modelStoreIndex=None):
	if(trainOrTest):
		loss, accuracy = trainBatch(batchIndex, batch, tokenizer, model, optim, modelStoreIndex)
	else:
		loss, accuracy = testBatch(batchIndex, batch, tokenizer, model, modelStoreIndex)

	return loss, accuracy	
	
def trainBatch(batchIndex, batch, tokenizer, model, optim, modelStoreIndex=None):
	if(debugDoNotTrainModel):
		loss = 0.0
		accuracy = 0.0
	else:
		optim.zero_grad()
	
		result = True
		if(useMultipleModels):
			loss, accuracy, result = propagateIndex(device, model, tokenizer, batch, modelStoreIndex, batchIndex)
		else:
			loss, accuracy = propagate(device, model, tokenizer, batch, batchIndex)
		if(result):
			loss.backward()
			optim.step()

		save = False       
		if(batchIndex % modelSaveNumberOfBatches == 0):
			save = True
		if(batchIndex == 0):
			if(continueTrainingModel()):
				save = False  #if continueTrainingModel then do not overwrite model during first batch
		if(save):
			print("saveModel: batchIndex = ", batchIndex, ", batchIndex//modelSaveNumberOfBatches = ", batchIndex//modelSaveNumberOfBatches)
			if(useMultipleModels):
				saveModelIndex(model, modelStoreIndex)
			else:
				saveModel(model)
		if(result):
			loss = loss.item()
	return loss, accuracy
			
def testBatch(batchIndex, batch, tokenizer, model, modelStoreIndex=None):

	if(useMultipleModels):
		loss, accuracy = propagateIndex(device, model, tokenizer, batch, modelStoreIndex)
	else:
		loss, accuracy = propagate(device, model, tokenizer, batch, batchIndex)

	loss = loss.detach().cpu().numpy()
	
	return loss, accuracy


if(__name__ == '__main__'):
	main()

