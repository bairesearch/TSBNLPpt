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
SBNLPpt main - Syntactic Bias natural language processing (SBNLP): various neural architectures with syntactic inductive biases (recursiveLayers, positionalEmbeddingTransformation)

"""


import torch
from tqdm.auto import tqdm
from pathlib import Path

from transformers import AdamW
import math 
import os

from SBNLPpt_globalDefs import *
import SBNLPpt_data
if(useAlgorithmTransformer):
	from SBNLPpt_transformer import createModel, loadModel, saveModel, propagate
	import useAlgorithmTransformer
elif(useAlgorithmRNN):
	from SBNLPpt_RNN import createModel, loadModel, saveModel, propagate
	import useAlgorithmRNN
elif(useAlgorithmSANI):
	from SBNLPpt_SANI import createModel, loadModel, saveModel, propagate
	import useAlgorithmSANI
elif(useAlgorithmGIA):
	from SBNLPpt_GIA import createModel, loadModel, saveModel, propagate
	import SBNLPpt_GIA

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def main():
	if(statePreprocessDataset):
		dataset = SBNLPpt_data.downloadDataset()
		SBNLPpt_data.preprocessDataset(dataset)
	
	if(Path(dataFolder).exists()):
		pathsGlob = Path(dataFolder).glob('**/*.txt')
		if(sortDataFilesByName):
			pathsGlob = sorted(pathsGlob, key=os.path.getmtime)	#key required because path names indices are not padded with 0s
		paths = [str(x) for x in pathsGlob]
	else:
		print("main error: Path does not exist, dataFolder = ", dataFolder)
		exit()
	
	if(usePretrainedModelDebug):
		tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
		testDataset(tokenizer, paths)
	else:
		if(stateTrainTokenizer):
			SBNLPpt_data.trainTokenizer(paths, vocabularySize)
		if(stateTrainDataset or stateTestDataset):
			tokenizer = SBNLPpt_data.loadTokenizer()
		if(stateTrainDataset):
			trainDataset(tokenizer, paths)
		if(stateTestDataset):
			testDataset(tokenizer, paths)
			
def continueTrainingModel():
	continueTrain = False
	if((trainStartEpoch > 0) or (trainStartDataFile > 0)):
		continueTrain = True	#if trainStartEpoch=0 and trainStartDataFile=0 will recreate model, if trainStartEpoch>0 or trainStartDataFile>0 will load existing model
	return continueTrain	
	
def trainDataset(tokenizer, paths):

	#vocabSize = countNumberOfTokens(tokenizer)
	model, optim = prepareModelTrainWrapper()

	numberOfDataFiles = len(paths)

	pathIndexMin = trainStartDataFile
	if(reserveValidationSet and trainNumberOfDataFiles==-1):
		pathIndexMax = int(numberOfDataFiles*trainSplitFraction)
	else:
		pathIndexMax = pathIndexMin+trainNumberOfDataFiles
	if(useAlgorithmTransformer):
		useMLM = True
	else:
		useMLM = False
	loader = SBNLPpt_data.createDataLoader(useMLM, tokenizer, paths, pathIndexMin, pathIndexMax)
	
	for epoch in range(trainStartEpoch, trainStartEpoch+trainNumberOfEpochs):
		loop = tqdm(loader, leave=True)
		for batchIndex, batch in enumerate(loop):

			loss, accuracy = trainBatchWrapper(batchIndex, batch, tokenizer, model, optim)
			
			loop.set_description(f'Epoch {epoch}')
			loop.set_postfix(batchIndex=batchIndex, loss=loss, accuracy=accuracy)
		
		saveModel(model)

def testDataset(tokenizer, paths):

	model = prepareModelTestWrapper()
	
	numberOfDataFiles = len(paths)

	pathIndexMin = int(numberOfDataFiles*trainSplitFraction)
	pathIndexMax = pathIndexMin+testNumberOfDataFiles
	if(useAlgorithmTransformer):
		useMLM = True
	else:
		useMLM = False
		
	loader = SBNLPpt_data.createDataLoader(useMLM, tokenizer, paths, pathIndexMin, pathIndexMax)
		
	for epoch in range(trainStartEpoch, trainStartEpoch+trainNumberOfEpochs):
		loop = tqdm(loader, leave=True)
		
		averageAccuracy = 0.0
		averageLoss = 0.0
		batchCount = 0
		
		for batchIndex, batch in enumerate(loop):
			
			loss, accuracy = testBatchWrapper(batchIndex, batch, tokenizer, model)
			
			if(not math.isnan(accuracy)):	#required for usePretrainedModelDebug only
				averageAccuracy = averageAccuracy + accuracy
				averageLoss = averageLoss + loss
				batchCount = batchCount + 1

			loop.set_description(f'Epoch {epoch}')
			loop.set_postfix(batchIndex=batchIndex, loss=loss, accuracy=accuracy)

		averageAccuracy = averageAccuracy/batchCount
		averageLoss = averageLoss/batchCount
		print("averageAccuracy = ", averageAccuracy)
		print("averageLoss = ", averageLoss)


def prepareModelTrainWrapper():
	if(useAlgorithmGIA):
		SBNLPpt_GIA.preparePOSdictionary()	#required for SBNLPpt_getAllPossiblePosTags.getAllPossiblePosTags(word)
	if(useMultipleModels):
		for vectorSpaceIndex, vectorSpace in enumerate(SBNLPpt_GIA.vectorSpaceList):
			vocabSize = vocabularySize
			if(encode3tuples):
				if(vectorSpace.intermediateType == SBNLPpt_GIA.intermediateTypePOS):
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
		for vectorSpaceIndex, vectorSpace in enumerate(SBNLPpt_GIA.vectorSpaceList):
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
				
		for vectorSpaceIndex, vectorSpace in enumerate(SBNLPpt_GIA.vectorSpaceList):
			model = vectorSpace.model
			optim = vectorSpace.optim
				
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
		
		for vectorSpace, vectorSpaceIndex in enumerate(SBNLPpt_GIA.vectorSpaceList):
			model = vectorSpace.model
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

