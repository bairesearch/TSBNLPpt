"""TSBNLPpt_GIAsemanticRelationVectorised.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see TSBNLPpt_main.py

# Usage:
see TSBNLPpt_main.py

# Description:
TSBNLPpt GIA semantic relation standard
	
"""

import torch as pt
from torch import nn
import torch

from TSBNLPpt_globalDefs import *
import TSBNLPpt_data
import TSBNLPpt_GIAmodel
import torch.nn.functional as F
import math

import TSBNLPpt_GIAvectorSpaces

keypoint1offset = 1
	
def calculateXYlabels(tokenizer, batch, vocabSize, posVectorList):
	batchTokenIDs = batch['labels']
	
	batchPOSflagsList = prepareBatchPOSflagsList(batchTokenIDs, posVectorList)
	labelsList, labelsFoundList = getModelSamples(batchTokenIDs, posVectorList, batchPOSflagsList, vocabSize)
	
	return labelsList, labelsFoundList

def prepareBatchPOSflagsList(batchTokenIDs, posVectorList):
	batchPOSflagsList = []
	for POSindex, POSitem in enumerate(TSBNLPpt_GIAvectorSpaces.keypointsDict.items()):
		posVector = posVectorList[POSindex]	
		batchPOSflags = torch.zeros(batchTokenIDs.shape)	#dtype=torch.bool
		for sampleIndex in range(batchSize):
			for tokenIndex in range(sequenceMaxNumTokens):
				sampleToken = batchTokenIDs[sampleIndex][tokenIndex]	#token ID
				posFlag = posVector[sampleToken]	#.bool()
				batchPOSflags[sampleIndex][tokenIndex] = posFlag
		batchPOSflagsList.append(batchPOSflags)
	return batchPOSflagsList

def getModelSamples(batchTokenIDs, posVectorList, batchPOSflagsList, vocabSize):
	semanticRelationIDinputKeypoint0List = []
	semanticRelationIDinputKeypoint1List = []
	semanticRelationIDinputKeypoint2List = []
	for vectorSpaceIndex, vectorSpace in enumerate(TSBNLPpt_GIAvectorSpaces.vectorSpaceList):
		batchPOSflagsKeypoint0 = getBatchPOSflags(vectorSpace.keyPrior, batchPOSflagsList, vocabSize)
		batchPOSflagsKeypoint1 = getBatchPOSflags(vectorSpace.keyIntermediate, batchPOSflagsList, vocabSize)
		batchPOSflagsKeypoint2 = getBatchPOSflags(vectorSpace.keyAfter, batchPOSflagsList, vocabSize)
		semanticRelationIDinputKeypoint0List.append(batchPOSflagsKeypoint0)
		semanticRelationIDinputKeypoint1List.append(batchPOSflagsKeypoint1)
		semanticRelationIDinputKeypoint2List.append(batchPOSflagsKeypoint2)
	semanticRelationIDinputKeypoint0 = torch.stack(semanticRelationIDinputKeypoint0List, dim=1)	#C
	semanticRelationIDinputKeypoint1 = torch.stack(semanticRelationIDinputKeypoint1List, dim=1)	#C
	semanticRelationIDinputKeypoint2 = torch.stack(semanticRelationIDinputKeypoint2List, dim=1)	#C
	
	result = semanticRelationIdentification(semanticRelationIDinputKeypoint0, semanticRelationIDinputKeypoint1, semanticRelationIDinputKeypoint2)

	labelsList = []
	labelsFoundList = []
	for vectorSpaceIndex, vectorSpace in enumerate(TSBNLPpt_GIAvectorSpaces.vectorSpaceList):
		modelSamplesX = []
		modelSamplesY = []
		modelSamplesI = []
		for sampleIndex in range(batchSize):
			for tokenIndex in range(sequenceMaxNumTokens):
				if(result[sampleIndex, vectorSpaceIndex, tokenIndex] > 0):
					wordIndexKeypoint0 = batchTokenIDs[sampleIndex, tokenIndex]
					wordIndexKeypoint1 = batchTokenIDs[sampleIndex, tokenIndex+keypoint1offset]
					keypoint2offset = result[sampleIndex, vectorSpaceIndex, tokenIndex]	#wordIndexKeypoint2sequenceIndexOffset
					wordIndexKeypoint2 = batchTokenIDs[sampleIndex, tokenIndex+keypoint2offset]
					TSBNLPpt_GIAvectorSpaces.addModelSampleToList(vectorSpace, modelSamplesX, modelSamplesY, modelSamplesI, wordIndexKeypoint0, wordIndexKeypoint1, wordIndexKeypoint2)
		labels, labelsFound = TSBNLPpt_GIAvectorSpaces.createXYlabelsFromModelSampleList(vectorSpace, modelSamplesX, modelSamplesY, modelSamplesI, vocabSize)
		labelsList.append(labels)
		labelsFoundList.append(labelsFound)
		
	return labelsList, labelsFoundList

def semanticRelationIdentification(semanticRelationIDinputKeypoint0, semanticRelationIDinputKeypoint1, semanticRelationIDinputKeypoint2):
	keypoint0 = semanticRelationIDinputKeypoint0 
	keypoint0 = keypoint0.bool()
	keypoint1 = semanticRelationIDinputKeypoint1[:, :, keypoint1offset:]
	keypoint1 = F.pad(keypoint1, pad=(0, keypoint1offset), value=0)	#pad last dim end by keypoint1offset with zeros
	keypoint1 = keypoint1.bool()
	keypoint2PartList = []
	for x in range(TSBNLPpt_GIAvectorSpaces.keypointMaxDetectionDistance):
		keypoint2offset = keypoint1offset+x
		keypoint2Part = semanticRelationIDinputKeypoint2[:, :, keypoint2offset:]
		keypoint2Part = F.pad(keypoint2Part, pad=(0, keypoint2offset), value=0)	#pad last dim end by keypoint2offset with zeros
		keypoint2Part = keypoint2Part * (TSBNLPpt_GIAvectorSpaces.keypointMaxDetectionDistance-x)	#keypoint2Part * 5, 4, 3, 2, 1
		keypoint2PartList.append(keypoint2Part)
	keypoint2 = torch.stack(keypoint2PartList, dim=3)
	keypoint2index = torch.argmax(keypoint2, dim=3)	#get relative sequence index of keypoint2 (if existent)
	keypoint2 = torch.gt(keypoint2index, 0)
	
	#print("keypoint0 = ", keypoint0)
	#print("keypoint1 = ", keypoint1)
	#print("keypoint2 = ", keypoint2)
	#print("keypoint2index = ", keypoint2index)
	
	result = torch.logical_and(keypoint0, keypoint1)
	result = torch.logical_and(result, keypoint2)	

	result = result.int()	#FUTURE: consider using int64 to support large keypointMaxDetectionDistance
	result = result * keypoint2index	#return offset of keypoint2 (or 0 if semantic relation not found)
	#print("result = ", result)

	return result

def getBatchPOSflags(keypoint, batchPOSflagsList, vocabSize):
	if(keypoint == TSBNLPpt_GIAvectorSpaces.keypointNone):
		batchPOSflags = torch.ones(batchPOSflagsList[0].shape, requires_grad=False)
	else:
		POSvectorIndex = list(TSBNLPpt_GIAvectorSpaces.keypointsDict.keys()).index(keypoint)
		batchPOSflags = batchPOSflagsList[POSvectorIndex]
	return batchPOSflags
	
#NOTUSED;			
def getPOSvectorList(keypoint, posVectorList, vocabSize):
	if(keypoint == TSBNLPpt_GIAvectorSpaces.keypointNone):
		posVector = torch.ones([vocabSize])
	else:
		POSvectorIndex = list(TSBNLPpt_GIAvectorSpaces.keypointsDict.keys()).index(keypoint)
		posVector = posVectorList[POSvectorIndex]
	return posVector
