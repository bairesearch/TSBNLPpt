"""TSBNLPpt_GIAsemanticRelationStandard.py

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
import torch

from TSBNLPpt_globalDefs import *
import TSBNLPpt_POSgetAllPossiblePosTags
import torch.nn.functional as F
import TSBNLPpt_GIAvectorSpaces

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def preparePOSdictionary():
	TSBNLPpt_POSgetAllPossiblePosTags.constructPOSdictionary()	#required for TSBNLPpt_POSgetAllPossiblePosTags.getAllPossiblePosTags(word)

def calculateXYlabels(tokenizer, vectorSpace, vectorSpaceIndex, batch, vocabSize):
	batchTokenIDs = batch['labels']
	modelSamplesX, modelSamplesY, modelSamplesI = getModelSamplesStart(tokenizer, batchTokenIDs, vectorSpace)
	labels, labelsFound = TSBNLPpt_GIAvectorSpaces.createXYlabelsFromModelSampleList(vectorSpace, modelSamplesX, modelSamplesY, modelSamplesI, vocabSize)
	return labels, labelsFound
	
def getKeypoint(vectorSpace, keypointIndex):
	if(keypointIndex == 0):
		keypoint = vectorSpace.keyPrior
	elif(keypointIndex == 1):
		keypoint = vectorSpace.keyIntermediate
	elif(keypointIndex == 2):
		keypoint = vectorSpace.keyAfter
	else:
		print("getKeypoint error: keypointIndex unknown, keypointIndex = ", keypointIndex)
		exit()
	return keypoint

def getKeypointType(keypoint):
	if((keypoint == TSBNLPpt_GIAvectorSpaces.keypointPosNoun) or (keypoint == TSBNLPpt_GIAvectorSpaces.keypointPosVerb) or (keypoint == TSBNLPpt_GIAvectorSpaces.keypointPosAdjective) or (keypoint == TSBNLPpt_GIAvectorSpaces.keypointPosAdverb) or (keypoint == TSBNLPpt_GIAvectorSpaces.keypointPosPreposition)):
		keypointType = TSBNLPpt_GIAvectorSpaces.keypointTypePOS
	elif((keypoint == TSBNLPpt_GIAvectorSpaces.keypointWordAuxiliaryPossessive) or (keypoint == TSBNLPpt_GIAvectorSpaces.keypointWordAuxiliaryBeingDefinition) or (keypoint == TSBNLPpt_GIAvectorSpaces.keypointWordAuxiliaryBeingQuality)):
		keypointType = TSBNLPpt_GIAvectorSpaces.keypointTypeWord
	elif((keypoint == TSBNLPpt_GIAvectorSpaces.keypointNone)):
		keypointType = TSBNLPpt_GIAvectorSpaces.keypointTypeNone
	else:
		print("getKeypointType error: keypointType unknown, keypoint = ", keypoint)
		exit()
	return keypointType

def getModelSamplesStart(tokenizer, batchTokenIDs, vectorSpace):

	modelSamplesX = []
	modelSamplesY = []
	modelSamplesI = []
	
	#batchSize = batchTokenIDs.shape[0]
	for sampleIndex in range(batchSize):
		sampleTokenIDsTensor = batchTokenIDs[sampleIndex]
		sampleTokenIDsList = sampleTokenIDsTensor.tolist()
		textWordList = convertIDlistToTokensList(tokenizer, sampleTokenIDsList)

		getModelSamples(modelSamplesX, modelSamplesY, modelSamplesI, textWordList, vectorSpace, keypointIndex=0, startSearchIndex=0, endSearchIndex=len(textWordList), wordIndexKeypoint0=None, wordIndexKeypoint1=None)
	
	return modelSamplesX, modelSamplesY, modelSamplesI

def convertIDlistToTokensList(tokenizer, sampleTokenIDsList):
	if(useFullwordTokenizerClass):
		textWordList = tokenizer.convert_ids_to_tokens(sampleTokenIDsList)
		#print("sampleTokenIDsList = ", sampleTokenIDsList)
	else:
		textWordList = [tokenizer.list[x] for x in sampleTokenIDsList]
	return textWordList
	
def getModelSamples(modelSamplesX, modelSamplesY, modelSamplesI, textWordList, vectorSpace, keypointIndex, startSearchIndex, endSearchIndex, wordIndexKeypoint0, wordIndexKeypoint1):
	keypoint = getKeypoint(vectorSpace, keypointIndex)
	keypointType = getKeypointType(keypoint)
	for wordIndex, word in enumerate(textWordList[startSearchIndex:endSearchIndex]):
		if(keypointIndex == 0):
			wordIndexKeypoint0 = wordIndex
		if(keypointIndex == 1):
			wordIndexKeypoint1 = wordIndex
		#print("word = ", word)
		keypointFound, keypointIndexLast = isKeypointFound(textWordList, keypointIndex, keypoint, keypointType, word, wordIndex)
		if(keypointFound):
			if(debugPrintRelationExtractionProgress):
				if(keypointIndex > 0):
					print("found keypoint, keypointIndex = ", keypointIndex, ", wordIndex = ", wordIndex)
			findNextKeypoint = False
			if(keypointIndex == 0):
				if(getKeypointType(getKeypoint(vectorSpace, 1)) == TSBNLPpt_GIAvectorSpaces.keypointTypeNone):
					keypointIndexN=2
				else:
					keypointIndexN=1
				findNextKeypoint = True
			elif(keypointIndex == 1):
				keypointIndexN=2
				findNextKeypoint = True
				
			if(findNextKeypoint):
				startSearchIndexN = keypointIndexLast+1
				if(vectorSpace.detectionType==TSBNLPpt_GIAvectorSpaces.detectionTypeNearest):
					endSearchIndexN = max([startSearchIndexN+TSBNLPpt_GIAvectorSpaces.keypointMaxDetectionDistance, len(textWordList)])
				elif(vectorSpace.detectionType==TSBNLPpt_GIAvectorSpaces.detectionTypeAdjacent):
					endSearchIndexN = startSearchIndexN+1
					
				getModelSamples(modelSamplesX, modelSamplesY, modelSamplesI, textWordList, vectorSpace, keypointIndexN, startSearchIndexN, endSearchIndexN, wordIndexKeypoint0, wordIndexKeypoint1)
			else:
				wordIndexKeypoint2 = wordIndex
				TSBNLPpt_GIAvectorSpaces.addModelSampleToList(vectorSpace, modelSamplesX, modelSamplesY, modelSamplesI, wordIndexKeypoint0, wordIndexKeypoint1, wordIndexKeypoint2)
					
def isKeypointFound(textWordList, keypointIndex, keypoint, keypointType, word, wordIndex):
	keypointObject = TSBNLPpt_GIAvectorSpaces.keypointsDict[keypoint]
	keypointFound = False
	keypointIndexLast = wordIndex
	if(keypointType == TSBNLPpt_GIAvectorSpaces.keypointTypePOS):
		#keypointType POS
		posValues = TSBNLPpt_POSgetAllPossiblePosTags.getAllPossiblePosTags(word)
		#print("keypointType POS: posValues = ", posValues)
		if(TSBNLPpt_GIAvectorSpaces.isAnyPosListValueInPosList(keypointObject, posValues)):
			keypointFound = True
			#print("keypointFound, keypointType POS: posValues = ", posValues)
	elif(keypointType == TSBNLPpt_GIAvectorSpaces.keypointTypeWord):
		#print("keypointType word: keypointType = ", keypointType)
		#keypointType word
		for keypointWordIndex, keypointWord in enumerate(keypointObject):
			if(type(keypointWord) is list):
				for keypointWordIndex2, keypointWord2 in enumerate(keypointWord):
					foundKeypointWord = True
					wordCurrent = textWordList[wordIndex+keypointWordIndex2]
					if(wordCurrent != keypointWord2):
						foundKeypointWord = False
				if(foundKeypointWord):
					keypointFound = True
					keypointIndexLast = wordIndex+len(keypointWord)-1
					#print("keypointFound; type(keypointWord) is list, keypointType word: keypointType = ", keypointType, ", keypointObject = ", keypointObject)
			else:
				if(word == keypointWord):
					foundKeypointWord = True
					keypointFound = True
					#print("keypointFound, keypointType word: keypointType = ", keypointType)
	else:
		print("isKeypointFound error: keypointIndex = ", keypointIndex, ", keypointType = ", keypointType, ", keypointObject = ", keypointObject)
		exit()
	return keypointFound, keypointIndexLast
				

		

