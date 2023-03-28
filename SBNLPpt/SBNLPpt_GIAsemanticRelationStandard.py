"""SBNLPpt_GIAsemanticRelationStandard.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SBNLPpt_main.py

# Usage:
see SBNLPpt_main.py

# Description:
SBNLPpt GIA semantic relation standard
	
"""

import torch as pt
import torch

from SBNLPpt_globalDefs import *
import SBNLPpt_POSgetAllPossiblePosTags
import torch.nn.functional as F
import SBNLPpt_GIAvectorSpaces

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def preparePOSdictionary():
	SBNLPpt_POSgetAllPossiblePosTags.constructPOSdictionary()	#required for SBNLPpt_POSgetAllPossiblePosTags.getAllPossiblePosTags(word)

def calculateXYlabels(tokenizer, vectorSpace, vectorSpaceIndex, batch, vocabSize):
	batchTokenIDs = batch['labels']
	modelSamplesX, modelSamplesY, modelSamplesI = getModelSamplesStart(tokenizer, batchTokenIDs, vectorSpace)
	labels, labelsFound = SBNLPpt_GIAvectorSpaces.createXYlabelsFromModelSampleList(vectorSpace, modelSamplesX, modelSamplesY, modelSamplesI, vocabSize)
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
	if((keypoint == SBNLPpt_GIAvectorSpaces.keypointPosNoun) or (keypoint == SBNLPpt_GIAvectorSpaces.keypointPosVerb) or (keypoint == SBNLPpt_GIAvectorSpaces.keypointPosAdjective) or (keypoint == SBNLPpt_GIAvectorSpaces.keypointPosAdverb) or (keypoint == SBNLPpt_GIAvectorSpaces.keypointPosPreposition)):
		keypointType = SBNLPpt_GIAvectorSpaces.keypointTypePOS
	elif((keypoint == SBNLPpt_GIAvectorSpaces.keypointWordAuxiliaryPossessive) or (keypoint == SBNLPpt_GIAvectorSpaces.keypointWordAuxiliaryBeingDefinition) or (keypoint == SBNLPpt_GIAvectorSpaces.keypointWordAuxiliaryBeingQuality)):
		keypointType = SBNLPpt_GIAvectorSpaces.keypointTypeWord
	elif((keypoint == SBNLPpt_GIAvectorSpaces.keypointNone)):
		keypointType = SBNLPpt_GIAvectorSpaces.keypointTypeNone
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
				if(getKeypointType(getKeypoint(vectorSpace, 1)) == SBNLPpt_GIAvectorSpaces.keypointTypeNone):
					keypointIndexN=2
				else:
					keypointIndexN=1
				findNextKeypoint = True
			elif(keypointIndex == 1):
				keypointIndexN=2
				findNextKeypoint = True
				
			if(findNextKeypoint):
				startSearchIndexN = keypointIndexLast+1
				if(vectorSpace.detectionType==SBNLPpt_GIAvectorSpaces.detectionTypeNearest):
					endSearchIndexN = max([startSearchIndexN+SBNLPpt_GIAvectorSpaces.keypointMaxDetectionDistance, len(textWordList)])
				elif(vectorSpace.detectionType==SBNLPpt_GIAvectorSpaces.detectionTypeAdjacent):
					endSearchIndexN = startSearchIndexN+1
					
				getModelSamples(modelSamplesX, modelSamplesY, modelSamplesI, textWordList, vectorSpace, keypointIndexN, startSearchIndexN, endSearchIndexN, wordIndexKeypoint0, wordIndexKeypoint1)
			else:
				wordIndexKeypoint2 = wordIndex
				SBNLPpt_GIAvectorSpaces.addModelSampleToList(vectorSpace, modelSamplesX, modelSamplesY, modelSamplesI, wordIndexKeypoint0, wordIndexKeypoint1, wordIndexKeypoint2)
					
def isKeypointFound(textWordList, keypointIndex, keypoint, keypointType, word, wordIndex):
	keypointObject = SBNLPpt_GIAvectorSpaces.keypointsDict[keypoint]
	keypointFound = False
	keypointIndexLast = wordIndex
	if(keypointType == SBNLPpt_GIAvectorSpaces.keypointTypePOS):
		#keypointType POS
		posValues = SBNLPpt_POSgetAllPossiblePosTags.getAllPossiblePosTags(word)
		#print("keypointType POS: posValues = ", posValues)
		if(SBNLPpt_GIAvectorSpaces.isAnyPosListValueInPosList(keypointObject, posValues)):
			keypointFound = True
			#print("keypointFound, keypointType POS: posValues = ", posValues)
	elif(keypointType == SBNLPpt_GIAvectorSpaces.keypointTypeWord):
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
				

		

