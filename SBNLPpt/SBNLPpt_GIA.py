"""SBNLPpt_GIA.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SBNLPpt_main.py

# Usage:
see SBNLPpt_main.py

# Description:
SBNLPpt GIA
	
"""

import torch as pt
import torch

from SBNLPpt_globalDefs import *
import SBNLPpt_data
import SBNLPpt_GIAmodel
import SBNLPpt_getAllPossiblePosTags
import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

embeddingLayerSize = 768	#word vector embedding size (cany vary based on GIA word vector space)

#nltk pos tags
nltkPOStagsConjunction = ["CC"]
nltkPOStagsNumber = ["CD"]
nltkPOStagsDeterminer = ["DT"]
nltkPOStagsExistentialThere = ["EX"]
nltkPOStagsForeignWord = ["FW"]
nltkPOStagsPreposition = ["IN"]
nltkPOStagsListMarker = ["LS"]
nltkPOStagsModal = ["MD"]
nltkPOStagsAdjective = ["JJ", "JJR", "JJS"]
nltkPOStagsNoun = ["NN", "NNP", "NNS"]
nltkPOStagsPredeterminer = ["PDT"]
nltkPOStagsPossessiveEnding = ["POS"]
nltkPOStagsPersonalPronoun = ["PRP"]
nltkPOStagsAdverb = ["RB", "RBR", "RBS"]
nltkPOStagsSymbol = ["SYM"]
nltkPOStagsPrepositionTo = ["TO"]
nltkPOStagsInterjection = ["UH"]
nltkPOStagsVerb = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
nltkPOStagsWhDeterminer = ["WDT"]
nltkPOStagsWhPossessivePronoun = ["WP$"]
nltkPOStagsWhAdverb = ["WRB"]

#semantic relation detection keypointPos/keypointWords
keypointPosNoun = nltkPOStagsNoun + nltkPOStagsPersonalPronoun
keypointPosVerb = nltkPOStagsVerb
keypointPosAdjective = nltkPOStagsAdjective
keypointPosAdverb = nltkPOStagsAdverb
keypointPosPreposition = nltkPOStagsPreposition + nltkPOStagsPrepositionTo
keypointWordAuxiliaryPossessive = ["have", "has", "had", "'s"]
keypointWordAuxiliaryBeingDefinition = [["is", "a"], ["is", "the"], "are"]
keypointWordAuxiliaryBeingQuality = ["am", "is", "are", "was", "were", "being", "been", "will be"]
keypointNone = None

detectionTypeAdjacent = 0
detectionTypeNearest = 1

intermediateTypeNone = 0
intermediateTypeWord = 1
intermediateTypePOS = 2

directionForward = 0
directionReverse = 1

keypointTypeNone = 0
keypointTypePOS = 1
keypointTypeWord = 2

keypointMaxDetectionDistance = 5 #max number of tokens between keypoints

class vectorSpaceProperties():
	def __init__(self, vectorSpaceName, direction, detectionType, intermediateType, keyPrior, keyAfter, keyIntermediate):
		self.vectorSpaceName = vectorSpaceName
		self.direction = direction
		self.detectionType = detectionType
		self.intermediateType = intermediateType
		self.keyPrior = keyPrior
		self.keyIntermediate = keyIntermediate
		self.keyAfter = keyAfter
		
		self.config = None
		self.model = None
		self.optim = None


if(debugUseSmallNumberOfModels):
	vectorSpaceList = [
	vectorSpaceProperties("action", directionForward, detectionTypeNearest, intermediateTypeNone, keypointPosNoun, keypointPosVerb, keypointNone)
	]
else:
	vectorSpaceList = [
	vectorSpaceProperties("definition", directionForward, detectionTypeNearest, intermediateTypeWord, keypointPosNoun, keypointPosNoun, keypointWordAuxiliaryBeingDefinition),
	vectorSpaceProperties("action", directionForward, detectionTypeNearest, intermediateTypeNone, keypointPosNoun, keypointPosVerb, keypointNone),
	vectorSpaceProperties("property", directionForward, detectionTypeNearest, intermediateTypeWord, keypointPosNoun, keypointPosNoun, keypointWordAuxiliaryPossessive),
	vectorSpaceProperties("quality1", directionForward, detectionTypeAdjacent, intermediateTypeWord, keypointPosNoun, keypointPosAdjective, keypointWordAuxiliaryBeingQuality),
	vectorSpaceProperties("quality2", directionForward, detectionTypeAdjacent, intermediateTypeNone, keypointPosAdjective, keypointPosNoun, keypointNone),
	vectorSpaceProperties("quality3", directionForward, detectionTypeAdjacent, intermediateTypeNone, keypointPosVerb, keypointPosAdverb, keypointNone),
	vectorSpaceProperties("actionSubjectObject", directionForward, detectionTypeNearest, intermediateTypeNone, keypointPosVerb, keypointPosNoun, keypointNone),
	vectorSpaceProperties("preposition", directionForward, detectionTypeNearest, intermediateTypePOS, keypointPosNoun, keypointPosNoun, keypointPosPreposition)
	]
	if(useIndependentReverseRelationsModels):
		vectorSpaceListR = [
			vectorSpaceProperties("definitionR", directionReverse, detectionTypeNearest, intermediateTypeWord, keypointPosNoun, keypointPosNoun, keypointWordAuxiliaryBeingDefinition),
			vectorSpaceProperties("actionR", directionReverse, detectionTypeNearest, intermediateTypeNone, keypointPosNoun, keypointPosVerb, keypointNone),
			vectorSpaceProperties("propertyR", directionReverse, detectionTypeNearest, intermediateTypeWord, keypointPosNoun, keypointPosNoun, keypointWordAuxiliaryPossessive),
			vectorSpaceProperties("quality1R", directionReverse, detectionTypeAdjacent, intermediateTypeWord, keypointPosNoun, keypointPosAdjective, keypointWordAuxiliaryBeingQuality),
			vectorSpaceProperties("quality2R", directionReverse, detectionTypeAdjacent, intermediateTypeNone, keypointPosAdjective, keypointPosNoun, keypointNone),
			vectorSpaceProperties("quality3R", directionReverse, detectionTypeAdjacent, intermediateTypeNone, keypointPosVerb, keypointPosAdverb, keypointNone),
			vectorSpaceProperties("actionSubjectObjectR", directionReverse, detectionTypeNearest, intermediateTypeNone, keypointPosVerb, keypointPosNoun, keypointNone),
			vectorSpaceProperties("prepositionR", directionReverse, detectionTypeNearest, intermediateTypePOS, keypointPosNoun, keypointPosNoun, keypointPosPreposition)
		]
		vectorSpaceList = vectorSpaceList + vectorSpaceListR


		
modelPathName = modelFolderName + '/modelGIA.pt'

def preparePOSdictionary():
	SBNLPpt_getAllPossiblePosTags.constructPOSdictionary()	#required for SBNLPpt_getAllPossiblePosTags.getAllPossiblePosTags(word)

def createModel(vocabSize):
	print("creating new model")
	config = SBNLPpt_GIAmodel.GIAconfig(vocabSize, embeddingLayerSize)
	model = SBNLPpt_GIAmodel.GIAmodel(config)
		
	return model

def loadModel():
	print("loading existing model")
	model = pt.load(modelPathName)
	return model
	
def saveModel(model):
	pt.save(model, modelPathName)

def propagate(device, model, tokenizer, labels):
	(xLabels, yLabels) = labels
	loss, outputs = model(xLabels, yLabels)
	accuracy = 0
	return loss, accuracy


def calculateXYlabels(tokenizer, vectorSpace, vectorSpaceIndex, batch, vocabSize):
	batchTokenIDs = batch['labels']
	modelSamplesX, modelSamplesY = getModelSamplesStart(tokenizer, batchTokenIDs, vectorSpace)

	if(debugDoNotTrainModel):
		xLabels = None
		yLabels = None
	else:
		xLabels = torch.Tensor(modelSamplesX).to(torch.long).to(device)
		yLabels = torch.Tensor(modelSamplesY).to(torch.long).to(device)
		xLabels = F.one_hot(xLabels, num_classes=vocabSize).to(torch.float)
		yLabels = F.one_hot(yLabels, num_classes=vocabSize).to(torch.float)
		#print(xLabels)
		#print(yLabels)
	
	labels = (xLabels, yLabels)
	return labels
	
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
	if((keypoint == keypointPosNoun) or (keypoint == keypointPosVerb) or (keypoint == keypointPosAdjective) or (keypoint == keypointPosAdverb) or (keypoint == keypointPosPreposition)):
		keypointType = keypointTypePOS
	elif((keypoint == keypointWordAuxiliaryPossessive) or (keypoint == keypointWordAuxiliaryBeingDefinition) or (keypoint == keypointWordAuxiliaryBeingQuality)):
		keypointType = keypointTypeWord
	elif((keypoint == keypointNone)):
		keypointType = keypointTypeNone
	else:
		print("getKeypointType error: keypointType unknown, keypoint = ", keypoint)
		exit()
	return keypointType

def getModelSamplesStart(tokenizer, batchTokenIDs, vectorSpace):	

	modelSamplesX = []
	modelSamplesY = []
	
	#batchSize = batchTokenIDs.shape[0]
	for sampleIndex in range(batchSize):
		sampleTokenIDsTensor = batchTokenIDs[sampleIndex]
		sampleTokenIDsList = sampleTokenIDsTensor.tolist()
		#print("sampleTokenIDsList = ", sampleTokenIDsList)
		textWordList = convertIDlistToTokensList(tokenizer, sampleTokenIDsList)

		getModelSamples(modelSamplesX, modelSamplesY, textWordList, vectorSpace, keypointIndex=0, startSearchIndex=0, endSearchIndex=len(textWordList), wordIndexKeypoint0=None)
	
	return modelSamplesX, modelSamplesY

def convertIDlistToTokensList(tokenizer, sampleTokenIDsList):
	if(useFullwordTokenizerClass):
		textWordList = tokenizer.convert_ids_to_tokens(sampleTokenIDsList)
	else:
		textWordList = [tokenizer.list[x] for x in sampleTokenIDsList]
	return textWordList
	
def getModelSamples(modelSamplesX, modelSamplesY, textWordList, vectorSpace, keypointIndex, startSearchIndex, endSearchIndex, wordIndexKeypoint0):
	keypoint = getKeypoint(vectorSpace, keypointIndex)
	keypointType = getKeypointType(keypoint)
	for wordIndex, word in enumerate(textWordList[startSearchIndex:endSearchIndex]):
		if(keypointIndex == 0):
			wordIndexKeypoint0 = wordIndex
		#print("word = ", word)
		keypointFound, keypointIndexLast = isKeypointFound(textWordList, keypointIndex, keypoint, keypointType, word, wordIndex)
		if(keypointFound):
			if(debugPrintRelationExtractionProgress):
				if(keypointIndex > 0):
					print("found keypoint, keypointIndex = ", keypointIndex, ", wordIndex = ", wordIndex)
			findNextKeypoint = False
			if(keypointIndex == 0):
				if(getKeypointType(getKeypoint(vectorSpace, 1)) == keypointTypeNone):
					keypointIndexN=2
				else:
					keypointIndexN=1
				findNextKeypoint = True
			elif(keypointIndex == 1):
				keypointIndexN=2
				findNextKeypoint = True
				
			if(findNextKeypoint):
				startSearchIndexN = keypointIndexLast+1
				if(vectorSpace.detectionType==detectionTypeNearest):
					endSearchIndexN = max([startSearchIndexN+keypointMaxDetectionDistance, len(textWordList)])
				elif(vectorSpace.detectionType==detectionTypeAdjacent):
					endSearchIndexN = startSearchIndexN+1
					
				getModelSamples(modelSamplesX, modelSamplesY, textWordList, vectorSpace, keypointIndexN, startSearchIndexN, endSearchIndexN, wordIndexKeypoint0)
			else:
				wordIndexKeypoint2 = wordIndex
				if(useIndependentReverseRelationsModels):
					if(vectorSpace.direction == directionForward):
						xTokenIndex = wordIndexKeypoint0
						yTokenIndex = wordIndexKeypoint2
					elif(vectorSpace.direction == directionReverse):
						xTokenIndex = wordIndexKeypoint2
						yTokenIndex = wordIndexKeypoint0
				else:		
					xTokenIndex = wordIndexKeypoint0
					yTokenIndex = wordIndexKeypoint2	
				modelSamplesX.append(xTokenIndex)
				modelSamplesY.append(yTokenIndex)

def keypointInPosList(keypoint, posValues):
	keypointPOSfound = False
	keypointSet = set(keypoint)
	posValuesSet = set(posValues)
	if(keypointSet & posValuesSet):
		keypointPOSfound = True
	return keypointPOSfound	
					
def isKeypointFound(textWordList, keypointIndex, keypoint, keypointType, word, wordIndex):
	keypointFound = False
	keypointIndexLast = wordIndex
	if(keypointType == keypointTypePOS):
		#keypointType POS
		posValues = SBNLPpt_getAllPossiblePosTags.getAllPossiblePosTags(word)
		#print("keypointType POS: posValues = ", posValues)
		if(keypointInPosList(keypoint, posValues)):
			keypointFound = True
			#print("keypointFound, keypointType POS: posValues = ", posValues)
	elif(keypointType == keypointTypeWord):
		#print("keypointType word: keypointType = ", keypointType)
		#keypointType word
		for keypointWordIndex, keypointWord in enumerate(keypoint):
			if(type(keypointWord) is list):
				for keypointWordIndex2, keypointWord2 in enumerate(keypointWord):
					foundKeypointWord = True
					wordCurrent = textWordList[wordIndex+keypointWordIndex2]
					if(wordCurrent != keypointWord2):
						foundKeypointWord = False
				if(foundKeypointWord):
					keypointFound = True
					keypointIndexLast = wordIndex+len(keypointWord)-1
					#print("keypointFound; type(keypointWord) is list, keypointType word: keypointType = ", keypointType, ", keypoint = ", keypoint)
			else:
				if(word == keypointWord):
					foundKeypointWord = True
					keypointFound = True
					#print("keypointFound, keypointType word: keypointType = ", keypointType)
	else:
		print("isKeypointFound error: keypointIndex = ", keypointIndex, ", keypointType = ", keypointType, ", keypoint = ", keypoint)
		exit()
	return keypointFound, keypointIndexLast
				

		

