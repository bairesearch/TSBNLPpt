"""SBNLPpt_GIAdefinePOSwordLists.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SBNLPpt_main.py

# Usage:
see SBNLPpt_main.py

# Description:
SBNLPpt GIA define POS wordLists

# Usage (generateWordlists for LRPdata):
source activate transformersenv
python SBNLPpt_GIAdefinePOSwordLists.py

"""

import torch as pt
import torch

from SBNLPpt_globalDefs import *
import SBNLPpt_getAllPossiblePosTags
import torch.nn.functional as F

import nltk
from nltk.corpus import words as NLTKwords

import SBNLPpt_getAllPossiblePosTags

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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

#auxiliary relation words
wordAuxiliaryHavingPossessive = ["have", "has", "had", "'s"]
if(useVectorisedSemanticRelationIdentification):
	wordAuxiliaryBeingDefinition = ["is", "are"]	#useVectorisedSemanticRelationIdentification does not currently support multiword relations
else:
	wordAuxiliaryBeingDefinition = [["is", "a"], ["is", "the"], "are"]
wordAuxiliaryBeingQuality = ["am", "is", "are", "was", "were", "being", "been", "be"]	#will be

nltkPOStagsDict = {
"Conjunction": nltkPOStagsConjunction,
"Number":  nltkPOStagsNumber,
"Determiner":  nltkPOStagsDeterminer,
"ExistentialThere":  nltkPOStagsExistentialThere,
"ForeignWord": nltkPOStagsForeignWord,
"Preposition": nltkPOStagsPreposition,
"ListMarker": nltkPOStagsListMarker,
"Modal": nltkPOStagsModal,
"Adjective": nltkPOStagsAdjective,
"Noun": nltkPOStagsNoun,
"Predeterminer": nltkPOStagsPredeterminer,
"PossessiveEnding": nltkPOStagsPossessiveEnding,
"PersonalPronoun": nltkPOStagsPersonalPronoun,
"Adverb": nltkPOStagsAdverb,
"Symbol": nltkPOStagsSymbol,
"PrepositionTo": nltkPOStagsPrepositionTo,
"Interjection": nltkPOStagsInterjection,
"Verb": nltkPOStagsVerb,
"WhDeterminer": nltkPOStagsWhDeterminer,
"WhPossessivePronoun": nltkPOStagsWhPossessivePronoun,
"WhAdverb": nltkPOStagsWhAdverb
}
wordListAllname = "All"

wordAuxiliaryRelationDict = {
"AuxiliaryHavingPossessive": wordAuxiliaryHavingPossessive,
"AuxiliaryBeingDefinition": wordAuxiliaryBeingDefinition,
"AuxiliaryBeingQuality": wordAuxiliaryBeingQuality
}

#semantic relation detection keypointPos/keypointWords
keypointPosNoun = "Noun" #+ PersonalPronoun
keypointPosVerb = "Verb"
keypointPosAdjective = "Adjective"
keypointPosAdverb = "Adverb"
keypointPosPreposition = "Preposition" #+ nltkPOStagsPrepositionTo
keypointWordAuxiliaryPossessive = "AuxiliaryHavingPossessive"
keypointWordAuxiliaryBeingDefinition = "AuxiliaryBeingDefinition"
keypointWordAuxiliaryBeingQuality = "AuxiliaryBeingQuality"
keypointNone = None

keypointsDict = {
keypointPosNoun: nltkPOStagsNoun,
keypointPosVerb: nltkPOStagsVerb,
keypointPosAdjective: nltkPOStagsWhDeterminer,
keypointPosAdverb: nltkPOStagsWhPossessivePronoun,
keypointPosPreposition: nltkPOStagsWhAdverb,
keypointWordAuxiliaryPossessive: wordAuxiliaryHavingPossessive,
keypointWordAuxiliaryBeingDefinition: wordAuxiliaryBeingDefinition,
keypointWordAuxiliaryBeingQuality: wordAuxiliaryBeingQuality
}

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
#if(useVectorisedSemanticRelationIdentification):
#	if(keypointMaxDetectionDistance%2 == 0):
#		print("SBNLPpt_GIAsemanticRelationVectorised error: keypointMaxDetectionDistance must be an odd number (Conv1d kernel size)")
#		exit()
			
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


#useVectorisedSemanticRelationIdentification: detectionTypeNearest is not supported for keyIntermediate (detectionTypeAdjacent only)

if(debugUseSmallNumberOfModels):
	vectorSpaceList = [
	vectorSpaceProperties("action", directionForward, detectionTypeNearest, intermediateTypeNone, keypointPosNoun, keypointPosVerb, keypointNone)
	]
else:
	vectorSpaceList = [
	vectorSpaceProperties("definition", directionForward, detectionTypeNearest, intermediateTypeWord, keypointPosNoun, keypointPosNoun, keypointWordAuxiliaryBeingDefinition),
	vectorSpaceProperties("action", directionForward, detectionTypeNearest, intermediateTypePOS, keypointPosNoun, keypointPosNoun, keypointPosVerb),
	vectorSpaceProperties("actionSubject", directionForward, detectionTypeNearest, intermediateTypeNone, keypointPosNoun, keypointPosVerb, keypointNone),
	vectorSpaceProperties("actionObject", directionForward, detectionTypeNearest, intermediateTypeNone, keypointPosVerb, keypointPosNoun, keypointNone),
	vectorSpaceProperties("property", directionForward, detectionTypeNearest, intermediateTypeWord, keypointPosNoun, keypointPosNoun, keypointWordAuxiliaryPossessive),
	vectorSpaceProperties("qualitySubstance1", directionForward, detectionTypeAdjacent, intermediateTypeWord, keypointPosNoun, keypointPosAdjective, keypointWordAuxiliaryBeingQuality),
	vectorSpaceProperties("qualitySubstance2", directionForward, detectionTypeAdjacent, intermediateTypeNone, keypointPosAdjective, keypointPosNoun, keypointNone),
	vectorSpaceProperties("qualityAction1", directionForward, detectionTypeAdjacent, intermediateTypeNone, keypointPosVerb, keypointPosAdverb, keypointNone),
	vectorSpaceProperties("preposition", directionForward, detectionTypeNearest, intermediateTypePOS, keypointPosNoun, keypointPosNoun, keypointPosPreposition)
	]
	if(useIndependentReverseRelationsModels):
		vectorSpaceListR = [
			vectorSpaceProperties("definitionR", directionReverse, detectionTypeNearest, intermediateTypeWord, keypointPosNoun, keypointPosNoun, keypointWordAuxiliaryBeingDefinition),
			vectorSpaceProperties("actionR", directionForward, detectionTypeNearest, intermediateTypePOS, keypointPosNoun, keypointPosNoun, keypointPosVerb),
			vectorSpaceProperties("actionSubjectR", directionReverse, detectionTypeNearest, intermediateTypeNone, keypointPosNoun, keypointPosVerb, keypointNone),
			vectorSpaceProperties("actionObjectR", directionReverse, detectionTypeNearest, intermediateTypeNone, keypointPosVerb, keypointPosNoun, keypointNone),
			vectorSpaceProperties("propertyR", directionReverse, detectionTypeNearest, intermediateTypeWord, keypointPosNoun, keypointPosNoun, keypointWordAuxiliaryPossessive),
			vectorSpaceProperties("qualitySubstance1R", directionReverse, detectionTypeAdjacent, intermediateTypeWord, keypointPosNoun, keypointPosAdjective, keypointWordAuxiliaryBeingQuality),
			vectorSpaceProperties("qualitySubstance2R", directionReverse, detectionTypeAdjacent, intermediateTypeNone, keypointPosAdjective, keypointPosNoun, keypointNone),
			vectorSpaceProperties("qualityAction1R", directionReverse, detectionTypeAdjacent, intermediateTypeNone, keypointPosVerb, keypointPosAdverb, keypointNone),
			vectorSpaceProperties("prepositionR", directionReverse, detectionTypeNearest, intermediateTypePOS, keypointPosNoun, keypointPosNoun, keypointPosPreposition)
		]
		vectorSpaceList = vectorSpaceList + vectorSpaceListR
		
		

def generateWordlists():
	SBNLPpt_getAllPossiblePosTags.constructPOSdictionary()	#required for SBNLPpt_getAllPossiblePosTags.getAllPossiblePosTags(word)
	generatePOSwordLists()
	generatePOSwordListVectors(vocabularySize)
	
def generatePOSwordLists():	
	numberOfPOSwordLists = len(nltkPOStagsDict)
	posWordLists = [[] for x in range(numberOfPOSwordLists)]
	for word in NLTKwords.words():
		wordPosValues = SBNLPpt_getAllPossiblePosTags.getAllPossiblePosTags(word)
		for POSindex, POSitem in enumerate(nltkPOStagsDict.items()):
			POSname = POSitem[0]
			POStags = POSitem[1]
			if(isAnyPosListValueInPosList(wordPosValues, POStags)):
				posWordLists[POSindex].append(word)
				#print("word = ", word)
				#print("wordPosValues = ", wordPosValues)
				#print("POSname = ", POSname)
				#print("POStags = ", POStags)
	for posWordIndex, posWordList in enumerate(posWordLists):
		POSname = list(nltkPOStagsDict)[posWordIndex]
		posWordList = posWordLists[posWordIndex]
		posWordList = list(set(posWordList))	#CHECKTHIS: remove duplicates
		writeWordList(POSname, posWordList)
	
	posWordListAll = getAllNLTKwords()
	writeWordList(wordListAllname, posWordListAll)

def getAllNLTKwords():
	posWordListAll = []
	for word in NLTKwords.words():
		posWordListAll.append(word)
	if(fixNLTKwordListAll):
		posWordListAll.extend(["has", "having", "'s"])
	#print("getAllNLTKwords: len(posWordListAll) = ", len(posWordListAll))
	return posWordListAll
					
def isAnyPosListValueInPosList(posList, posValues):
	POSfound = False
	posListSet = set(posList)
	posValuesSet = set(posValues)
	if(posListSet & posValuesSet):
		POSfound = True
	return POSfound
			
def writeWordList(POSname, wordList):
	wordlistFileName = generateWordlistFileName("wordlist" + POSname)
	with open(wordlistFileName, 'w') as f:
		f.write("\n".join(wordList))
		
def readWordList(POSname):
	wordList = []
	wordlistFileName = generateWordlistFileName("wordlist" + POSname)
	with open(wordlistFileName, 'r') as f:
		wordList = f.read().splitlines()	#or eval(f.read())
		#wordList = eval(f.read())
	return wordList

def generateWordlistFileName(fileName):
	print("fileName = ", fileName)
	wordlistFileName = LRPfolderName + "/" + fileName + ".txt"
	return wordlistFileName
	
def createDictionaryItemsFromList(lst, startIndex):
	list1 = lst
	list2 = range(startIndex, len(lst)+startIndex)
	dictionaryItems = zip(list1, list2)
	return dictionaryItems

POSwordListVectorValueFalse = "0"
POSwordListVectorValueTrue = "1"
def generatePOSwordListVectors(vocabSize):
	#posVectorListList = []
	POSwordListAll = readWordList(wordListAllname)
	POSwordDictAllItems = createDictionaryItemsFromList(POSwordListAll, 0)
	POSwordDictAll = dict(POSwordDictAllItems)
	for POSindex, POSitem in enumerate(keypointsDict.items()):
		posList = [POSwordListVectorValueFalse]*vocabSize
		POSname = POSitem[0]
		POSwordList = readWordList(POSname)
		for POSword in POSwordList:
			POSwordIndex = POSwordDictAll[POSword]
			posList[POSwordIndex] = POSwordListVectorValueTrue
		#posVectorListList.append(posVector)
		posVectorFileName = "Vector" + POSname
		writeWordList(posVectorFileName, posList)

if(__name__ == '__main__'):
	generateWordlists()


def loadPOSwordListVectors():
	posVectorList = []
	for POSindex, POSitem in enumerate(keypointsDict.items()):
		POSname = POSitem[0]
		posVectorFileName = "Vector" + POSname
		posList = readWordList(posVectorFileName)
		posList = [int(i) for i in posList]	#convert list to ints
		posVector = torch.tensor(posList, requires_grad=False)	#.bool()
		posVectorList.append(posVector)
	return posVectorList


def addModelSampleToList(vectorSpace, modelSamplesX, modelSamplesY, modelSamplesI, wordIndexKeypoint0, wordIndexKeypoint1, wordIndexKeypoint2):
	if(useIndependentReverseRelationsModels):
		if(vectorSpace.direction == directionForward):
			xTokenIndex = wordIndexKeypoint0
			yTokenIndex = wordIndexKeypoint2
			iTokenIndex = wordIndexKeypoint1
		elif(vectorSpace.direction == directionReverse):
			xTokenIndex = wordIndexKeypoint2
			yTokenIndex = wordIndexKeypoint0
			iTokenIndex = wordIndexKeypoint1
	else:		
		xTokenIndex = wordIndexKeypoint0
		yTokenIndex = wordIndexKeypoint2	
		iTokenIndex = wordIndexKeypoint1

	if(not debugTruncateBatch or len(modelSamplesX)==0):
		modelSamplesX.append(xTokenIndex)
		modelSamplesY.append(yTokenIndex)
		modelSamplesI.append(iTokenIndex)

def createXYlabelsFromModelSampleList(vectorSpace, modelSamplesX, modelSamplesY, modelSamplesI, vocabSize):
	labelsFound = False
	if(debugDoNotTrainModel):
		labels = None
	else:
		if(len(modelSamplesX) > 0):
			labelsFound = True
			xLabels = torch.Tensor(modelSamplesX).to(torch.long).to(device)
			xLabels = F.one_hot(xLabels, num_classes=vocabSize).to(torch.float)
			yLabels = torch.Tensor(modelSamplesY).to(torch.long).to(device)
			yLabels = F.one_hot(yLabels, num_classes=vocabSize).to(torch.float)
			if(encode3tuples):
				if(vectorSpace.intermediateType == intermediateTypePOS):
					iLabels = torch.Tensor(modelSamplesI).to(torch.long).to(device)
					iLabels = F.one_hot(iLabels, num_classes=vocabSize).to(torch.float)
					xLabels = torch.concat((xLabels, iLabels), dim=1)
					yLabels = torch.concat((yLabels, iLabels), dim=1)
			labels = (xLabels, yLabels)
		else:
			labels = None
	return labels, labelsFound
	
