"""TSBNLPpt_POSwordLists.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see TSBNLPpt_main.py

# Usage:
see TSBNLPpt_main.py

# Description:
TSBNLPpt POS wordLists

# Usage (generateWordlists for LRPdata):
python	
import nltk
nltk.download('words')
nltk.download('brown')
Ctrl-D

source activate transformersenv
python TSBNLPpt_POSwordLists.py

"""

import torch as pt
import torch

from TSBNLPpt_globalDefs import *
import TSBNLPpt_POSgetAllPossiblePosTags
import torch.nn.functional as F

import nltk
from nltk.corpus import words as NLTKwords

import TSBNLPpt_POSgetAllPossiblePosTags


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
if(GIAuseVectorisedPOSidentification):
	wordAuxiliaryBeingDefinition = ["is", "are"]	#GIAuseVectorisedSemanticRelationIdentification does not currently support multiword relations
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

wordListVectorsDictAll = {
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
"WhAdverb": nltkPOStagsWhAdverb,
"AuxiliaryHavingPossessive": wordAuxiliaryHavingPossessive,
"AuxiliaryBeingDefinition": wordAuxiliaryBeingDefinition,
"AuxiliaryBeingQuality": wordAuxiliaryBeingQuality
}

def generateWordlists(wordListVectorsDict):
	TSBNLPpt_POSgetAllPossiblePosTags.constructPOSdictionary()	#required for TSBNLPpt_POSgetAllPossiblePosTags.getAllPossiblePosTags(word)
	generatePOSwordLists()
	generatePOSwordListVectors(vocabularySize, wordListVectorsDict)
	
def generatePOSwordLists():	
	numberOfPOSwordLists = len(nltkPOStagsDict)
	posWordLists = [[] for x in range(numberOfPOSwordLists)]
	for word in NLTKwords.words():
		wordPosValues = TSBNLPpt_POSgetAllPossiblePosTags.getAllPossiblePosTags(word)
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
	#print("writeWordList: POSname = ", POSname)
	wordlistFileName = generateWordlistFileName("wordlist" + POSname)
	with open(wordlistFileName, 'w') as f:
		f.write("\n".join(wordList))
		
def readWordList(POSname):
	#print("readWordList: POSname = ", POSname)
	wordList = []
	wordlistFileName = generateWordlistFileName("wordlist" + POSname)
	with open(wordlistFileName, 'r') as f:
		wordList = f.read().splitlines()	#or eval(f.read())
		#wordList = eval(f.read())
	return wordList

def generateWordlistFileName(fileName):
	print("generateWordlistFileName: fileName = ", fileName)
	wordlistFileName = LRPpathName + "/" + fileName + ".txt"
	return wordlistFileName
	
def createDictionaryItemsFromList(lst, startIndex):
	list1 = lst
	list2 = range(startIndex, len(lst)+startIndex)
	dictionaryItems = zip(list1, list2)
	return dictionaryItems

POSwordListVectorValueFalse = "0"
POSwordListVectorValueTrue = "1"
def generatePOSwordListVectors(vocabSize, wordListVectorsDict):
	#posVectorListList = []
	POSwordListAll = readWordList(wordListAllname)
	POSwordDictAllItems = createDictionaryItemsFromList(POSwordListAll, 0)
	POSwordDictAll = dict(POSwordDictAllItems)
	for POSindex, POSitem in enumerate(wordListVectorsDict.items()):
		posList = [POSwordListVectorValueFalse]*vocabSize
		POSname = POSitem[0]
		POSwordList = readWordList(POSname)
		for POSword in POSwordList:
			POSwordIndex = POSwordDictAll[POSword]
			posList[POSwordIndex] = POSwordListVectorValueTrue
		#posVectorListList.append(posVector)
		posVectorFileName = "Vector" + POSname
		writeWordList(posVectorFileName, posList)

def loadPOSwordListVectors(wordListVectorsDict):
	posVectorList = []
	for POSindex, POSitem in enumerate(wordListVectorsDict.items()):
		POSname = POSitem[0]
		posVectorFileName = "Vector" + POSname
		posList = readWordList(posVectorFileName)
		posList = [int(i) for i in posList]	#convert list to ints
		#print("len(posList) = ", len(posList))
		posVector = torch.tensor(posList, requires_grad=False)	#.bool()
		posVectorList.append(posVector)
	return posVectorList

if(__name__ == '__main__'):
	generateWordlists(wordListVectorsDictAll)
