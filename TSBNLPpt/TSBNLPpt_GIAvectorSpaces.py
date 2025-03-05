"""TSBNLPpt_GIAvectorSpaces.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see TSBNLPpt_main.py

# Usage:
see TSBNLPpt_main.py

# Description:
TSBNLPpt GIA vector spaces

# Usage (generateWordlists for LRPdata):
source activate transformersenv
python TSBNLPpt_GIAvectorSpaces.py

"""

import torch as pt
import torch

from TSBNLPpt_globalDefs import *
import TSBNLPpt_POSgetAllPossiblePosTags
import torch.nn.functional as F

import nltk
from nltk.corpus import words as NLTKwords

import TSBNLPpt_POSgetAllPossiblePosTags
import TSBNLPpt_POSwordLists
from TSBNLPpt_POSwordLists import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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
keypointPosAdjective: nltkPOStagsAdjective,
keypointPosAdverb: nltkPOStagsAdverb,
keypointPosPreposition: nltkPOStagsPreposition,
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
#if(GIAuseVectorisedSemanticRelationIdentification):
#	if(keypointMaxDetectionDistance%2 == 0):
#		print("TSBNLPpt_GIAsemanticRelationVectorised error: keypointMaxDetectionDistance must be an odd number (Conv1d kernel size)")
#		exit()

if(GIAgenerateUniqueWordVectorsForRelationTypes):
	keypointNone = -1
			
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


#GIAuseVectorisedSemanticRelationIdentification: detectionTypeNearest is not supported for keyIntermediate (detectionTypeAdjacent only)

if(debugUseSmallNumberOfModels):
	vectorSpaceList = [
	vectorSpaceProperties("action", directionForward, detectionTypeNearest, intermediateTypeNone, keypointPosNoun, keypointPosVerb, keypointNone)
	]
else:
	if(GIAgenerateUniqueWordVectorsForRelationTypes):
		if(GIArelationTypesIntermediate):
			vectorSpaceList = [
			vectorSpaceProperties("definition", directionForward, detectionTypeNearest, intermediateTypeWord, keypointPosNoun, keypointPosNoun, keypointWordAuxiliaryBeingDefinition),
			vectorSpaceProperties("action", directionForward, detectionTypeNearest, intermediateTypePOS, keypointPosNoun, keypointPosNoun, keypointPosVerb),
			vectorSpaceProperties("property", directionForward, detectionTypeNearest, intermediateTypeWord, keypointPosNoun, keypointPosNoun, keypointWordAuxiliaryPossessive),
			vectorSpaceProperties("qualitySubstance1", directionForward, detectionTypeAdjacent, intermediateTypeWord, keypointPosNoun, keypointPosAdjective, keypointWordAuxiliaryBeingQuality),
			vectorSpaceProperties("qualitySubstance2", directionForward, detectionTypeAdjacent, intermediateTypeNone, keypointPosAdjective, keypointPosNoun, keypointNone),
			vectorSpaceProperties("qualityAction1", directionForward, detectionTypeAdjacent, intermediateTypeNone, keypointPosAdverb, keypointPosVerb, keypointNone),
			vectorSpaceProperties("qualityAction2", directionReverse, detectionTypeAdjacent, intermediateTypeNone, keypointPosVerb, keypointPosAdverb, keypointNone),
			vectorSpaceProperties("preposition", directionForward, detectionTypeNearest, intermediateTypePOS, keypointPosNoun, keypointPosNoun, keypointPosPreposition)
			]
		else:
			#does not support intermediateTypePOS;
			vectorSpaceList = [
			vectorSpaceProperties("definition", directionForward, detectionTypeNearest, intermediateTypeWord, keypointPosNoun, keypointPosNoun, keypointWordAuxiliaryBeingDefinition),
			vectorSpaceProperties("actionSubject", directionForward, detectionTypeNearest, intermediateTypeNone, keypointPosNoun, keypointPosVerb, keypointNone),
			vectorSpaceProperties("actionObject", directionReverse, detectionTypeNearest, intermediateTypeNone, keypointPosVerb, keypointPosNoun, keypointNone),
			vectorSpaceProperties("property", directionForward, detectionTypeNearest, intermediateTypeWord, keypointPosNoun, keypointPosNoun, keypointWordAuxiliaryPossessive),
			vectorSpaceProperties("qualitySubstance1", directionForward, detectionTypeAdjacent, intermediateTypeWord, keypointPosNoun, keypointPosAdjective, keypointWordAuxiliaryBeingQuality),	
			vectorSpaceProperties("qualitySubstance2", directionForward, detectionTypeAdjacent, intermediateTypeNone, keypointPosAdjective, keypointPosNoun, keypointNone),
			#vectorSpaceProperties("qualityAction1", directionForward, detectionTypeAdjacent, intermediateTypeNone, keypointPosAdverb, keypointPosVerb, keypointNone),	#temporarily disable to ensure vectorSpaceListLen is a factor of hiddenLayerSizeTransformer
			#vectorSpaceProperties("qualityAction2", directionReverse, detectionTypeAdjacent, intermediateTypeNone, keypointPosVerb, keypointPosAdverb, keypointNone),	#temporarily disable to ensure vectorSpaceListLen is a factor of hiddenLayerSizeTransformer
			vectorSpaceProperties("prepositionSubject", directionForward, detectionTypeNearest, intermediateTypeNone, keypointPosNoun, keypointPosPreposition, keypointNone),
			vectorSpaceProperties("prepositionObject", directionReverse, detectionTypeNearest, intermediateTypeNone, keypointPosPreposition, keypointPosNoun, keypointNone)
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
		#vectorSpaceProperties("qualityAction1", directionForward, detectionTypeAdjacent, intermediateTypeNone, keypointPosAdverb, keypointPosVerb, keypointNone),
		#vectorSpaceProperties("qualityAction2", directionForward, detectionTypeAdjacent, intermediateTypeNone, keypointPosVerb, keypointPosAdverb, keypointNone),
		vectorSpaceProperties("preposition", directionForward, detectionTypeNearest, intermediateTypePOS, keypointPosNoun, keypointPosNoun, keypointPosPreposition)
		]
		if(useIndependentReverseRelationsModels):
			vectorSpaceListR = [
				vectorSpaceProperties("definitionR", directionReverse, detectionTypeNearest, intermediateTypeWord, keypointPosNoun, keypointPosNoun, keypointWordAuxiliaryBeingDefinition),
				vectorSpaceProperties("actionR", directionForward, detectionTypeNearest, intermediateTypePOS, keypointPosNoun, keypointPosNoun, keypointPosVerb),
				vectorSpaceProperties("actionSubjectR", directionReverse, detectionTypeNearest, intermediateTypeNone, keypointPosNoun, keypointPosVerb, keypointNone),
				vectorSpaceProperties("actionObjectR", directionReverse, detectionTypeNearest, intermediateTypeNone, keypointPosVerb, keypointPosNoun, keypointNone),
				vectorSpaceProperties("propertyR", directionReverse, detectionTypeNearest, intermediateTypeWord, keypointPosNoun, keypointPosNoun, keypointWordAuxiliaryPossessive),
				vectorSpaceProperties("qualitySubstance1R", directionReverse, detectionTypeAdjacent, intermediateTypeWord, keypointPosNoun, keypointPosAdjective, keypointWordAuxiliaryBeingQuality),	#temporarily disable to ensure vectorSpaceListLen is a factor of hiddenLayerSizeTransformer
				vectorSpaceProperties("qualitySubstance2R", directionReverse, detectionTypeAdjacent, intermediateTypeNone, keypointPosAdjective, keypointPosNoun, keypointNone),
				#vectorSpaceProperties("qualityAction1R", directionReverse, detectionTypeAdjacent, intermediateTypeNone, keypointPosAdverb, keypointPosVerb, keypointNone),
				#vectorSpaceProperties("qualityAction2R", directionReverse, detectionTypeAdjacent, intermediateTypeNone, keypointPosVerb, keypointPosAdverb, keypointNone),
				vectorSpaceProperties("prepositionR", directionReverse, detectionTypeNearest, intermediateTypePOS, keypointPosNoun, keypointPosNoun, keypointPosPreposition)
			]
			vectorSpaceList = vectorSpaceList + vectorSpaceListR
	assert(len(vectorSpaceList) == vectorSpaceListLen)
		


if(__name__ == '__main__'):
	TSBNLPpt_POSwordLists.generateWordlists(keypointsDict)


def loadPOSwordListVectors():
	return TSBNLPpt_POSwordLists.loadPOSwordListVectors(keypointsDict)


def addModelSampleToList(vectorSpace, modelSamplesX, modelSamplesY, modelSamplesI, wordIndexKeypoint0, wordIndexKeypoint1, wordIndexKeypoint2):
	if(GIAgenerateUniqueWordVectorsForRelationTypes):
		if(GIArelationTypesIntermediate):
			if(vectorSpace.direction == directionForward):
				xTokenIndex = wordIndexKeypoint0
				yTokenIndex = wordIndexKeypoint2
				iTokenIndex = wordIndexKeypoint1
			elif(vectorSpace.direction == directionReverse):
				xTokenIndex = wordIndexKeypoint2
				yTokenIndex = wordIndexKeypoint0
				iTokenIndex = wordIndexKeypoint1
		else:
			if(vectorSpace.direction == directionForward):
				xTokenIndex = wordIndexKeypoint0
				yTokenIndex = wordIndexKeypoint2
			else:
				xTokenIndex = wordIndexKeypoint2
				yTokenIndex = wordIndexKeypoint0
	else:
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
		if(GIArelationTypesIntermediate):
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
	
