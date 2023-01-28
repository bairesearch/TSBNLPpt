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
import SBNLPpt_GIAmodel
import torch.nn.functional as F

import SBNLPpt_GIAdefinePOSwordLists
if(useVectorisedSemanticRelationIdentification):
	import SBNLPpt_GIAsemanticRelationVectorised
else:
	import SBNLPpt_GIAsemanticRelationStandard


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if(debugReduceEmbeddingLayerSize):
	embeddingLayerSize = 10
else:
	embeddingLayerSize = 768	#word vector embedding size (cany vary based on GIA word vector space)



modelPathName = modelPathName + '/modelGIA.pt'

def preparePOSdictionary():
	if(useVectorisedSemanticRelationIdentification):
		global posVectorList
		posVectorList = SBNLPpt_GIAdefinePOSwordLists.loadPOSwordListVectors()
		numberOfVectorSpaces = len(SBNLPpt_GIAdefinePOSwordLists.vectorSpaceList)
	else:
		SBNLPpt_GIAsemanticRelationStandard.preparePOSdictionary()
	
	
def createModel(vocabSize):
	print("creating new model")
	config = SBNLPpt_GIAmodel.GIAwordEmbeddingConfig(vocabSize, embeddingLayerSize)
	model = SBNLPpt_GIAmodel.GIAwordEmbeddingModel(config)
		
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


if(useVectorisedSemanticRelationIdentification):
	def calculateXYlabels(tokenizer, batch, vocabSize):
		return SBNLPpt_GIAsemanticRelationVectorised.calculateXYlabels(tokenizer, batch, vocabSize, posVectorList)
else:
	def calculateXYlabels(tokenizer, vectorSpace, vectorSpaceIndex, batch, vocabSize):
		return SBNLPpt_GIAsemanticRelationStandard.calculateXYlabels(tokenizer, vectorSpace, vectorSpaceIndex, batch, vocabSize)
	
