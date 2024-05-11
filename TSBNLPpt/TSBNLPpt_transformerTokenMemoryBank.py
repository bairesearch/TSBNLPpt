"""TSBNLPpt_transformerTokenMemoryBank.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see TSBNLPpt_main.py

# Usage:
see TSBNLPpt_main.py

# Description:
TSBNLPpt transformer token memory bank

"""


import torch
from torch import nn
import nncustom

from TSBNLPpt_globalDefs import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import torch.nn.functional as F
import torch.nn.init as init


if(tokenMemoryBankStorageSelectionAlgorithmAuto):
	from torchmetrics.classification import BinaryAccuracy
	#https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html

	class tokenMemoryBankStorageSelectionConfig():
		def __init__(self, numberOfHiddenLayers, inputLayerSize, hiddenLayerSize, outputLayerSize):
			self.numberOfHiddenLayers = numberOfHiddenLayers
			self.inputLayerSize = inputLayerSize
			self.hiddenLayerSize = hiddenLayerSize
			self.outputLayerSize = outputLayerSize

	class tokenMemoryBankStorageSelectionModel(nn.Module):
		def __init__(self, config, layerIndex):
			super().__init__()
			self.layerIndex = layerIndex	#not currently used
			self.hidden = nncustom.Linear(config.inputLayerSize, config.hiddenLayerSize)
			self.hiddenActivationFunction = nn.ReLU()
			self.output = nncustom.Linear(config.hiddenLayerSize, config.outputLayerSize)
			if(tokenMemoryBankStorageSelectionInitiationBias):
				init.normal_(self.output.weight, mean=tokenMemoryBankStorageSelectionInitiationBiasOutputLayerMean)	#std=0.1
			if(config.outputLayerSize == 1):
				self.outputActivationFunction = nn.Sigmoid()
				self.lossFunction = nn.BCELoss()	#nn.BCEWithLogitsLoss() includes sigmoid
			elif(config.outputLayerSize == 2):
				self.outputActivationFunction = nn.LogSoftmax(dim=1)
				self.lossFunction = nn.NLLLoss()	 #CrossEntropyLoss() includes softmax
			self.accuracyMetric = BinaryAccuracy(threshold=0.5)	#threshold=tokenMemoryBankStorageSelectionBinaryThreshold
		def forward(self, x):
			x = self.hidden(x)
			x = self.hiddenActivationFunction(x)
			y = self.output(x)
			y = self.outputActivationFunction(y)
			return y

	tokenMemoryBankStorageSelectionModelList = [None]*numberOfHiddenLayersTokenMemoryBankParameters	#similar to modelStoreList

	def createModelTokenMemoryBankStorageSelection(layerIndex, config):
		print("layerIndex = ", layerIndex)
		model = tokenMemoryBankStorageSelectionModel(config, layerIndex)
		tokenMemoryBankStorageSelectionModelList[layerIndex] = model
		return model

	sequenceRegisterMemoryBankHiddenStatesTrainRememberLayerList = [None]*numberOfHiddenLayersTokenMemoryBankParameters
	sequenceRegisterMemoryBankHiddenStatesTrainForgetLayerList = [None]*numberOfHiddenLayersTokenMemoryBankParameters

	def getTokenMemoryBankStorageSelectionModelBatch(layerIndex):
		#print("layerIndex = ", layerIndex)
		sequenceRegisterMemoryBankHiddenStatesTrainRememberLayer = sequenceRegisterMemoryBankHiddenStatesTrainRememberLayerList[layerIndex]
		sequenceRegisterMemoryBankHiddenStatesTrainForgetLayer = sequenceRegisterMemoryBankHiddenStatesTrainForgetLayerList[layerIndex]
		sequenceRegisterMemoryBankOutputsTrainRememberLayer = torch.ones(sequenceRegisterMemoryBankHiddenStatesTrainRememberLayer.shape[0])
		sequenceRegisterMemoryBankOutputsTrainForgetLayer = torch.zeros(sequenceRegisterMemoryBankHiddenStatesTrainForgetLayer.shape[0])
		xLabels = torch.cat([sequenceRegisterMemoryBankHiddenStatesTrainRememberLayer, sequenceRegisterMemoryBankHiddenStatesTrainForgetLayer])
		yLabels = torch.cat([sequenceRegisterMemoryBankOutputsTrainRememberLayer, sequenceRegisterMemoryBankOutputsTrainForgetLayer])
		yLabels = torch.unsqueeze(yLabels, dim=1)
		#print("xLabels.shape = ", xLabels.shape)
		#print("yLabels.shape = ", yLabels.shape)
		xLabels = xLabels.to(device)
		yLabels = yLabels.to(device)
		labels = {}
		labels['xLabels'] = xLabels
		labels['yLabels'] = yLabels
		return labels

class TokenMemoryBankClass():
	def __init__(self, config):
		self.hidden_size = config.hidden_size
		self.batchIndex = 0
		self.sequenceRegisterMemoryBankHiddenStates = None
		self.sequenceRegisterMemoryBankAccessTimes = None
		self.sequenceRegisterMemoryBankTokenTimes = None
		self.clearSequenceRegisterMemoryBank()

	def updateSequenceRegisterMemoryBank(self, layerIndex, hiddenStates):
		(attentionProbsMaxIndex, sequenceRegisterMemoryBankAccessTimes, sequenceRegisterMemoryBankTokenTimes) = (self.attentionProbsMaxIndex, self.sequenceRegisterMemoryBankAccessTimes, self.sequenceRegisterMemoryBankTokenTimes)

		#interpretation: access time 0 = recently activated

		if(tokenMemoryBankStorageSelectionAlgorithmAuto):
			sequenceRegisterContextualWindowHiddenStates = self.getSequenceRegisterContextualWindow(hiddenStates)	#shape: batchSize*contextualWindowSize*hiddenLayerSize
			tokenMemoryBankStorageSelectionModel = tokenMemoryBankStorageSelectionModelList[layerIndex]
			tokenMemoryBankStorageSelectionModel.eval()
			tokenMemoryBankStorageSelectionProbability = tokenMemoryBankStorageSelectionModel(sequenceRegisterContextualWindowHiddenStates).detach()	#shape: batchSize*contextualWindowSize*2
			sequenceRegisterContextualWindowStore = torch.gt(tokenMemoryBankStorageSelectionProbability, tokenMemoryBankStorageSelectionBinaryThreshold)
			if(tokenMemoryBankStorageSelectionModelOutputLayerSize == 1):
				sequenceRegisterContextualWindowStore = torch.squeeze(sequenceRegisterContextualWindowStore, dim=2)
			elif(tokenMemoryBankStorageSelectionModelOutputLayerSize == 2):
				print("error: tokenMemoryBankStorageSelectionModelOutputLayerSize==2 not yet coded")

			if(debugTokenMemoryBankStorageSelectionAlgorithmAuto):	
				print("tokenMemoryBankStorageSelectionProbability = ", tokenMemoryBankStorageSelectionProbability)
				print("sequenceRegisterContextualWindowStore = ", sequenceRegisterContextualWindowStore)

			sequenceRegisterMemoryBankAccessTimes = torch.add(sequenceRegisterMemoryBankAccessTimes, 1)	#update access time for next model propagation	#increment sequenceRegisterMemoryBankAccessTimes
			sequenceRegisterContextualWindowAccessTimes = (torch.logical_not(sequenceRegisterContextualWindowStore)).float()
			sequenceRegisterContextualWindowAccessTimes = torch.multiply(sequenceRegisterContextualWindowAccessTimes, sequenceRegisterMaxActivationTime)
			#sequenceRegisterAccessTimes = torch.cat((sequenceRegisterContextualWindowAccessTimes, sequenceRegisterMemoryBankAccessTimes), dim=1)	#not used (sequenceRegisterMemoryBankAccessTimes is recalculated independently later)

			sequenceRegisterAccessedNew = self.calculateSequenceRegisterAccessedNew(attentionProbsMaxIndex)
			sequenceRegisterMemoryBankAccessedNew = self.getSequenceRegisterMemoryBank(sequenceRegisterAccessedNew)
			sequenceRegisterMemoryBankAccessTimes = self.renewAccessTimeOfRecentlyAccessedTokens(sequenceRegisterMemoryBankAccessedNew, sequenceRegisterMemoryBankAccessTimes)
			sequenceRegisterAccessTimes = self.formSequenceRegisterValues(sequenceRegisterContextualWindowAccessTimes, sequenceRegisterMemoryBankAccessTimes)
		else:
			if(onlyAddAttendedContextualWindowTokensToMemoryBank):
				sequenceRegisterMemoryBankAccessTimes = torch.add(sequenceRegisterMemoryBankAccessTimes, 1)	#update access time for next model propagation	#increment sequenceRegisterMemoryBankAccessTimes
				sequenceRegisterContextualWindowAccessTimes = torch.full([batchSize, sequenceRegisterContextualWindowLength], sequenceRegisterMaxActivationTime).to(device)	#prepare default sequenceRegisterContextualWindowAccessTimes (unrenewed)
				sequenceRegisterAccessTimes = self.formSequenceRegisterValues(sequenceRegisterContextualWindowAccessTimes, sequenceRegisterMemoryBankAccessTimes)
			else:
				sequenceRegisterContextualWindowAccessTimes = torch.zeros(batchSize, sequenceRegisterContextualWindowLength).to(device)
				sequenceRegisterAccessTimes = self.formSequenceRegisterValues(sequenceRegisterContextualWindowAccessTimes, sequenceRegisterMemoryBankAccessTimes)
				sequenceRegisterAccessTimes = torch.add(sequenceRegisterAccessTimes, 1)	#update access time for next model propagation	#increment sequenceRegisterAccessTimes;

			sequenceRegisterAccessedNew = self.calculateSequenceRegisterAccessedNew(attentionProbsMaxIndex)
			sequenceRegisterAccessTimes = self.renewAccessTimeOfRecentlyAccessedTokens(sequenceRegisterAccessedNew, sequenceRegisterAccessTimes)

		sequenceRegisterMemoryBankTokenTimes = self.updateSequenceRegisterMemoryBankTokenTimes(sequenceRegisterMemoryBankTokenTimes)
		sequenceRegisterTokenTimes = self.getSequenceRegisterTokenTimes(sequenceRegisterMemoryBankTokenTimes, sequenceRegisterMemoryBankAccessTimes)

		if(debugPrintLowHiddenSize):
			print("sequenceRegisterAccessTimes = ", sequenceRegisterAccessTimes)

		#calculate if tokens are to be retained in the memory bank;
		#print("sequenceRegisterAccessTimes = ", sequenceRegisterAccessTimes)
		sequenceRegisterRetain = torch.lt(sequenceRegisterAccessTimes, sequenceRegisterMaxActivationTime)
		sequenceRegisterRetainSize = torch.sum(sequenceRegisterRetain.int(), dim=1)	

		if(tokenMemoryBankStorageSelectionAlgorithmAuto):
			#learn TrainRemember tokens that are accessed in the memory bank
			#learn TrainForget tokens that are never accessed in the memory bank
			sequenceRegisterTrainRemember = self.getSequenceRegisterMemoryBank(sequenceRegisterAccessedNew)
			sequenceRegisterTrainRememberSize = torch.sum(sequenceRegisterTrainRemember.int(), dim=1)	

			sequenceRegisterTrainForget = torch.logical_not(sequenceRegisterRetain)
			sequenceRegisterTrainForget = self.getSequenceRegisterMemoryBank(sequenceRegisterTrainForget)
			#print("sequenceRegisterTrainForget = ", sequenceRegisterTrainForget)
			sequenceRegisterMemoryBankTheoreticalNoAccessTimes = self.calculateSequenceRegisterMemoryBankTheoreticalNoAccessTimes(sequenceRegisterMemoryBankTokenTimes)
			sequenceRegisterMemoryBankNoAccess = torch.eq(sequenceRegisterMemoryBankTheoreticalNoAccessTimes, sequenceRegisterMaxActivationTime)	#CHECKTHIS
			sequenceRegisterTrainForget = torch.logical_and(sequenceRegisterTrainForget, sequenceRegisterMemoryBankNoAccess)
			sequenceRegisterTrainForgetSize = torch.sum(sequenceRegisterTrainForget.int(), dim=1)

			if(tokenMemoryBankStorageSelectionNormaliseForgetRememberSize):
				sequenceRegisterTrainRememberSizeReduced = torch.sum(sequenceRegisterTrainRememberSize)
				sequenceRegisterTrainForgetSizeReduced = torch.sum(sequenceRegisterTrainForgetSize)
				if(sequenceRegisterTrainRememberSizeReduced > sequenceRegisterTrainForgetSizeReduced):
					normalisationFactor = (1-(sequenceRegisterTrainForgetSizeReduced/sequenceRegisterTrainRememberSizeReduced))/2 + 0.5		#[biased forget] 0->0.5 [no bias] 0.5->1.0 [biased remember]
				elif(sequenceRegisterTrainRememberSizeReduced < sequenceRegisterTrainForgetSizeReduced):
					normalisationFactor = (1-(sequenceRegisterTrainRememberSizeReduced/sequenceRegisterTrainForgetSizeReduced))/2 + 0.5		#[biased forget] 0->0.5 [no bias] 0.5->1.0 [biased remember]
				else:
					normalisationFactor = 0.5
				if(debugTokenMemoryBankStorageSelectionAlgorithmAuto):
					#print("1 sequenceRegisterTrainRememberSizeReduced = ", sequenceRegisterTrainRememberSizeReduced)
					#print("1 sequenceRegisterTrainForgetSizeReduced = ", sequenceRegisterTrainForgetSizeReduced)
					pass
				if(normalisationFactor > tokenMemoryBankStorageSelectionNormaliseForgetRememberSizeBias):
					#too many remember tokens
					normalisationMask = torch.rand(sequenceRegisterTrainRemember.shape, device=device) > (normalisationFactor-tokenMemoryBankStorageSelectionNormaliseForgetRememberSizeBias)*2
					sequenceRegisterTrainRemember = torch.logical_and(sequenceRegisterTrainRemember, normalisationMask)
				elif(normalisationFactor < tokenMemoryBankStorageSelectionNormaliseForgetRememberSizeBias):
					#too many forget tokens
					normalisationMask = torch.rand(sequenceRegisterTrainForget.shape, device=device) > (tokenMemoryBankStorageSelectionNormaliseForgetRememberSizeBias-normalisationFactor)*2
					sequenceRegisterTrainForget = torch.logical_and(sequenceRegisterTrainForget, normalisationMask)
				'''
				#orig without support for tokenMemoryBankStorageSelectionNormaliseForgetRememberSizeBias:
				if(sequenceRegisterTrainRememberSizeReduced > sequenceRegisterTrainForgetSizeReduced):
					normalisationFactor = sequenceRegisterTrainForgetSizeReduced/sequenceRegisterTrainRememberSizeReduced
					normalisationMask = torch.rand(sequenceRegisterTrainRemember.shape, device=device) < normalisationFactor
					sequenceRegisterTrainRemember = torch.logical_and(sequenceRegisterTrainRemember, normalisationMask)
				elif(sequenceRegisterTrainForgetSizeReduced > sequenceRegisterTrainRememberSizeReduced):
					normalisationFactor = sequenceRegisterTrainRememberSizeReduced/sequenceRegisterTrainForgetSizeReduced
					normalisationMask = torch.rand(sequenceRegisterTrainForget.shape, device=device) < normalisationFactor
					sequenceRegisterTrainForget = torch.logical_and(sequenceRegisterTrainForget, normalisationMask)
				'''					
				sequenceRegisterTrainRememberSize = torch.sum(sequenceRegisterTrainRemember.int(), dim=1)	
				sequenceRegisterTrainForgetSize = torch.sum(sequenceRegisterTrainForget.int(), dim=1)
				sequenceRegisterTrainRememberSizeReduced = torch.sum(sequenceRegisterTrainRememberSize)
				sequenceRegisterTrainForgetSizeReduced = torch.sum(sequenceRegisterTrainForgetSize)
				if(debugTokenMemoryBankStorageSelectionAlgorithmAuto):
					print("2 sequenceRegisterTrainRememberSizeReduced = ", sequenceRegisterTrainRememberSizeReduced)
					print("2 sequenceRegisterTrainForgetSizeReduced = ", sequenceRegisterTrainForgetSizeReduced)

		#update memory bank;
		sequenceRegisterMemoryBankHiddenStatesList = []
		sequenceRegisterMemoryBankAccessTimesList = []
		sequenceRegisterMemoryBankTokenTimesList = []
		if(tokenMemoryBankStorageSelectionAlgorithmAuto):
			sequenceRegisterMemoryBankHiddenStatesTrainRememberList = []
			sequenceRegisterMemoryBankHiddenStatesTrainForgetList = []
		for sampleIndex in range(batchSize):
			#must execute mask select for each sample in batch because they will produce different length tensors
			sequenceRegisterRetainSample = sequenceRegisterRetain[sampleIndex]
			sequenceRegisterRetainSizeSample = sequenceRegisterRetainSize[sampleIndex]
			if(tokenMemoryBankStorageSelectionAlgorithmAuto):
				sequenceRegisterTrainRememberSample = sequenceRegisterTrainRemember[sampleIndex]
				sequenceRegisterTrainRememberSizeSample = sequenceRegisterTrainRememberSize[sampleIndex]
				sequenceRegisterTrainForgetSample = sequenceRegisterTrainForget[sampleIndex]
				sequenceRegisterTrainForgetSizeSample = sequenceRegisterTrainForgetSize[sampleIndex]

			hiddenStatesSample = hiddenStates[sampleIndex].detach()
			sequenceRegisterAccessTimesSample = sequenceRegisterAccessTimes[sampleIndex]
			sequenceRegisterMemoryBankTokenTimesSample = sequenceRegisterTokenTimes[sampleIndex]

			sequenceRegisterMemoryBankHiddenStatesSample = hiddenStatesSample[sequenceRegisterRetainSample]
			sequenceRegisterMemoryBankAccessTimesSample = sequenceRegisterAccessTimesSample[sequenceRegisterRetainSample]
			sequenceRegisterMemoryBankTokenTimesSample = sequenceRegisterMemoryBankTokenTimesSample[sequenceRegisterRetainSample]
			if(tokenMemoryBankStorageSelectionAlgorithmAuto):
				hiddenStatesSampleMemoryBank = self.getSequenceRegisterMemoryBankSample(hiddenStatesSample)
				sequenceRegisterMemoryBankHiddenStatesSampleTrainRemember = hiddenStatesSampleMemoryBank[sequenceRegisterTrainRememberSample]
				sequenceRegisterMemoryBankHiddenStatesSampleTrainForget = hiddenStatesSampleMemoryBank[sequenceRegisterTrainForgetSample]
				#print("sequenceRegisterMemoryBankHiddenStatesSampleTrainRemember.shape = ", sequenceRegisterMemoryBankHiddenStatesSampleTrainRemember.shape)
				#print("sequenceRegisterMemoryBankHiddenStatesSampleTrainForget.shape = ", sequenceRegisterMemoryBankHiddenStatesSampleTrainForget.shape)

			#if(sequenceRegisterRetainSizeSample > sequenceRegisterLength):
			if(debugPrintSequenceRegisterRetainSize):
				print("sequenceRegisterRetainSizeSample = ", sequenceRegisterRetainSizeSample.cpu().numpy())
			if(sequenceRegisterRetainSizeSample < sequenceRegisterLength):	#or sequenceRegisterRetainSizeSample != sequenceRegisterLength
				#pad sequence register with dummy tokens
				paddingSize = sequenceRegisterLength-sequenceRegisterRetainSizeSample
				hiddenSize = sequenceRegisterMemoryBankHiddenStatesSample.shape[1]

				sequenceRegisterMemoryBankHiddenStatesSamplePad = torch.zeros([paddingSize, hiddenSize]).to(device)
				sequenceRegisterMemoryBankAccessTimesSamplePad = torch.full([paddingSize], sequenceRegisterMemoryBankPaddingAccessTime).to(device)
				sequenceRegisterMemoryBankTokenTimesSamplePad = torch.full([paddingSize], sequenceRegisterMemoryBankPaddingTokenTime).to(device)

				sequenceRegisterMemoryBankHiddenStatesSample = torch.cat((sequenceRegisterMemoryBankHiddenStatesSample, sequenceRegisterMemoryBankHiddenStatesSamplePad), dim=0)
				sequenceRegisterMemoryBankAccessTimesSample = torch.cat((sequenceRegisterMemoryBankAccessTimesSample, sequenceRegisterMemoryBankAccessTimesSamplePad), dim=0)
				sequenceRegisterMemoryBankTokenTimesSample = torch.cat((sequenceRegisterMemoryBankTokenTimesSample, sequenceRegisterMemoryBankTokenTimesSamplePad), dim=0)

			if(sequenceRegisterVerifyMemoryBankSize):
				#sort memory bank by time to ensure that oldest tokens can easily be deleted if run out of space
				sequenceRegisterMemoryBankAccessTimesSample, indices = torch.sort(sequenceRegisterMemoryBankAccessTimesSample, dim=0)
				sequenceRegisterMemoryBankHiddenStatesSample = sequenceRegisterMemoryBankHiddenStatesSample[indices]
				sequenceRegisterMemoryBankTokenTimesSample = sequenceRegisterMemoryBankTokenTimesSample[indices]

			if(debugPrintLowHiddenSize):
				#print("sequenceRegisterMemoryBankHiddenStatesSample = ", sequenceRegisterMemoryBankHiddenStatesSample)			
				#print("sequenceRegisterMemoryBankAccessTimesSample = ", sequenceRegisterMemoryBankAccessTimesSample)
				print("sequenceRegisterMemoryBankTokenTimesSample = ", sequenceRegisterMemoryBankTokenTimesSample)

			sequenceRegisterMemoryBankHiddenStatesList.append(sequenceRegisterMemoryBankHiddenStatesSample)
			sequenceRegisterMemoryBankAccessTimesList.append(sequenceRegisterMemoryBankAccessTimesSample)
			sequenceRegisterMemoryBankTokenTimesList.append(sequenceRegisterMemoryBankTokenTimesSample)
			if(tokenMemoryBankStorageSelectionAlgorithmAuto):
				sequenceRegisterMemoryBankHiddenStatesTrainRememberList.append(sequenceRegisterMemoryBankHiddenStatesSampleTrainRemember)
				sequenceRegisterMemoryBankHiddenStatesTrainForgetList.append(sequenceRegisterMemoryBankHiddenStatesSampleTrainForget)

		sequenceRegisterMemoryBankHiddenStates = torch.stack(sequenceRegisterMemoryBankHiddenStatesList, dim=0)
		sequenceRegisterMemoryBankAccessTimes = torch.stack(sequenceRegisterMemoryBankAccessTimesList, dim=0)
		sequenceRegisterMemoryBankTokenTimes = torch.stack(sequenceRegisterMemoryBankTokenTimesList, dim=0)

		if(tokenMemoryBankStorageSelectionAlgorithmAuto):
			#print("layerIndex = ", layerIndex)
			sequenceRegisterMemoryBankHiddenStatesTrainRememberLayer = torch.cat(sequenceRegisterMemoryBankHiddenStatesTrainRememberList, dim=0)	#cat batch list
			sequenceRegisterMemoryBankHiddenStatesTrainForgetLayer = torch.cat(sequenceRegisterMemoryBankHiddenStatesTrainForgetList, dim=0)	#cat batch list
			sequenceRegisterMemoryBankHiddenStatesTrainRememberLayerList[layerIndex] = sequenceRegisterMemoryBankHiddenStatesTrainRememberLayer
			sequenceRegisterMemoryBankHiddenStatesTrainForgetLayerList[layerIndex] = sequenceRegisterMemoryBankHiddenStatesTrainForgetLayer

		#delete oldest tokens;
		sequenceRegisterMemoryBankHiddenStates = sequenceRegisterMemoryBankHiddenStates[:, 0:sequenceRegisterMemoryBankLength]
		sequenceRegisterMemoryBankAccessTimes = sequenceRegisterMemoryBankAccessTimes[:, 0:sequenceRegisterMemoryBankLength]
		sequenceRegisterMemoryBankTokenTimes = sequenceRegisterMemoryBankTokenTimes[:, 0:sequenceRegisterMemoryBankLength]

		#restore contextual window hidden states;
		hiddenStates = self.restoreContextualWindowHiddenStates(hiddenStates)

		(self.sequenceRegisterMemoryBankHiddenStates, self.sequenceRegisterMemoryBankAccessTimes, self.sequenceRegisterMemoryBankTokenTimes) = (sequenceRegisterMemoryBankHiddenStates, sequenceRegisterMemoryBankAccessTimes, sequenceRegisterMemoryBankTokenTimes)

		return hiddenStates

	def calculateSequenceRegisterAccessedNew(self, attentionProbsMaxIndex):
		sequenceRegisterAccessedNew = F.one_hot(attentionProbsMaxIndex, num_classes=sequenceRegisterLength)
		sequenceRegisterAccessedNew = torch.sum(sequenceRegisterAccessedNew, dim=1)
		sequenceRegisterAccessedNew = sequenceRegisterAccessedNew.bool()
		return sequenceRegisterAccessedNew

	def renewAccessTimeOfRecentlyAccessedTokens(self, sequenceRegisterAccessedNew, sequenceRegisterAccessTimes):
		sequenceRegisterAccessedNewNot = torch.logical_not(sequenceRegisterAccessedNew)
		sequenceRegisterAccessTimes = torch.multiply(sequenceRegisterAccessTimes, sequenceRegisterAccessedNewNot.float())
		sequenceRegisterAccessedNewTime = torch.multiply(sequenceRegisterAccessedNew.float(), sequenceRegisterRenewTime)
		sequenceRegisterAccessTimes = torch.add(sequenceRegisterAccessTimes, sequenceRegisterAccessedNewTime)
		return sequenceRegisterAccessTimes

	def loadSequenceRegisterHiddenStates(self, sequenceRegisterContextualWindowHiddenStates):
		if(self.batchIndex % orderedDatasetDocNumberSegments == 0):
			self.clearSequenceRegisterMemoryBank()
		self.batchIndex += 1
		hiddenStates = self.formSequenceRegisterValues(sequenceRegisterContextualWindowHiddenStates, self.sequenceRegisterMemoryBankHiddenStates)
		return hiddenStates

	def formSequenceRegisterValues(self, sequenceRegisterContextualWindowValues, sequenceRegisterMemoryBankValues):
		sequenceRegisterValues = torch.cat((sequenceRegisterContextualWindowValues, sequenceRegisterMemoryBankValues), dim=1)
		return sequenceRegisterValues

	def restoreContextualWindowHiddenStates(self, sequenceRegisterHiddenStates):
		hiddenStates = self.getSequenceRegisterContextualWindow(sequenceRegisterHiddenStates)
		return hiddenStates

	def getSequenceRegisterMemoryBank(self, sequenceRegisterHiddenStates):
		sequenceRegisterMemoryBank = sequenceRegisterHiddenStates[:, sequenceRegisterContextualWindowLength:]
		return sequenceRegisterMemoryBank

	def getSequenceRegisterContextualWindow(self, sequenceRegisterHiddenStates):
		sequenceRegisterContextualWindow = sequenceRegisterHiddenStates[:, 0:sequenceRegisterContextualWindowLength]
		return sequenceRegisterContextualWindow

	def getSequenceRegisterMemoryBankSample(self, sequenceRegisterHiddenStatesSample):
		sequenceRegisterMemoryBankSample = sequenceRegisterHiddenStatesSample[sequenceRegisterContextualWindowLength:]
		return sequenceRegisterMemoryBankSample


	def clearSequenceRegisterMemoryBank(self):
		self.sequenceRegisterMemoryBankHiddenStates = torch.zeros(batchSize, sequenceRegisterMemoryBankLength, self.hidden_size).to(device)	#CHECKTHIS; same shape as hidden_states
		self.sequenceRegisterMemoryBankAccessTimes = torch.full([batchSize, sequenceRegisterMemoryBankLength], sequenceRegisterMemoryBankPaddingAccessTime).to(device)	
		self.sequenceRegisterMemoryBankTokenTimes = torch.full([batchSize, sequenceRegisterMemoryBankLength], sequenceRegisterMemoryBankPaddingTokenTime).to(device)	

	def getSequenceRegisterTokenTimes(self, sequenceRegisterMemoryBankTokenTimes, sequenceRegisterMemoryBankAccessTimes):
		sequenceRegisterContextualWindowTokenTimes = self.calculateSequenceRegisterContextualWindowTokenTimes()
		if(calculateMemoryBankTokenTimesFromAccessTimes):
			sequenceRegisterMemoryBankTokenTimes = self.calculateSequenceRegisterMemoryBankTokenTimes(sequenceRegisterMemoryBankAccessTimes)
		sequenceRegisterTokenTimes = torch.cat((sequenceRegisterContextualWindowTokenTimes, sequenceRegisterMemoryBankTokenTimes), dim=1)
		return sequenceRegisterTokenTimes

	def calculateSequenceRegisterContextualWindowTokenTimes(self):
		sequenceRegisterContextualWindowTokenTimes = torch.arange(start=sequenceMaxNumTokens, end=0, step=-1, device=device).expand((batchSize, -1))
		return sequenceRegisterContextualWindowTokenTimes

	def calculateSequenceRegisterMemoryBankTokenTimes(self, sequenceRegisterMemoryBankAccessTimes):
		sequenceRegisterMemoryBankTokenTimes = torch.multiply(sequenceRegisterMemoryBankAccessTimes, sequenceRegisterTokenAccessTimeContextualWindow)
		return sequenceRegisterMemoryBankTokenTimes

	def calculateSequenceRegisterMemoryBankTheoreticalNoAccessTimes(self, sequenceRegisterMemoryBankTokenTimes):
		sequenceRegisterMemoryBankTheoreticalNoAccessTimes = torch.div(sequenceRegisterMemoryBankTokenTimes, sequenceRegisterTokenAccessTimeContextualWindow, rounding_mode="floor")	#or torch.floor_divide (depreciated)
		return sequenceRegisterMemoryBankTheoreticalNoAccessTimes

	def updateSequenceRegisterMemoryBankTokenTimes(self, sequenceRegisterMemoryBankTokenTimes):
		sequenceRegisterMemoryBankTokenTimes = torch.add(sequenceRegisterMemoryBankTokenTimes, sequenceRegisterTokenAccessTimeContextualWindow)
		return sequenceRegisterMemoryBankTokenTimes

	def calculateAttentionProbsMaxIndex(self, attention_scores):
		#attention_scores shape: batchSize * nheads * sequenceLength * sequenceLength
		attentionProbsMaxIndex = attention_scores
		attentionProbsMaxIndex = torch.permute(attentionProbsMaxIndex, (0, 2, 3, 1))
		attentionProbsMaxIndex = torch.topk(attentionProbsMaxIndex, k=tokenMemoryBankMaxAttentionHeads, dim=3).values
		attentionProbsMaxIndex = torch.permute(attentionProbsMaxIndex, (0, 3, 1, 2))
		attentionProbsMaxIndex = torch.argmax(attentionProbsMaxIndex, dim=3)	#batchSize * nheads * sequenceLength
		attentionProbsMaxIndex = attentionProbsMaxIndex.view(attentionProbsMaxIndex.shape[0], attentionProbsMaxIndex.shape[1]*attentionProbsMaxIndex.shape[2])	#merge contents from attention heads
		self.attentionProbsMaxIndex = attentionProbsMaxIndex
