"""SBNLPpt_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SBNLPpt_globalDefs.py

# Usage:
see SBNLPpt_globalDefs.py

# Description:
SBNLPpt globalDefs

"""

import torch
useLovelyTensors = False
if(useLovelyTensors):
	import lovely_tensors as lt
	lt.monkey_patch()
else:
	torch.set_printoptions(profile="full")
import math
import pynvml

debugCompareTokenMemoryBankPerformance = False
debugCreateOrderedDatasetFiles = False	#create data files comprising documents of sufficient length for createOrderedDataset
debugPrintPaths = False
debugPrintDataFileIndex = False
debugDoNotTrainModel = False
debugPrintLowHiddenSize = False
debugPrintMultipleModelAccuracy = False

#recursive algorithm selection:
useAlgorithmTransformer = True
useAlgorithmRNN = False
useAlgorithmSANI = False
useAlgorithmGIA = False

sortDataFilesByName = True	#orig; False

#syntactic bias selection (part 1):
recursiveLayers = True	#optional
memoryTraceBias = False	 #optional	#nncustom.Linear adjusts training/inference based on network prior activations
simulatedDendriticBranches = False	#optional #nncustom.Linear simulates multiple independent fully connected weights per neuron

memoryTraceAtrophy = False	#initialise (dependent var)

statePreprocessDataset = False	#only required once
stateTrainTokeniser = False	#only required once
stateTrainDataset = True
stateTestDataset = True	#requires reserveValidationSet

trainStartEpoch = 0	#start epoch of training (if continuing a training regime set accordingly >0)	#if trainStartEpoch=0 and trainStartDataFile=0 will recreate model, if trainStartEpoch>0 or trainStartDataFile>0 will load existing model
trainNumberOfEpochs = 1	#default: 10	#number of epochs to train (for production typically train x epochs at a time)
trainStartDataFile = 0	#default: 0	#start data file to train (if continuing a training regime set accordingly >0)	#if trainStartEpoch=0 and trainStartDataFile=0 will recreate model, if trainStartEpoch>0 or trainStartDataFile>0 will load existing model
trainNumberOfDataFiles = 100	#15	#100	#number of data files to train (for production typically train x dataFiles at a time)	#< datasetNumberOfDataFiles (30424) * trainSplitFraction
testNumberOfDataFiles = 1	#10

LRPdatabaseName = 'NLTK'	#wordnet
fixNLTKwordListAll = True	#add additional auxiliary having possessive words not found in NLTK word lists; ["has", "having", "'s"]

relativeFolderLocations = False
userName = 'user'	#default: user
tokenString = "INSERT_HUGGINGFACE_TOKEN_HERE"	#default: INSERT_HUGGINGFACE_TOKEN_HERE
import os
if(os.path.isdir('user')):
	from user.user_globalDefs import *

#storage location vars (requires 4TB harddrive);
datasetName = 'OSCAR1900'
#datasetName = 'OSCAR2201'

if(datasetName == 'OSCAR1900'):
	usePreprocessedDataset = True	#!usePreprocessedDataset not supported
elif(datasetName == 'OSCAR2201'):
	usePreprocessedDataset = False	#usePreprocessedDataset untested

if(usePreprocessedDataset):
	if(datasetName == 'OSCAR1900'):
		preprocessRemoveNewLineCharacters = True
		preprocessLimitNumberDataFiles = False
	elif(datasetName == 'OSCAR2201'):
		preprocessRemoveNewLineCharacters = False
		preprocessLimitNumberDataFiles = True	#preprocess a limited number of datafiles for tokeniser only
	if(preprocessLimitNumberDataFiles):
		preprocessNumberOfDataFiles = 1	#10	#100
else:
	preprocessRemoveNewLineCharacters = False
	
dataDrive = '/datasets/'
workingDrive = '/large/source/ANNpython/SBNLPpt/'

downloadCacheFolderName = 'cache'
dataFolderName = 'data'
modelFolderName = 'model'
LRPfolderName = 'LRPdata/' + LRPdatabaseName
if(relativeFolderLocations):
	downloadCachePathName = downloadCacheFolderName
	dataPathName = dataFolderName
	modelPathName = modelFolderName
	LRPpathName = LRPfolderName
else:	
	downloadCachePathName = '/media/' + userName + dataDrive + downloadCacheFolderName
	dataPathName = '/media/' + userName + dataDrive + dataFolderName
	modelPathName = '/media/' + userName + workingDrive + modelFolderName
	LRPpathName = '/media/' + userName + workingDrive + LRPfolderName
if(debugCreateOrderedDatasetFiles):
	dataFolderNameLargeDocuments = 'dataLargeDocuments'
dataPreprocessedFileNameStart = "/text_"
dataPreprocessedFileNameEnd = ".txt"
 
sequenceMaxNumTokensDefault = 512

#initialise (dependent vars);
useMultipleModels = False	
createOrderedDataset = False	
tokenMemoryBank = False	
tokenMemoryBankStorageSelectionAlgorithmAuto = False	
relativeTimeEmbeddings = False	
useGIAwordEmbeddings = False	
GIAsemanticRelationVectorSpaces = False 	
transformerPOSembeddings = False
transformerAttentionHeadPermutations = False	
transformerAttentionHeadPermutationsType = "none"	
transformerAttentionHeadPermutationsIndependent = False	
transformerAttentionHeadPermutationsIndependentOutput = False 
transformerSuperblocks = False

if(useAlgorithmTransformer):

	#syntactic bias selection (part 2):
	transformerPOSembeddings = False
	transformerAttentionHeadPermutations = False	#calculates KQ for all attention head permutations
	GIAsemanticRelationVectorSpaces = False	#use pretrained GIA word embeddings instead of nn.Embedding exclusively (transformer supports multiple embedding vectors)
	tokenMemoryBank = False	#apply attention to all tokens in sequenceRegister (standard contextualWindow + memoryBank), where memoryBank is updated based on recently attended tokens
	transformerSuperblocks = False	# super transformer blocks that wrap transfomer blocks (segregatedLayers)
	
	lowSequenceNumTokens = False
	mediumSequenceNumTokens = False	#initialise (dependent var)
	
	#initialise (dependent vars);
	transformerSuperblocksNumber = 1
	transformerSuperblocksRecursiveNumberIterations = 1
	recursiveLayersNumberIterations = 1
	recursiveLayersEmulateOrigImplementation = False
	
	#initialise (default vars);
	numberOfHiddenLayers = 6	#6	#default: 6
	numberOfAttentionHeads = 12	#default: 12	#numberOfAttentionHeadsDefault
	hiddenLayerSizeTransformer = 768	#default: 768 (can be overridden)

	if(transformerSuperblocks):
		transformerSuperblocksNumber = 2	#segregate nlp and logic layers
		transformerSuperblocksLayerNorm = True
		if(transformerSuperblocksLayerNorm):
			transformerSuperblocksLayerNormList = True	#separate norm function per layer
		transformerSuperblocksRecursive = False	#every super block is iterated multiple times
		if(transformerSuperblocksRecursive):
			transformerSuperblocksRecursiveNumberIterations = 2	#configure
			transformerSmall = True	#reduce GPU RAM
			if(transformerSmall):
				hiddenLayerSizeTransformer = 256
				numberOfHiddenLayers = 2
				numberOfAttentionHeads = 1	#prevents recursion across different attention heads, nullifying precise recursion
	if(recursiveLayers):
		recursiveLayersEmulateOrigImplementation = True	#emulate orig implementation so that archived models can be reloaded
		if(recursiveLayersEmulateOrigImplementation):
			recursiveLayersNumberIterations = numberOfHiddenLayers	#numberOfHiddenLayers is interpreted as recursiveLayersNumberIterations
		else:
			recursiveLayersNumberIterations = 2
			numberOfAttentionHeads = 1	#prevents recursion across different attention heads, nullifying precise recursion


	if(transformerAttentionHeadPermutations):
		transformerAttentionHeadPermutationsIndependentOutput = True	#SelfOutput executes dense linear in groups of size numberOfAttentionHeads	#does not support sharedLayerWeightsOutput
		transformerAttentionHeadPermutationsType = "dependent"	#perform softmax over all permutations (rather than over each permutation independently)
		#transformerAttentionHeadPermutationsType = "independent"
		mediumSequenceNumTokens = False	#optional
		if(mediumSequenceNumTokens):
			numberOfAttentionHeads = 12
		else:
			if(transformerAttentionHeadPermutationsType=="dependent"):
				numberOfAttentionHeads = 4
				hiddenLayerSizeTransformer = hiddenLayerSizeTransformer//numberOfAttentionHeads	#eg 192
			else:
				numberOfAttentionHeads = 4	#or 8 (slow)
			
	if(GIAsemanticRelationVectorSpaces):
		useGIAwordEmbeddings = True
	if(tokenMemoryBank):
		mediumSequenceNumTokens = True
		tokenMemoryBankStorageSelectionAlgorithmAuto = True	#automatically learn tokenMemoryBank storage selection algorithm
		if(tokenMemoryBankStorageSelectionAlgorithmAuto):
			useMultipleModels = True
			tokenMemoryBankStorageSelectionBinaryThreshold = 0.2	#0.2	#<0.5: decrease tokenMemoryBankStorageSelectionBinaryThreshold to ensure a majority of contextualWindow tokens are forgotten (not added to memory)	#[biased forget] 0->0.5 [no bias] 0.5->1.0 [biased remember]
			tokenMemoryBankStorageSelectionNormaliseForgetRememberSize = True
			if(tokenMemoryBankStorageSelectionNormaliseForgetRememberSize):
				tokenMemoryBankStorageSelectionNormaliseForgetRememberSizeBias = 0.5	#0.2	 #[biased forget] 0->0.5 [no bias] 0.5->1.0 [biased remember]
			tokenMemoryBankStorageSelectionInitiationBias = True
			if(tokenMemoryBankStorageSelectionInitiationBias):
				tokenMemoryBankStorageSelectionInitiationBiasOutputLayerMean = -0.1	#bias initiation of tokenMemoryBankStorageSelectionModel to not select tokens for memory bank insertion (such that sufficient number of tokens are forgotten to successfully train the model)
			debugTokenMemoryBankStorageSelectionAlgorithmAuto = True
			debugPrintMultipleModelAccuracy = True
		else:
			onlyAddAttendedContextualWindowTokensToMemoryBank = True	#optional #saves memory bank space by only adding attended contextual window tokens to memory bank 
		tokenMemoryBankMaxAttentionHeads = 12	#12	#1	#maximum number of attention heads to identify important tokens to remember	#max value allowed = numberOfAttentionHeads (12)
		createOrderedDataset = True
		#tokenMemoryBank algorithm requires continuous/contiguous textual input	#batchSize > 0, will need to feed contiguous input for each sample in batch
		relativeTimeEmbeddings = True	#attention scores are weighted based on a (learnable) function of the relative age between queried/keyed tokens
		if(lowSequenceNumTokens):
			memoryBankSizeMultiplier = 4
		else:
			memoryBankSizeMultiplier = (sequenceMaxNumTokensDefault//sequenceMaxNumTokens)	#*tokenMemoryBankMaxAttentionHeads	#relative size of transformer window with memory bank relative to standard transformer contextual window	#determines max number of tokens to be stored in memory bank
		sequenceRegisterContextualWindowLength = sequenceMaxNumTokens
		if(sequenceMaxNumTokens == sequenceMaxNumTokensDefault):
			sequenceRegisterMemoryBankLength = sequenceRegisterContextualWindowLength
		else:
			sequenceRegisterMemoryBankLength = sequenceRegisterContextualWindowLength*(memoryBankSizeMultiplier-1)
		#orig: sequenceRegisterMemoryBankLength = sequenceRegisterContextualWindowLength*memoryBankSizeMultiplier
		sequenceRegisterLength = sequenceRegisterContextualWindowLength + sequenceRegisterMemoryBankLength
		sequenceRegisterMaxActivationTime = memoryBankSizeMultiplier+1	#how long to remember unaccessed tokens	#will depend on memoryBankSizeMultiplier (not directly proportional, but this is a rough heuristic)
		sequenceRegisterRenewTime = 0	#if tokens are accessed, what time to renew them to	#interpretation: access time 0 = recently activated
		sequenceRegisterTokenAccessTimeContextualWindow = sequenceMaxNumTokens	#how to adjust the access time of a given token last accessed in a previous contextual window 	#will depend on sequenceMaxNumTokens (not directly equal, but this is a rough heuristic)
		sequenceRegisterVerifyMemoryBankSize = True	#if false, need to set memory bank size sufficiently high such that will never run out of space for retained tokens
		sequenceRegisterMemoryBankPaddingAccessTime = sequenceRegisterMaxActivationTime	#set access time of padding high to ensure that it will learn to be ignored (does not interfere with positional calculations); may not be required given that all hidden states are zeroed
		sequenceRegisterMemoryBankPaddingTokenTime = sequenceRegisterMemoryBankPaddingAccessTime*sequenceMaxNumTokens
		calculateMemoryBankTokenTimesFromAccessTimes = False #calculate memory bank token times based on last access times
		sequenceRegisterMaxTokenTime = (orderedDatasetDocNumberSegmentsDefault+1)*sequenceMaxNumTokensDefault	#CHECKTHIS: +1 because sequenceRegisterLength = sequenceRegisterContextualWindowLength + sequenceRegisterMemoryBankLength
		debugPrintSequenceRegisterRetainSize = False
		assert debugCompareTokenMemoryBankPerformance == False
	else:
		if(debugCompareTokenMemoryBankPerformance):
			sequenceRegisterLength = sequenceMaxNumTokens
		else:
			sequenceMaxNumTokens = sequenceMaxNumTokensDefault	#window length (transformer)
	if(lowSequenceNumTokens):
		sequenceMaxNumTokens = 8	#8 16 32 64
		orderedDatasetDocNumberSegmentsDefault = 1
	else:
		if(mediumSequenceNumTokens):
			sequenceMaxNumTokens = 128	#128 256 
		else:
			sequenceMaxNumTokens = sequenceMaxNumTokensDefault		#512	#default: sequenceMaxNumTokensDefault	#override
		orderedDatasetDocNumberSegmentsDefault = 10
else:
	sequenceMaxNumTokens = sequenceMaxNumTokensDefault	#window length (RNN/SANI)

if(debugCompareTokenMemoryBankPerformance):
	createOrderedDataset = True
if(debugCreateOrderedDatasetFiles):
	createOrderedDataset = True

useTrainedTokenizer = True	#initialise (dependent var)
useFullwordTokenizer = False	#initialise (dependent var)
useFullwordTokenizerClass = True	#initialise (dependent var)
tokeniserOnlyTrainOnDictionary = False	#initialise (dependent var)
useEffectiveFullwordTokenizer = False	#initialise (dependent var)
GIAmemoryTraceAtrophy = False	#initialise (dependent var)
if(transformerPOSembeddings):
	GIAuseVectorisedPOSidentification = True
	useEffectiveFullwordTokenizer = True
if(useAlgorithmGIA):
	GIAsemanticRelationVectorSpaces = True
	useGIAwordEmbeddings = True	#required to prepare for useAlgorithmTransformer with useGIAwordEmbeddings compatibility
if(GIAsemanticRelationVectorSpaces):
	GIAuseOptimisedEmbeddingLayer = True	#currently required
	if(GIAuseOptimisedEmbeddingLayer):
		GIAuseOptimisedEmbeddingLayer1 = False	#use nn.Embedding instead of nn.Linear	#incomplete (results in incorrect embedding shape)
		GIAuseOptimisedEmbeddingLayer2 = True	#generate n.Embedding posthoc from nn.Linear 
	GIAuseVectorisedSemanticRelationIdentification = True	#optional
	if(GIAuseVectorisedSemanticRelationIdentification):
		GIAuseVectorisedPOSidentification = True
	useEffectiveFullwordTokenizer = True	#required for useAlgorithmGIA
	GIAsuppressWordEmbeddingsForInvalidPOStype = True  #suppress GIA word embeddings (to zero) if wrong pos type - atrophy word vector weights for dict inputs that are never used
	if(GIAsuppressWordEmbeddingsForInvalidPOStype):
		GIAmemoryTraceAtrophy = True	#use LinearCustom MTB for GIA word embedding weights
	if(useGIAwordEmbeddings):
		GIAgenerateUniqueWordVectorsForRelationTypes = True	#only generate unique word vectors for semantic relation types (else use generic/noun word vectors)
		GIArelationTypesIntermediate = False	#useGIAwordEmbeddings does not currently support intermediate semantic relation word embeddings, as this would require the language model itself (eg transformer) to read token context to generate the word embeddings	#GIArelationTypesIntermediate: verb/preposition relations are defined by both subject and object tokens
	else:
		GIAgenerateUniqueWordVectorsForRelationTypes = False	#optional: generate word vectors for semantic relations
		GIArelationTypesIntermediate = True	#optional
		
	debugPrintModelPropagation = False
	debugPrintRelationExtractionProgress = False
	debugTruncateBatch = False	#reduce GPU memory	#only add a single batch of model samples
	debugReduceEmbeddingLayerSize = False	#reduce GPU memory	#not supported by useGIAwordEmbeddings
	debugUseSmallNumberOfModels = False	#reduce GPU memory
	debugDoNotTrainModel = False	#reduce GPU memory
	
	encode3tuples = True
	if(useAlgorithmGIA):
		useMultipleModels = True
	if(useGIAwordEmbeddings):
		useFullwordTokenizer = False	#useGIAwordEmbeddings requires GIAsemanticRelationVectorSpaces tokenisation to be the same as that used by transformer (ie trainTokeniserSubwords)
	else:
		useFullwordTokenizer = False	#optional	#tokenizer only identifies whole words
	if(useFullwordTokenizer):
		tokeniserOnlyTrainOnDictionary = True	#optional
		useFullwordTokenizerNLTK = False	#optional	#else use DistilBertTokenizer.basic_tokenizer.tokenize
		useFullwordTokenizerPretrained = False	#optional	#required for latest version of transformers library
		if(useFullwordTokenizerPretrained):
			useFullwordTokenizerPretrainedAuto = True	#optional
		else:
			useFullwordTokenizerFast = False	#optional
			if(not useFullwordTokenizerFast):
				useFullwordTokenizerClass = False
			useTrainedTokenizer = False
			tokensVocabPathName = modelPathName + "/" + "vocab-fullword.json"
			tokensSpecialPathName = modelPathName + "/" + "special_tokens-fullword.json" 
	else:
		tokeniserOnlyTrainOnDictionary = True	#optional	#ensures effective fullword tokenisation of dictionary words #CHECKTHIS
	useIndependentReverseRelationsModels = False	#initialise (dependent var)
	if(not GIAgenerateUniqueWordVectorsForRelationTypes):
		useIndependentReverseRelationsModels = False	#else take input linear layer as forward embeddings and output linear layer [inversed] as reverse embeddings
		
if(recursiveLayers or memoryTraceBias or simulatedDendriticBranches or GIAsemanticRelationVectorSpaces or tokenMemoryBank or transformerAttentionHeadPermutations or transformerPOSembeddings or transformerSuperblocks):
	useSyntacticBiases = True
else:
	useSyntacticBiases = False
if(memoryTraceBias or simulatedDendriticBranches):
	useLinearCustom = True
else:
	debugCustomLinearFunctionClass = False
	if(debugCustomLinearFunctionClass):
		useLinearCustom = True
	else:
		useLinearCustom = False
if(not simulatedDendriticBranches):
	debugIndependentTestDisablePostInit = False
if(useLinearCustom):
	useModuleLinearTemplateCurrent = True	#use current version of class Linear(nn.Module) from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py (instead of https://pytorch.org/docs/master/notes/extending.html)
	useAutoResizeInput = False	#legacy implementation	#required for original python LinearFunction implementation (compared to C++ implementation and modified python LinearFunction)	#see https://discuss.pytorch.org/t/exact-python-implementation-of-linear-function-class/170177

if(memoryTraceBias):
	createOrderedDataset = True
	#FUTURE: require update of SBNLPpt_data to ensure that continuous/contiguous textual input (for inference) is provided
	memoryTraceBiasHalflife = 2.0	#number of batches of batchSize=1 (ie sequences) to pass before memoryTrace is halved
	memoryTraceBiasWeightDirectionDependent = True
	memoryTraceBiasSigned = False	#initialise (dependent var)
	if(memoryTraceBiasWeightDirectionDependent):
		memoryTraceBiasWeightDependent = True	#optional: memory trace dependent on the strength of the weights
		memoryTraceBiasSigned = True	#optional: calculate positive/negative memory trace: positive memory trace calculated based on true positives and true negatives, negative memory trace calculated based on false positives and false negatives
	if(memoryTraceBiasSigned):
		normaliseActivationSparsity = False
	else:
		normaliseActivationSparsity = True
if(memoryTraceAtrophy or GIAmemoryTraceAtrophy):
	#only weaken unused connections (do not strengthen used connections)
	memoryTraceAtrophyMultiplication = True
	if(memoryTraceAtrophyMultiplication):
		memoryTraceAtrophyRate = 0.0001	#per batch: amount to decrease weights if input is zero
	else:
		memoryTraceAtrophyRate = 0.00001	#per batch: amount to decrease weights if input is zero
				
if(simulatedDendriticBranches):
	numberOfIndependentDendriticBranches = 2	#10	#2	#depends on GPU RAM (note recursiveLayers reduces RAM usage)
	normaliseActivationSparsity = True
else:
	numberOfIndependentDendriticBranches = 1
	
if(useAlgorithmTransformer):
	if(not useSyntacticBiases):
		officialRobertaBaseModel = False	#optional	#loads official huggingface model with default parameters - https://huggingface.co/roberta-base/tree/main
	else:
		officialRobertaBaseModel = False	#mandatory
	
useSmallDatasetDebug = False
useSingleHiddenLayerDebug = False
usePretrainedModelDebug = False	#executes stateTestDataset only	#useAlgorithmTransformer only
useSmallBatchSizeDebug = False

useSmallTokenizerTrainNumberOfFiles = True	#used during rapid testing only (FUTURE: assign est 80 hours to perform full tokenisation train)
if(useSmallTokenizerTrainNumberOfFiles):
	if(useFullwordTokenizer):
		trainTokenizerNumberOfFilesToUseSmall = 100	#100	#default: 100	#100: 2 hours
	else:
		trainTokenizerNumberOfFilesToUseSmall = 100	#100	#default 1000	#100: 15 min, 1000: 3.75 hours

reserveValidationSet = True	#reserves a fraction of the data for validation
trainSplitFraction = 0.9	#90% train data, 10% test data

if(useAlgorithmTransformer):
	batchSize = 8	#default: 8	#8 and 16 train at approx same rate (16 uses more GPU ram)	#depends on GPU RAM	#with 12GB GPU RAM, batchSize max = 16
	learningRate = 1e-4
elif(useAlgorithmRNN):
	batchSize = 8
	learningRate = 1e-4
elif(useAlgorithmSANI):
	batchSize = 8	#4	#8	#2	#depends on GPU memory
	learningRate = 1e-4
elif(useAlgorithmGIA):
	if(debugTruncateBatch):
		batchSize = 1	#useAlgorithmGIAsemanticRelationVectorSpace batchSize is dynamic (>batchSize)	
	else:
		batchSize = 8	#low batch size is not required for high vocabularySize (since useAlgorithmGIA model does not propagate every token in sequence contextual window simulataneously)
	learningRate = 1e-4

GPUramLimit12to32GB = True
if(GPUramLimit12to32GB):
	if(simulatedDendriticBranches):
		batchSize = batchSize//4	#requires more GPU RAM (reduced batchSize)
	if(memoryTraceBias):
		batchSize = 1	#CHECKTHIS - memoryTraceBias algorithm requires continuous/contiguous textual input
	if(tokenMemoryBank or debugCompareTokenMemoryBankPerformance):
		if((sequenceRegisterLength // sequenceMaxNumTokensDefault) > 1):
			batchSize = 2
	if(useGIAwordEmbeddings):
		batchSize = 2	#low batch size is required for high vocabularySize
	if(transformerAttentionHeadPermutations):
		if(transformerAttentionHeadPermutationsType=="independent"):
			batchSize = 8
		elif(transformerAttentionHeadPermutationsType=="dependent"):
			batchSize = 8
	if(transformerPOSembeddings):
		batchSize = 2
if(useSmallBatchSizeDebug):
	batchSize = 1	#use small batch size to enable simultaneous execution (GPU ram limited) 
print("batchSize = ", batchSize)
	
numberOfDocumentsPerDataFile = 10000	#if !usePreprocessedDataset; numberOfDocumentsPerDataFile = number of documents per artificial datafile index (e.g. trainNumberOfDataFiles)
if(datasetName == 'OSCAR1900'):
	datasetNumberOfDocuments = 304230423	#orig: dataFileLastIndex*numberOfDocumentsPerDataFile + datasetNumberOfSamplesPerDataFileLast = 30423*10000 + 423
	datasetNumberOfDataFiles =	math.ceil(datasetNumberOfDocuments/numberOfDocumentsPerDataFile) #30424
	datasetNumberOfSamplesPerDataFileLast = datasetNumberOfDocuments%numberOfDocumentsPerDataFile	#423
elif(datasetName == 'OSCAR2201'):
	datasetNumberOfDocuments = 431992659	#number of documents	#https://huggingface.co/datasets/oscar-corpus/OSCAR-2201
	datasetNumberOfDataFiles =	math.ceil(datasetNumberOfDocuments/numberOfDocumentsPerDataFile)
	datasetNumberOfSamplesPerDataFileLast = datasetNumberOfDocuments%numberOfDocumentsPerDataFile
dataFileLastIndex = datasetNumberOfDataFiles-1

modelSaveNumberOfBatches = 1000	#resave model after x training batches


#transformer only;
customMaskTokenID = 4	#3
fractionOfMaskedTokens = 0.15

#Warning: if change vocabularySize, require reexecution of python SBNLPpt_GIAdefinePOSwordLists.py (LRPdata/NLTK/wordlistVector*.txt)
if(useEffectiveFullwordTokenizer):
	if(useFullwordTokenizer):
		vocabularySize = 2000000	#approx number of unique words in dataset
	else:
		vocabularySize = 240000		#approx number of unique words in english	#236736	#requirement: must be >= size of NLTK wordlistAll.txt
else:
	vocabularySize = 30522	#default: 30522	#number of independent tokens identified by SBNLPpt_data.trainTokeniserSubwords

accuracyTopN = 1	#default: 1	#>= 1	#calculates batch accuracy based on top n dictionary predictions

specialTokens = ['<s>', '<pad>', '</s>', '<unk>', '<mask>']
specialTokenPadding = '<pad>'
specialTokenMask = '<mask>'

if(createOrderedDataset):
	if(sequenceMaxNumTokens > sequenceMaxNumTokensDefault):	#eg debugCompareTokenMemoryBankPerformance and sequenceMaxNumTokens=1024
		orderedDatasetDocNumberSegments = orderedDatasetDocNumberSegmentsDefault
		sufficientLengthMultiplier = sequenceMaxNumTokens//sequenceMaxNumTokensDefault
	else:
		orderedDatasetDocNumberSegments = orderedDatasetDocNumberSegmentsDefault * sequenceMaxNumTokensDefault//sequenceMaxNumTokens
		print("orderedDatasetDocNumberSegments = ", orderedDatasetDocNumberSegments)
		sufficientLengthMultiplier = 1
	orderedDatasetDocNumberTokens = orderedDatasetDocNumberSegments*sequenceMaxNumTokens
	orderedDatasetDocMinSizeCharacters = 10000	#prevents having to tokenise small document samples to count number of tokens
	orderedDatasetSplitDocumentsBySentences = False
	orderedDatasetDocumentProbabilityOfSufficientLength = pow(0.15, sufficientLengthMultiplier) #sequenceMaxNumTokens=512:0.15	#sequenceMaxNumTokens=1024:0.04	#0.01	#min probability that a document is of sufficient length that it can be split into orderedDatasetDocNumberSegments
	dataFilesFeedMultiplier = (1/orderedDatasetDocumentProbabilityOfSufficientLength)	#math.ceil
else:
	dataFilesFeedMultiplier = 1

printAccuracyRunningAverage = True
if(printAccuracyRunningAverage):
	runningAverageBatches = 10

def getModelPathNameFull(modelPathNameBase, modelName):
	modelPathNameFull = modelPathNameBase + '/' + modelName + '.pt'
	return modelPathNameFull
	
GIAmodelName = 'modelGIA'	
if(useAlgorithmTransformer):
	sharedLayerWeights = False	#initialise (dependent var)
	sharedLayerWeightsOutput = False	#initialise (dependent var)
	if(not usePretrainedModelDebug):
		if(recursiveLayers):
			sharedLayerWeights = False	#orig recursiveLayers implementation
			if(sharedLayerWeights):
				sharedLayerWeightsOutput = True	#share RobertaOutputSharedLayerOutput/RobertaSelfOutputSharedLayerOutput parameters also
			recursiveLayersNormaliseNumParameters = False	#default: True	#optional	#if use recursiveLayers normalise/equalise num of parameters with respect to !recursiveLayers
			if(recursiveLayersNormaliseNumParameters):
				recursiveLayersNormaliseNumParametersIntermediate = True	#normalise intermediateSize parameters also
		else:
			recursiveLayersNormaliseNumParameters = False	#mandatory
	
		if(officialRobertaBaseModel):
			numberOfHiddenLayers = 12	#default values
		else:
			if(useSingleHiddenLayerDebug):
				numberOfHiddenLayers = 1

		if(debugPrintLowHiddenSize):
			hiddenLayerSize = 24
		else:
			hiddenLayerSize = hiddenLayerSizeTransformer
		intermediateSize = 3072	#default: 3072
		if(recursiveLayers):
			#same model size irrespective of useSingleHiddenLayerDebug
			if(recursiveLayersNormaliseNumParameters):
				if(sharedLayerWeights):
					if(sharedLayerWeightsOutput):
						if(recursiveLayersNormaliseNumParametersIntermediate):
							hiddenLayerSizeMultiplier = (7/4)	#model size = 249MB	
							#hiddenLayerSizeMultiplier = (5/3)	#~230MB	
						else:
							hiddenLayerSizeMultiplier = 2	#model size = ~255MB
					else:
						if(recursiveLayersNormaliseNumParametersIntermediate):
							hiddenLayerSizeMultiplier = (4/3)	#model size = 273MB
						else:
							hiddenLayerSizeMultiplier = 1.5	#model size = ~255MB
				else:
					hiddenLayerSizeMultiplier = (7/4)	#model size = ~250MB	#optimisation failure observed
					#hiddenLayerSizeMultiplier = (11/6)	#model size = ~265MB	#optimisation failure observed
					#hiddenLayerSizeMultiplier = 2.0	#model size = ~280MB	#optimisation failure observed

				hiddenLayerSize = round(hiddenLayerSize*hiddenLayerSizeMultiplier)
				numberOfAttentionHeads = round(numberOfAttentionHeads*hiddenLayerSizeMultiplier)	#or: round(numberOfAttentionHeads)
				if(recursiveLayersNormaliseNumParametersIntermediate):
					intermediateSize = round(intermediateSize*hiddenLayerSizeMultiplier)
				print("hiddenLayerSize = ", hiddenLayerSize)
				print("numberOfAttentionHeads = ", numberOfAttentionHeads)
				print("intermediateSize = ", intermediateSize)
			else:
				if(sharedLayerWeights):
					if(sharedLayerWeightsOutput):
						pass	#model size = ~120MB
					else:
						pass	#model size = 176.7MB
				else:
					pass	#model size = 120.4MB
		else:
			if(useSingleHiddenLayerDebug):
				pass	#model size = 120.4MB
			else:
				pass	#model size = 255.6MB
			
		if(tokenMemoryBankStorageSelectionAlgorithmAuto):
			tokenMemoryBankStorageSelectionModelInputLayerSize = hiddenLayerSize
			tokenMemoryBankStorageSelectionModelHiddenLayerSize = 768//2	#between hiddenLayerSize (768) and number of outputs (2:store/delete)
			tokenMemoryBankStorageSelectionModelOutputLayerSize = 1	#1: BCELoss, 2: CrossEntropyLoss
			
			modelName = 'modelTransformer'
			tokenMemoryBankStorageSelectionModelName = 'modelTokenMemoryBankStorageSelection'	
			
			if(recursiveLayers):
				if(sharedLayerWeights):
					numberOfHiddenLayersTokenMemoryBankParameters = numberOfHiddenLayers
				else:
					numberOfHiddenLayersTokenMemoryBankParameters = numberOfHiddenLayers	#1
			else:
				numberOfHiddenLayersTokenMemoryBankParameters = numberOfHiddenLayers
elif(useAlgorithmRNN):
	hiddenLayerSize = 1024	#65536	#2^16 - large hidden size is required for recursive RNN as parameters are shared across a) sequence length and b) number of layers
	if(SBNLPpt_RNNmodel.applyIOconversionLayers):
		embeddingLayerSize = 768
	else:
		embeddingLayerSize = hiddenLayerSize

	numberOfHiddenLayers = 6

	modelName = 'modelRNN'
	modelPathNameFull = getModelPathNameFull(modelPathName, modelName)

	useBidirectionalRNN = False
	if(useBidirectionalRNN):
		bidirectional = 2
	else:
		bidirectional = 1
elif(useAlgorithmSANI):
	hiddenLayerSize = 1024	#1024	#8192	#1024	#depends on GPU memory	#2^16 = 65536 - large hidden size is required for recursive SANI as parameters are shared across a) sequence length and b) number of layers
	if(SBNLPpt_SANImodel.applyIOconversionLayers):
		embeddingLayerSize = 768
	else:
		embeddingLayerSize = hiddenLayerSize

	modelName = 'modelSANI'
	modelPathNameFull = getModelPathNameFull(modelPathName, modelName)
	
	#useBidirectionalSANI = False	#not currently supported
	#if(useBidirectionalSANI):
	#	bidirectional = 2
	#else:
	#	bidirectional = 1
elif(useAlgorithmGIA):
	modelName = GIAmodelName
	#modelPathNameFull = getModelPathNameFull(modelPathName, modelName)

if(transformerPOSembeddings):
	POSembeddingSize = 24	#sync with len(wordListVectorsDictAll)
	pretrainedHiddenSize = POSembeddingSize
	trainableHiddenSize = hiddenLayerSizeTransformer - POSembeddingSize
if(GIAsemanticRelationVectorSpaces):
	#ensure vectorSpaceListLen is a factor of hiddenLayerSizeTransformer
	if(GIAgenerateUniqueWordVectorsForRelationTypes):
		if(GIArelationTypesIntermediate):
			vectorSpaceListLen = 8
		else:
			vectorSpaceListLen = 8
	else:
		vectorSpaceListLen = 8	
		if(useIndependentReverseRelationsModels):
			vectorSpaceListLen = vectorSpaceListLen + vectorSpaceListLen
	
	if(GIAgenerateUniqueWordVectorsForRelationTypes):
		trainableEmbeddingSpaceFraction = 2	#interpretation: 1/x	#currently assign equal amount of embedding space memory to pretrained GIA word embeddings (pertaining to relation type tokens) and transformer trained word embeddings (pertaining to non-relation type tokens)
		embeddingListLen = vectorSpaceListLen * trainableEmbeddingSpaceFraction	
	else:
		embeddingListLen = vectorSpaceListLen
		
	if(debugReduceEmbeddingLayerSize):
		embeddingLayerSize = 10
	else:
		embeddingLayerSize = hiddenLayerSizeTransformer//embeddingListLen	#word vector embedding size (cany vary based on GIA word vector space)
		if(GIAgenerateUniqueWordVectorsForRelationTypes):
			pretrainedHiddenSize = hiddenLayerSizeTransformer//trainableEmbeddingSpaceFraction	#explicated for debugging only
			trainableHiddenSize = hiddenLayerSizeTransformer - pretrainedHiddenSize	#explicated for debugging only

def printCUDAmemory(tag):
	print(tag)
	
	pynvml.nvmlInit()
	h = pynvml.nvmlDeviceGetHandleByIndex(0)
	info = pynvml.nvmlDeviceGetMemoryInfo(h)
	total_memory = info.total
	memory_free = info.free
	memory_allocated = info.used
	'''
	total_memory = torch.cuda.get_device_properties(0).total_memory
	memory_reserved = torch.cuda.memory_reserved(0)
	memory_allocated = torch.cuda.memory_allocated(0)
	memory_free = memory_reserved-memory_allocated  # free inside reserved
	'''
	print("CUDA total_memory = ", total_memory)
	#print("CUDA memory_reserved = ", memory_reserved)
	print("CUDA memory_allocated = ", memory_allocated)
	print("CUDA memory_free = ", memory_free)

def printe(str):
	print(str)
	exit()
