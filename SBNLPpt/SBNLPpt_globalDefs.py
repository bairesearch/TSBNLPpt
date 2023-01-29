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


useLovelyTensors = True
if(useLovelyTensors):
	import lovely_tensors as lt
	lt.monkey_patch()
else:
	import torch as pt
	pt.set_printoptions(profile="full")
import math

#recursive algorithm selection:
useAlgorithmTransformer = True
useAlgorithmRNN = False
useAlgorithmSANI = False
useAlgorithmGIA = False

sortDataFilesByName = True	#orig; False

recursiveLayers = True	#optional
memoryTraceBias = False	 #optional	#nncustom.Linear adjusts training/inference based on network prior activations
simulatedDendriticBranches = False	#optional #nncustom.Linear simulates multiple independent fully connected weights per neuron

statePreprocessDataset = False	#only required once
stateTrainTokeniser = False	#only required once
stateTrainDataset = True
stateTestDataset = False	#requires reserveValidationSet

trainStartEpoch = 0	#start epoch of training (if continuing a training regime set accordingly >0)	#if trainStartEpoch=0 and trainStartDataFile=0 will recreate model, if trainStartEpoch>0 or trainStartDataFile>0 will load existing model
trainNumberOfEpochs = 1	#default: 10	#number of epochs to train (for production typically train x epochs at a time)
trainStartDataFile = 0	#default: 0	#start data file to train (if continuing a training regime set accordingly >0)	#if trainStartEpoch=0 and trainStartDataFile=0 will recreate model, if trainStartEpoch>0 or trainStartDataFile>0 will load existing model
trainNumberOfDataFiles = 100	#100	#default: -1 (all)	#number of data files to train (for production typically train x dataFiles at a time)	#< numberOfDataFiles (30424) * trainSplitFraction
testNumberOfDataFiles = 10	#10		#default: -1 (all)

LRPdatabaseName = 'NLTK'	#wordnet
fixNLTKwordListAll = True	#add additional auxiliary having possessive words not found in NLTK word lists; ["has", "having", "'s"]

relativeFolderLocations = False
userName = 'user'	#default: user
tokenString = "INSERT_HUGGINGFACE_TOKEN_HERE"	#default: INSERT_HUGGINGFACE_TOKEN_HERE
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
if(not relativeFolderLocations):
	downloadCachePathName = '/media/' + userName + dataDrive + downloadCacheFolderName
	dataPathName = '/media/' + userName + dataDrive + dataFolderName
	modelPathName = '/media/' + userName + workingDrive + modelFolderName
	LRPpathName = '/media/' + userName + workingDrive + LRPfolderName

sequenceMaxNumTokens = 512	#window length (transformer/RNN/SANI)

createOrderedDataset = False	#initialise (dependent var)	#CHECKTHIS
tokenMemoryBank = False	#initialise (dependent var)
relativeTimeEmbeddings = False	#initialise (dependent var)
if(useAlgorithmTransformer):
	tokenMemoryBank = True	#apply attention to all tokens in sequenceRegister (standard contextualWindow + memoryBank), where memoryBank is updated based on recently attended tokens
	tokenMemoryBankMaxAttentionHeads = 1	#maximum number of attention heads to identify important tokens to remember	#max value allowed = numberOfAttentionHeads (12)
	if(tokenMemoryBank):
		createOrderedDataset = True
		#tokenMemoryBank algorithm requires continuous/contiguous textual input	#batchSize > 0, will need to feed contiguous input for each sample in batch
		relativeTimeEmbeddings = True	#attention scores are weighted based on a (learnable) function of the relative age between queried/keyed tokens
		memoryBankSizeMultiplier = 1*tokenMemoryBankMaxAttentionHeads	#relative size of transformer window with memory bank relative to standard transformer contextual window	#determines max number of tokens to be stored in memory bank
		sequenceRegisterContextualWindowLength = sequenceMaxNumTokens
		sequenceRegisterMemoryBankLength = sequenceRegisterContextualWindowLength*memoryBankSizeMultiplier
		sequenceRegisterLength = sequenceRegisterContextualWindowLength + sequenceRegisterMemoryBankLength
		sequenceRegisterMaxActivationTime = memoryBankSizeMultiplier+1	#how long to remember unaccessed tokens	#will depend on memoryBankSizeMultiplier (not directly proportional, but this is a rough heuristic)
		sequenceRegisterRenewTime = 0	#if tokens are accessed, what time to renew them to
		sequenceRegisterTokenAccessTimeContextualWindow = sequenceMaxNumTokens	#how to adjust the access time of a given token last accessed in a previous contextual window 	#will depend on sequenceMaxNumTokens (not directly equal, but this is a rough heuristic)
		sequenceRegisterVerifyMemoryBankSize = True	#if false, need to set memory bank size sufficiently high such that will never run out of space for retained tokens
		sequenceRegisterMemoryBankPaddingAccessTime = sequenceRegisterMaxActivationTime	#set access time of padding high to ensure that it will learn to be ignored (does not interfere with positional calculations); may not be required given that all hidden states are zeroed
		debugPrintSequenceRegisterRetainSize = False
	
semanticRelationVectorSpaces = False
useMultipleModels = False
useTrainedTokenizer = True
useFullwordTokenizer = False
useFullwordTokenizerClass = True	
tokeniserOnlyTrainOnDictionary = False
debugDoNotTrainModel = False
useEffectiveFullwordTokenizer = False
if(useAlgorithmGIA):
	semanticRelationVectorSpaces = True
	useVectorisedSemanticRelationIdentification = True	#optional
	useEffectiveFullwordTokenizer = True	#required for useAlgorithmGIA

	debugPrintModelPropagation = False
	debugPrintRelationExtractionProgress = False
	debugTruncateBatch = True	#reduce GPU memory during
	debugReduceEmbeddingLayerSize = True	#reduce GPU memory during
	debugUseSmallNumberOfModels = False	#reduce GPU memory
	debugDoNotTrainModel = False	#reduce GPU memory
	
	encode3tuples = True
	useMultipleModels = True
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
		tokeniserOnlyTrainOnDictionary = True	#optional	#ensures effective fullword tokenisation of dictionary words
	useIndependentReverseRelationsModels = False	#else take input linear layer as forward embeddings and output linear layer [inversed] as reverse embeddings

if(recursiveLayers or memoryTraceBias or simulatedDendriticBranches or semanticRelationVectorSpaces or tokenMemoryBank):
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
	memoryTraceWeightDirectionDependent = True
	memoryTraceSigned = False	#initialise (dependent var)
	if(memoryTraceWeightDirectionDependent):
		memoryTraceWeightDependent = True	#optional: memory trace dependent on the strength of the weights
		memoryTraceSigned = True	#optional: calculate positive/negative memory trace: positive memory trace calculated based on true positives and true negatives, negative memory trace calculated based on false positives and false negatives
	if(memoryTraceSigned):
		normaliseActivationSparsity = False
	else:
		normaliseActivationSparsity = True
	
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
		batchSize = 8
	learningRate = 1e-4
	
if(simulatedDendriticBranches):
	batchSize = batchSize//4	#requires more GPU RAM (reduced batchSize)
if(memoryTraceBias):
	batchSize = 1	#CHECKTHIS - memoryTraceBias algorithm requires continuous/contiguous textual input
if(tokenMemoryBank):
	batchSize = 2	#CHECKTHIS - tokenMemoryBank algorithm requires continuous/contiguous textual input	#batchSize > 0, will need to feed contiguous input for each sample in batch
		
if(useSmallBatchSizeDebug):
	batchSize = 1	#use small batch size to enable simultaneous execution (GPU ram limited) 
	
numberOfDocumentsPerDataFile = 10000	#if !usePreprocessedDataset; numberOfDocumentsPerDataFile = number of documents per artificial datafile index (e.g. trainNumberOfDataFiles)
if(datasetName == 'OSCAR1900'):
	numberOfDocuments = 304230423	#orig: dataFileLastIndex*numberOfDocumentsPerDataFile + numberOfSamplesPerDataFileLast = 30423*10000 + 423
	numberOfDataFiles =	math.ceil(numberOfDocuments/numberOfDocumentsPerDataFile) #30424
	numberOfSamplesPerDataFileLast = numberOfDocuments%numberOfDocumentsPerDataFile	#423
elif(datasetName == 'OSCAR2201'):
	numberOfDocuments = 431992659	#number of documents	#https://huggingface.co/datasets/oscar-corpus/OSCAR-2201
	numberOfDataFiles =	math.ceil(numberOfDocuments/numberOfDocumentsPerDataFile)
	numberOfSamplesPerDataFileLast = numberOfDocuments%numberOfDocumentsPerDataFile
dataFileLastIndex = numberOfDataFiles-1

modelSaveNumberOfBatches = 1000	#resave model after x training batches


#transformer only;
customMaskTokenID = 4	#3
fractionOfMaskedTokens = 0.15

if(useEffectiveFullwordTokenizer):
	if(useFullwordTokenizer):
		vocabularySize = 2000000	#approx number of unique words in dataset
	else:
		vocabularySize = 240000	#approx number of unique words in english	#236736	#200000
else:
	vocabularySize = 30522	#default: 30522	#number of independent tokens identified by SBNLPpt_data.trainTokeniserSubwords

accuracyTopN = 1	#default: 1	#>= 1	#calculates batch accuracy based on top n dictionary predictions

specialTokens = ['<s>', '<pad>', '</s>', '<unk>', '<mask>']
specialTokenPadding = '<pad>'
specialTokenMask = '<mask>'

if(createOrderedDataset):
	orderedDatasetDocNumberSamples = 10
	orderedDatasetDocNumberTokens = orderedDatasetDocNumberSamples*sequenceMaxNumTokens
	orderedDatasetDocMinSizeCharacters = 10000	#prevents having to tokenise small document samples to count number of tokens
	orderedDatasetSplitDocumentsBySentences = False
