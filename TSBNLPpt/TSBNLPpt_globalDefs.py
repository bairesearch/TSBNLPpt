"""TSBNLPpt_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see TSBNLPpt_globalDefs.py

# Usage:
see TSBNLPpt_globalDefs.py

# Description:
TSBNLPpt globalDefs

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

#GPU parameters;
useGPU = True	#default: True

#prosody delimiters;
prosodyDelimitedData = False
if(prosodyDelimitedData):
	prosodyDelimitedType = "controlTokens"	#txtptc
	#prosodyDelimitedType = "repeatTokens"	#txtptr
	#prosodyDelimitedType = "uniqueTokens"	#txtptu
	
#masked/causal LM support;
useMaskedLM = False
useTrainWarmup = False	#orig: False (may be required for recursiveLayersNormaliseNumParameters)
if(useTrainWarmup):
	warmupSteps = 4000
	warmupLearningRateStart = 1e-7
	warmupLearningRateIncrement = 2.5e-8
	warmupLearningRateEnd = 1e-4	#==learningRate

#legacy implementation;
legacyDataloaderCode2 = False	#wo patch TSBNLPpt_dataTokeniser:getSampleEncodings to calculate labels = addLabelsPredictionMaskTokens (convert paddingTokenID [1] to labelPredictionMaskTokenID [-100])	#orig; True
sortDataFilesByName = True	#orig; False	#only stateTrainTokeniser and legacyDataloaderCode1 uses sortDataFilesByName (!legacyDataloaderCode1 assumes sortDataFilesByName=True)

#recursive algorithm selection:
useAlgorithmTransformer = True
useAlgorithmRNN = False
useAlgorithmSANI = False
useAlgorithmGIA = False

#state selection;
statePreprocessDataset = False	#only required once
stateTrainTokeniser = False	#only required once
stateTrainDataset = True
stateTestDataset = False	#requires reserveValidationSet

#training data selection;
trainStartEpoch = 0	#start epoch of training (if continuing a training regime set accordingly >0)	#if trainStartEpoch=0 and trainStartDataFile=0 will recreate model, if trainStartEpoch>0 or trainStartDataFile>0 will load existing model
trainNumberOfEpochs = 1	#default: 10	#number of epochs to train (for production typically train x epochs at a time)
trainStartDataFile = 0	#default: 0	#start data file to train (if continuing a training regime set accordingly >0)	#if trainStartEpoch=0 and trainStartDataFile=0 will recreate model, if trainStartEpoch>0 or trainStartDataFile>0 will load existing model
trainNumberOfDataFiles = 100	#15	#default: 100	#number of data files to train (for production typically train x dataFiles at a time)	#< datasetNumberOfDataFiles (30424) * trainSplitFraction
if(prosodyDelimitedData):
	testNumberOfDataFiles = 6
else:
	testNumberOfDataFiles = 10	#default: 10

debugCompareTokenMemoryBankPerformance = False
debugCreateOrderedDatasetFiles = False	#create data files comprising documents of sufficient length for createOrderedDataset
debugPrintPaths = False
debugPrintDataFileIndex = False
debugDoNotTrainModel = False
debugPrintLowHiddenSize = False
debugPrintMultipleModelAccuracy = False

#initialise (dependent vars);
recursiveLayers = False
memoryTraceBias = False
simulatedDendriticBranches = False
memoryTraceAtrophy = False

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
workingDrive = '/large/source/ANNpython/TSBNLPpt/'
ssdDrive = '/extssd/'


downloadCacheFolderName = 'cache'
if(prosodyDelimitedData):
	dataFolderName = 'dataLibrivoxBooksPreprocessed'
else:
	if(datasetName=='OSCAR1900'):
		dataFolderName = 'dataOSCAR1900preprocessed'
	elif(datasetName=='OSCAR2201'):
		dataFolderName = 'dataOSCAR2201preprocessed'
modelFolderName = 'model'
LRPfolderName = 'LRPdata/' + LRPdatabaseName
conceptExpertsFolderName = 'conceptExperts'
if(relativeFolderLocations):
	downloadCachePathName = downloadCacheFolderName
	dataPathName = dataFolderName
	modelPathName = modelFolderName
	LRPpathName = LRPfolderName
	conceptExpertsPathName = conceptExpertsFolderName
else:	
	downloadCachePathName = '/media/' + userName + dataDrive + downloadCacheFolderName
	dataPathName = '/media/' + userName + dataDrive + dataFolderName
	modelPathName = '/media/' + userName + workingDrive + modelFolderName
	LRPpathName = '/media/' + userName + workingDrive + LRPfolderName
	conceptExpertsPathName = '/media/' + userName + ssdDrive + conceptExpertsFolderName

if(debugCreateOrderedDatasetFiles):
	dataFolderNameLargeDocuments = 'dataLargeDocuments'
dataPreprocessedFileNameStart = "/text_"
if(prosodyDelimitedData):
	if(prosodyDelimitedType=="controlTokens"):
		dataPreprocessedFileNameEnd = ".txtptc"
	elif(prosodyDelimitedType=="repeatTokens"):
		dataPreprocessedFileNameEnd = ".txtptr"
	elif(prosodyDelimitedType=="uniqueTokens"):
		dataPreprocessedFileNameEnd = ".txtptu"
else:
	dataPreprocessedFileNameEnd = ".txt"
pytorchTensorFileExtension = ".pt"

if(prosodyDelimitedData):
	sequenceMaxNumTokensDefault = 256
else:
	sequenceMaxNumTokensDefault = 512

#initialise (dependent vars);
detectLocalConceptColumns = False
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

	#syntactic bias selection (part 1):
	recursiveLayers = False	#optional	#recursive transformer layers (reuse transformer blocks)
	memoryTraceBias = False	 #optional	#nncustom.Linear adjusts training/inference based on network prior activations
	simulatedDendriticBranches = False	#optional #nncustom.Linear simulates multiple independent fully connected weights per neuron

	#syntactic bias selection (part 2):
	localConceptColumnExperts = False	#apply MLP experts in each transformer block depending on sequence token local concept column
	localConceptColumnAttention = False	#performs standard (global) attention for each query token and only the tokens contained within the token's local concept column
	transformerPOSembeddings = False	#add POS embeddings to trainable embeddings
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
	transformerSuperblocksRecursive = False
	numberOfHiddenLayers = 1	#dynamically assigned: 1 (with recursiveLayers) or 6 (with !recursiveLayers, recursiveLayersOrigImplementation, or recursiveLayersEmulateOrigImplementation)
	transformerSuperblocksLayerNorm = False
	localConceptColumnExpertsApplyWithSharedMLPthenResidual = False

	#initialise (default vars);
	transformerBlockMLPlayer = True	#default: True	#apply all MLP layers
	transformerBlockMLPlayerLast = False	#default: False	#only apply last MLP layer (requires !transformerBlockMLPlayer)
	numberOfHiddenLayersDefault = 6
	numberOfAttentionHeads = 12	#default: 12	#numberOfAttentionHeadsDefault
	hiddenLayerSizeTransformer = 768	#default: 768 (can be overridden)
	positionEmbeddingType = "relative_key"	#default:"relative_key"	#orig (Nov 2022):"absolute"
	recursiveLayersOrigImplementation = False	#execute orig codebase with orig implementation so that archived models can be reloaded
	recursiveLayersEmulateOrigImplementation2 = False
	
	if(localConceptColumnExperts or localConceptColumnAttention):
		detectLocalConceptColumns = True
		localConceptColumnExpertsNoColumnID = -1
		localConceptColumnExpertsNoDictionaryNounID = 0
		debugDetectLocalConceptColumns = False
		localConceptColumnExpertsApplyToAllTokens = True	#requires higher processing power and GPU RAM (but same CPU RAM and SSD storage) #else only apply concept experts to concept featue (noun) tokens
		if(localConceptColumnExperts):
			localConceptColumnExpertsApplyWithSharedMLPthenResidual = False	#apply expert MLPs and shared MLP, and then apply residual to summed output
			if(localConceptColumnExpertsApplyWithSharedMLPthenResidual):
				localConceptColumnExpertsSharedMLPratio = 0.5	#default: 0.5 - may be increased depending on training performance (to reduce dependence on expert MLP)	#weight shared MLP over experts MLP
			else:
				localConceptColumnExpertsResidualRatio = 0.5	#default: 0.5 - may be increased depending on training performance (to reduce dependence on expert MLP)	#weight residual over experts MLP
			localConceptColumnExpertsApplyToAllTokens = False	#else restrict to nouns: only apply experts to concept features (nouns), not contextual features (non-nouns)	#reduces RAM usage
			localConceptColumnExpertsIntermediateSizeMax = 128	#default: 128	#ideal: 512 (sequenceMaxNumTokensDefault)	#GPU/CPU RAM dependent 	#requires chunking implementation
			localConceptColumnExpertsIntermediateSize = 64		#affects numerOfRecentlyAccessedExperts (GPU ram availability) and speed of processing
			approxNumNonNounsPerNoun = 10
			maxNumExpertsRequiredToProcessBatch = int(sequenceMaxNumTokensDefault / approxNumNonNounsPerNoun * 8)	#max number of experts required to process a batch (est approx =~400 = sequence length 512 / 5 concepts per sequence * 8 batch size)
			numerOfRecentlyAccessedExpertsMin = 1000	#default: 1000	#needs to be higher than the maxNumExpertsRequiredToProcessBatch*2, where the 2 is a factor used to account for memory management
			assert numerOfRecentlyAccessedExpertsMin > maxNumExpertsRequiredToProcessBatch*2
			numerOfRecentlyAccessedExperts = numerOfRecentlyAccessedExpertsMin * int(localConceptColumnExpertsIntermediateSizeMax/localConceptColumnExpertsIntermediateSize)
			#ratioOfGPUtoCPUramAvailableForExperts = 1.0	#clear all experts from cpu before processing batch for debug
			ratioOfGPUtoCPUramAvailableForExperts = maxNumExpertsRequiredToProcessBatch/numerOfRecentlyAccessedExperts
			print("localConceptColumnExperts parameters:")
			print("localConceptColumnExpertsApplyToAllTokens = ", localConceptColumnExpertsApplyToAllTokens)
			print("localConceptColumnExpertsIntermediateSizeMax = ", localConceptColumnExpertsIntermediateSizeMax)
			print("localConceptColumnExpertsIntermediateSize = ", localConceptColumnExpertsIntermediateSize)
			print("maxNumExpertsRequiredToProcessBatch = ", maxNumExpertsRequiredToProcessBatch)
			print("numerOfRecentlyAccessedExpertsMin = ", numerOfRecentlyAccessedExpertsMin)
			print("numerOfRecentlyAccessedExperts (num_experts_cpu) = ", numerOfRecentlyAccessedExperts)
			print("ratioOfGPUtoCPUramAvailableForExperts = ", ratioOfGPUtoCPUramAvailableForExperts)
			debugLocalConceptColumnExpertsFileIO = False

	if(transformerSuperblocks):
		transformerSuperblocksNumber = 2	#segregate nlp and logic layers
		transformerSuperblocksLayerNorm = True
		if(transformerSuperblocksLayerNorm):
			transformerSuperblocksLayerNormList = True	#separate norm function per layer
		transformerSuperblocksRecursive = True	#every super block is iterated multiple times
		if(transformerSuperblocksRecursive):
			transformerSuperblocksRecursiveNumberIterations = 2	#configure
			transformerSmall = False	#reduce GPU RAM
			if(transformerSmall):
				hiddenLayerSizeTransformer = 256	#384
				numberOfHiddenLayersDefault = 2
	if(recursiveLayers):
		recursiveLayersEvalOverride = False	#currently only supported by recursiveLayersOrigImplementation; if !recursiveLayersOrigImplementation then just change numberOfHiddenLayersDefault:recursiveLayersNumberIterations
		if(recursiveLayersEvalOverride):
			recursiveLayersNumberIterationsEvalOverride = 6
		recursiveLayersEmulateOrigImplementation = True	#execute new codebase but emulate implementation #1 (Nov 2022) so that archived models can be reloaded	#depreciated (use recursiveLayersOrigImplementation instead)
		recursiveLayersEmulateOrigImplementation2 = True	#emulate implementation #2 (May 2023); always iterate over transformerSuperblocks
		if(recursiveLayersOrigImplementation):
			numberOfHiddenLayers = numberOfHiddenLayersDefault
			recursiveLayersNumberIterations = 1	#numberOfHiddenLayers is interpreted as number of repetitive (duplicate) layers in layerList
		else:
			if(recursiveLayersEmulateOrigImplementation):
				numberOfHiddenLayers = numberOfHiddenLayersDefault
				recursiveLayersNumberIterations = numberOfHiddenLayers	#numberOfHiddenLayers is interpreted as recursiveLayersNumberIterations
			else:
				numberOfHiddenLayers = 1
				recursiveLayersNumberIterations = numberOfHiddenLayersDefault
			if(recursiveLayersEvalOverride):
				recursiveLayersNumberIterations = recursiveLayersNumberIterationsEvalOverride
	else:
		numberOfHiddenLayers = numberOfHiddenLayersDefault
				
	useTransformerRecursive = False	
	if(recursiveLayers or transformerSuperblocks):
		useTransformerRecursive = True
	
	#if(recursiveLayers or transformerSuperblocksRecursive):
	#	numberOfAttentionHeads = 1	#prevents recursion across different attention heads, nullifying precise recursion
			
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
		positionEmbeddingType = "relative_time"	#calculates relative time between layer tokens
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

#recent tokenizer options;
useSubwordTokenizerFast = False
if(detectLocalConceptColumns):
	useSubwordTokenizerFast = True	#required to obtain offsets during tokenization
usePretrainedRobertaTokenizer = False	#incomplete #do not use pretrained tokenizer

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
	#FUTURE: require update of TSBNLPpt_data to ensure that continuous/contiguous textual input (for inference) is provided
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
	if(recursiveLayers or memoryTraceBias or simulatedDendriticBranches or GIAsemanticRelationVectorSpaces or tokenMemoryBank or transformerAttentionHeadPermutations or transformerPOSembeddings or transformerSuperblocks):
		useSyntacticBiases = True
	else:
		useSyntacticBiases = False
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
if(prosodyDelimitedData):
	trainSplitFraction = 0.95
else:
	trainSplitFraction = 0.9	#90% train data, 10% test data

if(useAlgorithmTransformer):
	batchSize = 8	#default: 8	#8 and 16 train at approx same rate (16 uses more GPU ram)	#depends on GPU RAM	#with 12GB GPU RAM, batchSize max = 16
	learningRate = 1e-4	#0.0001
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

if(useSmallDatasetDebug):
	batchSize = 1
	
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
if(prosodyDelimitedData):
	datasetNumberOfDocuments = 105*10000
else:
	if(datasetName == 'OSCAR1900'):
		datasetNumberOfDocuments = 304230423	#orig: dataFileLastIndex*numberOfDocumentsPerDataFile + datasetNumberOfSamplesPerDataFileLast = 30423*10000 + 423
	elif(datasetName == 'OSCAR2201'):
		datasetNumberOfDocuments = 431992659	#number of documents	#https://huggingface.co/datasets/oscar-corpus/OSCAR-2201
datasetNumberOfDataFiles =	math.ceil(datasetNumberOfDocuments/numberOfDocumentsPerDataFile) #30424
datasetNumberOfSamplesPerDataFileLast = datasetNumberOfDocuments%numberOfDocumentsPerDataFile	#423
dataFileLastIndex = datasetNumberOfDataFiles-1

modelSaveNumberOfBatches = 1000	#resave model after x training batches


#transformer only;
if(useMaskedLM):
	customMaskTokenID = 4	#3
	fractionOfMaskedTokens = 0.15
if(not legacyDataloaderCode2):
	paddingTokenID = 1
	labelPredictionMaskTokenID = -100	#https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaForCausalLM.forward.labels

#Warning: if change vocabularySize, require reexecution of python TSBNLPpt_GIAdefinePOSwordLists.py (LRPdata/NLTK/wordlistVector*.txt)
if(useEffectiveFullwordTokenizer):
	if(useFullwordTokenizer):
		vocabularySize = 2000000	#approx number of unique words in dataset
	else:
		vocabularySize = 240000		#approx number of unique words in english	#236736	#requirement: must be >= size of NLTK wordlistAll.txt
else:
	vocabularySize = 30522	#default: 30522	#number of independent tokens identified by TSBNLPpt_data.trainTokeniserSubwords

accuracyTopN = 1	#default: 1	#>= 1	#calculates batch accuracy based on top n dictionary predictions

specialTokens = ['<s>', '<pad>', '</s>', '<unk>', '<mask>']
specialTokenPadding = '<pad>'
if(useMaskedLM):
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
	#initialise (dependent vars);
	sharedLayerWeights = False
	sharedLayerWeightsAttention = False
	sharedLayerWeightsMLP = False
	sharedLayerWeightsSelfAttention = False
	sharedLayerWeightsSelfOutput = False
	sharedLayerWeightsIntermediate = False
	sharedLayerWeightsOutput = False
	if(not usePretrainedModelDebug):
		if(recursiveLayers):
			sharedLayerWeights = False	#orig recursiveLayers implementation
			if(sharedLayerWeights):
				sharedLayerWeightsAttention = True	#default: true
				sharedLayerWeightsMLP = True	#default: true
				if(sharedLayerWeightsAttention):
					sharedLayerWeightsSelfAttention = True	#default: true
					sharedLayerWeightsSelfOutput = True	#default: true
				if(sharedLayerWeightsMLP):
					sharedLayerWeightsIntermediate = True	#default: true
					sharedLayerWeightsOutput = True	#default: true
			#shared layer configuration;
			sharedLayerWeightsMLPonly = False
			sharedLayerWeightsWithOutputs = False	
			sharedLayerWeightsWithoutOutputs = False
			if(sharedLayerWeights):
				if(sharedLayerWeightsSelfAttention and sharedLayerWeightsIntermediate and sharedLayerWeightsSelfOutput and sharedLayerWeightsOutput):
					sharedLayerWeightsWithOutputs = True
				elif(sharedLayerWeightsSelfAttention and sharedLayerWeightsIntermediate and not sharedLayerWeightsSelfOutput and not sharedLayerWeightsOutput):
					sharedLayerWeightsWithoutOutputs = True
				if(sharedLayerWeightsMLP and not sharedLayerWeightsAttention):
					sharedLayerWeightsMLPonly = True
			#normalisation;
			recursiveLayersNormaliseNumParameters = False	#default: False	#optional	#if use recursiveLayers normalise/equalise num of parameters with respect to !recursiveLayers	#legacy
			if(recursiveLayersNormaliseNumParameters):
				recursiveLayersNormaliseNumParametersAttentionHeads = True	#default: true
				recursiveLayersNormaliseNumParametersIntermediate = True	#default: true	#normalise intermediateSize parameters also
				recursiveLayersNormaliseNumParametersIntermediateOnly = False	#only normalise intermediary MLP layer	#requires recursiveLayersNormaliseNumParametersIntermediate
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
				if(not transformerBlockMLPlayer):
					hiddenLayerSizeMultiplier = 2.2	#model size = 255MB
				elif(recursiveLayersNormaliseNumParametersIntermediateOnly):
					if(sharedLayerWeightsMLPonly):
						hiddenLayerSizeMultiplier = 1
						intermediateLayerSizeMultiplier = 6	#model size = 257MB	#hiddenLayerSize 768, intermediateSize 18432
					else:
						hiddenLayerSizeMultiplier = 1
						intermediateLayerSizeMultiplier = 8	#model size = 248MB	#hiddenLayerSize 768, intermediateSize 24576
				elif(sharedLayerWeightsWithoutOutputs):
					if(recursiveLayersNormaliseNumParametersIntermediate):
						hiddenLayerSizeMultiplier = (4/3)	#model size = 273MB
						intermediateLayerSizeMultiplier = hiddenLayerSizeMultiplier
					else:
						hiddenLayerSizeMultiplier = 1.5	#model size = ~255MB
				else:	#elif((not sharedLayerWeights) or sharedLayerWeightsWithOutputs):
					if(recursiveLayersNormaliseNumParametersIntermediate):
						hiddenLayerSizeMultiplier = (7/4)	#model size = 249MB		#hiddenLayerSize 1344, intermediateSize 5376
						intermediateLayerSizeMultiplier = hiddenLayerSizeMultiplier
					else:
						hiddenLayerSizeMultiplier = 2	#model size = ~255-263MB	#hiddenLayerSize 1536, numberOfAttentionHeads 24
				attentionHeadMultiplier = hiddenLayerSizeMultiplier
				hiddenLayerSize = round(hiddenLayerSize*hiddenLayerSizeMultiplier)
				if(recursiveLayersNormaliseNumParametersAttentionHeads):
					numberOfAttentionHeads = round(numberOfAttentionHeads*attentionHeadMultiplier)	#or: round(numberOfAttentionHeads)
				if(recursiveLayersNormaliseNumParametersIntermediate):
					intermediateSize = round(intermediateSize*intermediateLayerSizeMultiplier)
				print("hiddenLayerSize = ", hiddenLayerSize)
				print("numberOfAttentionHeads = ", numberOfAttentionHeads)
				print("intermediateSize = ", intermediateSize)
			else:
				if(sharedLayerWeights):
					if(sharedLayerWeightsWithOutputs):
						pass	#model size = ~120MB
					elif(sharedLayerWeightsWithoutOutputs):
						pass	#model size = 176.7MB
					else:
						pass	#model size = unknown
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
	#syntactic bias selection (part 1):
	recursiveLayers = True
	
	hiddenLayerSize = 1024	#65536	#2^16 - large hidden size is required for recursive RNN as parameters are shared across a) sequence length and b) number of layers
	if(TSBNLPpt_RNNmodel.applyIOconversionLayers):
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
	#syntactic bias selection (part 1):
	recursiveLayers = True
	
	hiddenLayerSize = 1024	#1024	#8192	#1024	#depends on GPU memory	#2^16 = 65536 - large hidden size is required for recursive SANI as parameters are shared across a) sequence length and b) number of layers
	if(TSBNLPpt_SANImodel.applyIOconversionLayers):
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
