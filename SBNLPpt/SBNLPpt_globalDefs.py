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

#recursive algorithm selection:
useAlgorithmTransformer = False
useAlgorithmRNN = False
useAlgorithmSANI = False
useAlgorithmGIA = True	#useAlgorithmGIAsemanticRelationVectorSpace

sortDataFilesByName = True	#orig; False

recursiveLayers = True	#optional
memoryTraceBias = False	 #optional	#nncustom.Linear adjusts training/inference based on network prior activations
simulatedDendriticBranches = False	#optional #nncustom.Linear simulates multiple independent fully connected weights per neuron

statePreprocessDataset = False	#only required once
stateTrainTokenizer = False	#only required once
stateTrainDataset = True
stateTestDataset = False	#requires reserveValidationSet

trainStartEpoch = 0	#start epoch of training (if continuing a training regime set accordingly >0)	#if trainStartEpoch=0 and trainStartDataFile=0 will recreate model, if trainStartEpoch>0 or trainStartDataFile>0 will load existing model
trainNumberOfEpochs = 1	#default: 10	#number of epochs to train (for production typically train x epochs at a time)
trainStartDataFile = 0	#default: 0	#start data file to train (if continuing a training regime set accordingly >0)	#if trainStartEpoch=0 and trainStartDataFile=0 will recreate model, if trainStartEpoch>0 or trainStartDataFile>0 will load existing model
trainNumberOfDataFiles = 100	#2	#100	#default: -1 (all)	#number of data files to train (for production typically train x dataFiles at a time)	#< numberOfDataFiles (30424) * trainSplitFraction
testNumberOfDataFiles = 10	#2	#10	#default: -1 (all)

relativeFolderLocations = False
userName = 'user'	#default: user
#storage location vars (requires 4TB harddrive);
if(relativeFolderLocations):
	downloadCacheFolder = 'cache'
	dataFolder = 'data'
	modelFolderName = 'model'
else:
	downloadCacheFolder = '/media/' + userName + '/datasets/cache'
	dataFolder = '/media/' + userName + '/datasets/data'
	modelFolderName = '/media/' + userName + '/large/source/ANNpython/SBNLPpt/model'	#modelTemp, model
	
useMultipleModels = False
useTrainedTokenizer = True
useFullwordTokenizer = False
useFullwordTokenizerClass = True
debugDoNotTrainModel = False
if(useAlgorithmGIA):
	debugPrintRelationExtractionProgress = False
	debugUseSmallNumberOfModels = True
	debugDoNotTrainModel = False
	
	useMultipleModels = True
	useFullwordTokenizer = False	#optional	#tokenizer only identifies whole words
	if(useFullwordTokenizer):
		useFullwordTokenizerNLTK = False	#optional	#else use DistilBertTokenizer.basic_tokenizer.tokenize
		useFullwordTokenizerPretrained = False	#optional	#required for latest version of transformers library
		if(useFullwordTokenizerPretrained):
			useFullwordTokenizerPretrainedAuto = True	#optional
		else:
			useFullwordTokenizerFast = False	#optional
			if(not useFullwordTokenizerFast):
				useFullwordTokenizerClass = False
			useTrainedTokenizer = False
			tokensVocabPathName = modelFolderName + "/" + "vocab-fullword.json"
			tokensSpecialPathName = modelFolderName + "/" + "special_tokens-fullword.json" 
	useIndependentReverseRelationsModels = False	#else take input linear layer as forward embeddings and output linear layer [inversed] as reverse embeddings

if(recursiveLayers or memoryTraceBias or simulatedDendriticBranches):
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
		trainTokenizerNumberOfFilesToUseSmall = 100	#default: 100	#100: 2 hours
	else:
		trainTokenizerNumberOfFilesToUseSmall = 100	#default 1000	#100: 15 min, 1000: 3.75 hours

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
	batchSize = 1	#useAlgorithmGIAsemanticRelationVectorSpace batchSize is dynamic (>batchSize)
	learningRate = 1e-4
	
if(simulatedDendriticBranches):
	batchSize = batchSize//4	#requires more GPU RAM (reduced batchSize)
if(memoryTraceBias):
	batchSize = 1	#CHECKTHIS - memoryTraceBias algorithm requires continuous/contiguous textual input (for inference)
	
if(useSmallBatchSizeDebug):
	batchSize = 1	#use small batch size to enable simultaneous execution (GPU ram limited) 
	
numberOfSamplesPerDataFile = 10000
numberOfSamplesPerDataFileLast = 423
dataFileLastSampleIndex = 30423


modelSaveNumberOfBatches = 1000	#resave model after x training batches


sequenceMaxNumTokens = 512	#window length (transformer/RNN/SANI)

#transformer only;
customMaskTokenID = 4	#3
fractionOfMaskedTokens = 0.15

if(useAlgorithmGIA):
	if(useFullwordTokenizer):
		vocabularySize = 2000000	#approx number of unique words in dataset
	else:
		vocabularySize = 200000	#approx number of unique words in english
else:
	vocabularySize = 30522	#default: 30522	#number of independent tokens identified by SBNLPpt_data.trainTokenizerSubwords

accuracyTopN = 1	#default: 1	#>= 1	#calculates batch accuracy based on top n dictionary predictions

useLovelyTensors = True
if(useLovelyTensors):
	import lovely_tensors as lt
	lt.monkey_patch()


