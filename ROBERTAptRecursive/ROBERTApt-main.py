"""ROBERTApt-main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
conda create -n pytorchsenv
source activate pytorchsenv
conda install python=3.12
pip install networkx
pip install matplotlib
pip install yattag
pip install torch
pip install torch_geometric
pip install nltk 
pip install spacy
pip install benepar
pip install datasets
pip install transfomers
pip install lovely-tensors
pip install torchmetrics
pip install pynvml
python3 -m spacy download en_core_web_md

# Usage:
source activate pytorchsenv
python ROBERTApt-main.py

# Description:
ROBERTApt main - trains a RoBERTa transformer with a number of syntactic inductive biases:
- recursiveLayers
	- Roberta number of layers = 6 supports approximately 2^6 words per sentence (contextual window = 512 tokens)

See RobertaForMaskedLM tutorial; 
	https://huggingface.co/blog/how-to-train
	https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6

"""

from modeling_roberta_recursiveLayers import recursiveLayers
from modeling_roberta_recursiveLayers import centralSequencePrediction
sequenceStartToken = '<s>'	#tokenizer.bos_token
sequenceEndToken = '</s>'	#tokenizer.eos_token
if(centralSequencePrediction):
	import nltk
	from nltk.tokenize import sent_tokenize
	nltk.download('punkt')
	from modeling_roberta_recursiveLayers import maxConclusionLength
	from modeling_roberta_recursiveLayers import maxIntroLength
	from modeling_roberta_recursiveLayers import maxCentralLength
	from modeling_roberta_recursiveLayers import maxIntroCentralLength
	centralSequencePredictionConclusionEndToken = '<conclusion_end>'
	centralSequencePredictionIntroStartToken = '<intro_start>'
	
usePretrainedRobertaTokenizer = False	#incomplete #do not use pretrained tokenizer

prosodyDelimitedData = False
if(prosodyDelimitedData):
	#prosodyDelimitedType = "txtDebug"	#txt	#not currently used
	prosodyDelimitedType = "controlTokens"	#txtptc
	#prosodyDelimitedType = "repeatTokens"	#txtptr	#not currently used
	#prosodyDelimitedType = "uniqueTokens"	#txtptu

useMaskedLM = False
useTrainWarmup = False	#orig: False (may be required for recursiveLayersNormaliseNumParameters)
if(useTrainWarmup):
	warmupSteps = 4000
	warmupLearningRateStart = 1e-7
	warmupLearningRateIncrement = 2.5e-8
	warmupLearningRateEnd = 1e-4	#==learningRate

relativeFolderLocations = False

legacyDataloaderCode1 = False
legacyDataloaderCode2 = False	#wo patch SBNLPpt_dataTokeniser:getSampleEncodings to calculate labels = addLabelsPredictionMaskTokens (convert paddingTokenID [1] to labelPredictionMaskTokenID [-100])
sortDataFilesByName = True	#orig; False	#only stateTrainTokeniser and legacyDataloaderCode1 uses sortDataFilesByName (!legacyDataloaderCode1 assumes sortDataFilesByName=True)

#user config vars:

useSmallDatasetDebug = False
useSingleHiddenLayerDebug = False
usePretrainedModelDebug = False	#executes stateTestDataset only
useSmallBatchSizeDebug = False

useSmallTokenizerTrainNumberOfFiles = True	#used during rapid testing only (FUTURE: assign est 80 hours to perform full tokenisation train)

statePreprocessDataset = False	#only required once
if(prosodyDelimitedData):
	stateTrainTokenizer = True	#only required once
else:
	stateTrainTokenizer = False	#only required once
stateTrainDataset = True
stateTestDataset = True	#requires reserveValidationSet

if(recursiveLayers):
	from modeling_roberta_recursiveLayers import sharedLayerWeights
	from modeling_roberta_recursiveLayers import sharedLayerWeightsMLPonly
	from modeling_roberta_recursiveLayers import sharedLayerWeightsWithOutputs
	from modeling_roberta_recursiveLayers import sharedLayerWeightsWithoutOutputs
	from modeling_roberta_recursiveLayers import transformerBlockMLPlayer
	recursiveLayersNormaliseNumParameters = False	#optional	#if use recursiveLayers normalise/equalise num of parameters with respect to !recursiveLayers
	if(recursiveLayersNormaliseNumParameters):
		recursiveLayersNormaliseNumParametersAttentionHeads = True	#default: true
		recursiveLayersNormaliseNumParametersIntermediate = True	#default: true	#normalise intermediateSize parameters also	
		recursiveLayersNormaliseNumParametersIntermediateOnly = False	#default: false	#only normalise intermediary MLP layer	#requires recursiveLayersNormaliseNumParametersIntermediate
else:
	recursiveLayersNormaliseNumParameters = False	#mandatory

trainStartEpoch = 0	#start epoch of training (if continuing a training regime set accordingly >0)	#if trainStartEpoch=0 and trainStartDataFile=0 will recreate model, if trainStartEpoch>0 or trainStartDataFile>0 will load existing model
trainNumberOfEpochs = 1	#default: 10	#number of epochs to train (for production typically train x epochs at a time)
trainStartDataFile = 0	#default: 0	#start data file to train (if continuing a training regime set accordingly >0)	#if trainStartEpoch=0 and trainStartDataFile=0 will recreate model, if trainStartEpoch>0 or trainStartDataFile>0 will load existing model
trainNumberOfDataFiles = 100	#default: 100	#number of data files to train (for production typically train x dataFiles at a time)	#< numberOfDataFiles (30424) * trainSplitFraction
if(prosodyDelimitedData):
	testNumberOfDataFiles = 6
else:
	testNumberOfDataFiles = 10	#default: 10

if(not usePretrainedModelDebug):

	positionEmbeddingType = "relative_key"	#default:"relative_key"	#orig (Nov 2022):"absolute"
	
	if(useSingleHiddenLayerDebug):
		numberOfHiddenLayers = 1
	else:
		numberOfHiddenLayers = 6	#default: 6

	vocabularySize = 30522	#default: 30522
	hiddenLayerSize = 768	#default: 768
	numberOfAttentionHeads = 12	#default: 12
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
		
reserveValidationSet = True	#reserves a fraction of the data for validation
if(prosodyDelimitedData):
	trainSplitFraction = 0.95
else:
	trainSplitFraction = 0.9	#90% train data, 10% test data

if(recursiveLayersNormaliseNumParameters):
	batchSize = 8	#recursiveLayersNormaliseNumParameters uses ~16x more GPU RAM than !recursiveLayersNormaliseNumParameters, and ~2x more GPU RAM than !recursiveLayers
	learningRate = 1e-4
else:
	batchSize = 8	#8  #default: 16	#8 and 16 train at approx same rate (16 uses more GPU ram)	#depends on GPU RAM	#with 12GB GPU RAM, batchSize max = 16
	learningRate = 1e-4
		
if(useSmallBatchSizeDebug):
	batchSize = 1	#use small batch size to enable simultaneous execution (GPU ram limited) 

numberOfSamplesPerDataFile = 10000
numberOfSamplesPerDataFileLast = 423
if(prosodyDelimitedData):
	dataFileLastSampleIndex = 105
else:
	dataFileLastSampleIndex = 30423
datasetNumberOfDataFiles = dataFileLastSampleIndex+1

dataPreprocessedFileNameStart = "/text_"
if(prosodyDelimitedData):
	if(prosodyDelimitedType=="controlTokens"):
		dataPreprocessedFileNameEnd = ".txtptc"
	elif(prosodyDelimitedType=="repeatTokens"):
		dataPreprocessedFileNameEnd = ".txtptr"
	elif(prosodyDelimitedType=="uniqueTokens"):
		dataPreprocessedFileNameEnd = ".txtptu"
	elif(prosodyDelimitedType=="txtDebug"):
		dataPreprocessedFileNameEnd = ".txt"
else:
	dataPreprocessedFileNameEnd = ".txt"

#storage location vars (requires 4TB harddrive);
downloadCacheFolderName = 'cache'
if(relativeFolderLocations):
	dataFolderName = 'data'
	downloadCacheFolder = downloadCacheFolderName
	dataFolder = dataFolderName
else:
	if(prosodyDelimitedData):
		dataFolderName = 'dataLibrivoxBooksPreprocessed'
	else:
		dataFolderName = 'dataOSCAR1900preprocessed'
	downloadCacheFolder = '/media/user/datasets/' + downloadCacheFolderName
	dataFolder = '/media/user/datasets/' + dataFolderName
modelFolderName = 'model'

modelSaveNumberOfBatches = 1000	#resave model after x training batches

accuracyTopN = 1	#default: 1	#>= 1	#calculates batch accuracy based on top n dictionary predictions

from datasets import load_dataset
from tqdm.auto import tqdm
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
import os
from transformers import RobertaTokenizer
import torch
from transformers import RobertaConfig
if(useMaskedLM):
	if(recursiveLayers):
		from modeling_roberta_recursiveLayers import RobertaForMaskedLM as RobertaLM
	else:
		from transformers import RobertaForMaskedLM as RobertaLM
else:
	if(recursiveLayers):
		from modeling_roberta_recursiveLayers import RobertaForCausalLM as RobertaLM
	else:
		from transformers import RobertaForCausalLM as RobertaLM
from transformers import AdamW
from transformers import pipeline
from torchsummary import summary
import math 

#torch.set_printoptions(threshold=10_000)
torch.set_printoptions(profile="full")

#store models to large datasets partition cache folder (not required)
#os.environ['TRANSFORMERS_CACHE'] = '/media/user/datasets/models/'	#select partition with 3TB+ disk space

if(prosodyDelimitedData):
	sequenceMaxNumTokens = 256
else:
	sequenceMaxNumTokens = 512
if(useMaskedLM):
	customMaskTokenID = 4	#3
	fractionOfMaskedTokens = 0.15
if(not legacyDataloaderCode2):
	paddingTokenID = 1
	labelPredictionMaskTokenID = -100	#https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaForCausalLM.forward.labels

def downloadDataset():
	if(useSmallDatasetDebug):
		dataset = load_dataset('nthngdy/oscar-small', 'unshuffled_original_en', cache_dir=downloadCacheFolder)	#unshuffled_deduplicated_en
	else:
		dataset = load_dataset('oscar', 'unshuffled_deduplicated_en', cache_dir=downloadCacheFolder)
	
	return dataset

def preprocessDataset(dataset):
	textData = []
	fileCount = 0
	for sample in tqdm(dataset['train']):
		sample = sample['text'].replace('\n', '')
		textData.append(sample)
		if(len(textData) == numberOfSamplesPerDataFile):
			writeDataFile(fileCount, textData)
			textData = []
			fileCount += 1
	writeDataFile(fileCount, textData)	#remaining data file will be < numberOfSamplesPerDataFile
	
def writeDataFile(fileCount, textData):
	fileName = dataFolder + "/text_" + str(fileCount) + ".txt"
	with open(fileName, 'w', encoding='utf-8') as fp:
		fp.write('\n'.join(textData))

def trainTokenizer(paths):
	if(useSmallTokenizerTrainNumberOfFiles):
		trainTokenizerNumberOfFilesToUse = 100	#100	#default 1000	#100: 15 min, 1000: 3.75 hours
	else:
		trainTokenizerNumberOfFilesToUse = len(paths)

	special_tokens=[sequenceStartToken, '<pad>', sequenceEndToken, '<unk>', '<mask>']
	if(centralSequencePrediction):
		special_tokens = special_tokens + [centralSequencePredictionConclusionEndToken, centralSequencePredictionIntroStartToken]
		
	tokenizer = ByteLevelBPETokenizer()

	print("paths = ", paths)
	
	tokenizer.train(files=paths[:trainTokenizerNumberOfFilesToUse], vocab_size=vocabularySize, min_frequency=2, special_tokens=special_tokens)
	
	os.mkdir(modelFolderName)

	tokenizer.save_model(modelFolderName)
		
	return tokenizer

		
def loadTokenizer():
	if(usePretrainedRobertaTokenizer):
		tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
	else:
		tokenizer = RobertaTokenizer.from_pretrained(modelFolderName, max_len=sequenceMaxNumTokens)

	return tokenizer

def addLabelsPredictionMaskTokens(input_ids):
	mask_arr = (input_ids == paddingTokenID)
	mask_arr = mask_arr*(labelPredictionMaskTokenID-paddingTokenID)
	labels = input_ids + mask_arr
	#print("labels = ", labels)
	return labels

def addMaskTokens(input_ids):
	rand = torch.rand(input_ids.shape)
	mask_arr = (rand < fractionOfMaskedTokens) * (input_ids > 2)	#or * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
	for i in range(input_ids.shape[0]):
		selection = torch.flatten(mask_arr[i].nonzero()).tolist()
		input_ids[i, selection] = customMaskTokenID
	return input_ids

if(legacyDataloaderCode1):
	def dataFileIndexListContainsLastFile(dataFileIndexList, paths):
		containsDataFileLastSample = False
		for dataFileIndex in dataFileIndexList:
			path = paths[dataFileIndex]
			if(str(dataFileLastSampleIndex) in path):
				containsDataFileLastSample = True
		return containsDataFileLastSample
else:
	def getNumberOfDocumentsHDD(numberOfDocumentsEst, dataFileIndexList):
		containsDataFileLastDocument = dataFileIndexListContainsLastDocument(dataFileIndexList)
		numberOfDocuments = len(dataFileIndexList)*numberOfSamplesPerDataFile
		if(containsDataFileLastDocument):
			numberOfDocuments = numberOfDocuments-numberOfSamplesPerDataFile + datasetNumberOfSamplesPerDataFileLast
		return numberOfDocuments

	def dataFileIndexListContainsLastDocument(dataFileIndexList):
		containsDataFileLastDocument = False
		for dataFileIndex in dataFileIndexList:
			if(str(dataFileIndex) == dataFileLastSampleIndex):
				containsDataFileLastDocument = True
		return containsDataFileLastDocument
if(not legacyDataloaderCode1):
	def generateDataFileName(fileIndex):
		fileName = dataFolder + dataPreprocessedFileNameStart + str(fileIndex) + dataPreprocessedFileNameEnd
		return fileName
	
class DatasetHDD(torch.utils.data.Dataset):
	def __init__(self, numberOfDocumentsEst, dataFileIndexList, paths):
		self.dataFileIndexList = dataFileIndexList
		self.paths = paths
		self.encodings = None
		if(legacyDataloaderCode1):
			self.containsDataFileLastSample = dataFileIndexListContainsLastFile(dataFileIndexList, paths)
		else:
			self.numberOfDocuments = getNumberOfDocumentsHDD(numberOfDocumentsEst, dataFileIndexList)

	def __len__(self):
		if(legacyDataloaderCode1):
			numberOfSamples = len(self.dataFileIndexList)*numberOfSamplesPerDataFile
			if(self.containsDataFileLastSample):
				numberOfSamples = numberOfSamples-numberOfSamplesPerDataFile + numberOfSamplesPerDataFileLast
			return numberOfSamples
		else:
			return self.numberOfDocuments

	def reorderSampleStartConclusionSentence(self, tokenizer, sample, lines):
		#print("sample = ", sample)
		tokensNewList = []
		attentionMaskNewList = []
		for lineIndex, line in enumerate(lines):
			tokens = tokenizer.convert_ids_to_tokens(sample.input_ids[lineIndex])
			token_ids = sample.input_ids[lineIndex]
			attention_mask = sample.attention_mask[lineIndex]
			token_offsets = []
			current_offset = 0
			i = 0
			#print("len(line) = ", len(line))
			for token, token_id in zip(tokens, token_ids):
				tokenFormatted =  token.replace('\u0120', '')	#remove start/end word 'G' characters from token
				tokenFormatted = tokenFormatted.lower()
				#print("\ttokenFormatted = ", tokenFormatted)
				start_pos = line.lower().find(tokenFormatted, current_offset)
				if(start_pos != -1):
					#<s>, </s>, and many other tokens produced by the tokenizer are not found in lines (cannot rely on a complete token_offsets);
					end_pos = start_pos + len(tokenFormatted)
					token_offsets.append((token, token_id, start_pos, end_pos))
					current_offset = end_pos
				#else:
				#	print("\tline.lower().find fail: i = ", i)
				i += 1
			#conclusionSentencePosEnd = end_pos

			introFirstTokenIndex = 1	#skip <s> token
			conclusionFirstTokenIndex = None
			conclusionLastTokenIndex = None
			
			sentences = nltk.sent_tokenize(line)
			current_offset = 0
			sentencePos = 0
			for sentenceIndex, sentence in enumerate(sentences):
				#print("sentence = ", sentence)
				start_pos = line.find(sentence, current_offset)
				end_pos = start_pos + len(sentence)
				current_offset = end_pos
				sentencePosStart = start_pos
				sentencePosEnd = end_pos-1

				for tokenIndex, tokenTuple in enumerate(token_offsets):
					start_pos = tokenTuple[2]
					end_pos = tokenTuple[3]
					#print("\tstart_pos = ", start_pos)
					#print("\tend_pos = ", end_pos)
					if(start_pos == sentencePosStart):
						conclusionFirstTokenIndex = tokenIndex
						conclusionSentencePosStart = sentencePosStart
					if(start_pos == sentencePosEnd):
						conclusionLastTokenIndex = tokenIndex
						conclusionSentencePosEnd = sentencePosEnd
				if(conclusionLastTokenIndex == None):
					conclusionLastTokenIndex = tokenIndex-1	#last token in context window, skip </s> token
							
			tokenidsConclusion = torch.concat((torch.tensor([tokenizer.convert_tokens_to_ids(sequenceStartToken)]), 
				token_ids[conclusionFirstTokenIndex:conclusionLastTokenIndex+1], 
				torch.tensor([tokenizer.convert_tokens_to_ids(centralSequencePredictionConclusionEndToken)])), dim=0)
			tokenidsConclusion = self.resizeSubtensor(tokenidsConclusion, maxConclusionLength, tokenizer.pad_token_id)
			tokenidsIntroCentral = torch.concat((torch.tensor([tokenizer.convert_tokens_to_ids(centralSequencePredictionIntroStartToken)]), 
				token_ids[introFirstTokenIndex:conclusionFirstTokenIndex], 
				torch.tensor([tokenizer.convert_tokens_to_ids(sequenceEndToken)])), dim=0)
			tokenidsIntroCentral = self.resizeSubtensor(tokenidsIntroCentral, maxIntroCentralLength, tokenizer.pad_token_id)
			
			attentionMaskConclusion = torch.concat((torch.ones(1), attention_mask[conclusionFirstTokenIndex:conclusionLastTokenIndex+1], torch.ones(1)), dim=0)
			attentionMaskConclusion = self.resizeSubtensor(attentionMaskConclusion, maxConclusionLength, 0)
			attentionMaskIntroCentral = torch.concat((torch.ones(1), attention_mask[introFirstTokenIndex:conclusionFirstTokenIndex], torch.ones(1)), dim=0)
			attentionMaskIntroCentral = self.resizeSubtensor(attentionMaskIntroCentral, maxIntroCentralLength, 0)

			inputidsNew = torch.cat((tokenidsIntroCentral, tokenidsConclusion), dim=0)
			attentionMaskNew = torch.cat((attentionMaskIntroCentral, attentionMaskConclusion), dim=0)
			
			#print("inputidsNew.shape = ", inputidsNew.shape)
			#print("attentionMaskNew.shape = ", attentionMaskNew.shape)
			
			tokensNewList.append(inputidsNew)
			attentionMaskNewList.append(attentionMaskNew)

		inputidsNew = torch.stack(tokensNewList, dim=0)
		attentionMaskNew = torch.stack(attentionMaskNewList, dim=0)
		sample.input_ids = inputidsNew
		sample.attention_mask = attentionMaskNew
		
		
	def resizeSubtensor(self, tokens, maxLength, padTokenID):
		if(tokens.shape[0] > maxLength):
			tokens = tokens[0:maxLength]
		if(tokens.shape[0] < maxLength):
			paddingLength = maxLength-tokens.shape[0]
			tokensPadding = torch.full((paddingLength,), padTokenID, dtype=torch.long)
			tokens = torch.cat((tokens, tokensPadding), dim=0)
		return tokens
						
	def __getitem__(self, i):
	
		loadNextDataFile = False
		sampleIndex = i // numberOfSamplesPerDataFile
		itemIndexInSample = i % numberOfSamplesPerDataFile
		if(itemIndexInSample == 0):
			loadNextDataFile = True	
		dataFileIndex = self.dataFileIndexList[sampleIndex]
					
		if(loadNextDataFile):
			
			if(legacyDataloaderCode1):
				path = self.paths[dataFileIndex]
			else:
				path = generateDataFileName(dataFileIndex)	#OLD: self.paths[dataFileIndex] - requires dataFolder to contain all dataFiles up to dataFileIndex
			
			with open(path, 'r', encoding='utf-8') as fp:
				lines = fp.read().split('\n')
			
			sample = tokenizer(lines, max_length=sequenceMaxNumTokens, padding='max_length', truncation=True, return_tensors='pt')
			if(centralSequencePrediction):
				self.reorderSampleStartConclusionSentence(tokenizer, sample, lines)
	
			input_ids = []
			mask = []
			labels = []
			if(legacyDataloaderCode2):
				labels.append(sample.input_ids)
			else:
				labels.append(addLabelsPredictionMaskTokens(sample.input_ids))
			mask.append(sample.attention_mask)
			sample_input_ids = (sample.input_ids).detach().clone()
			if(useMaskedLM):
				input_ids.append(addMaskTokens(sample_input_ids))
			else:
				input_ids.append(sample_input_ids)	#labels are redundant (equivalent to input_ids)
			input_ids = torch.cat(input_ids)
			mask = torch.cat(mask)
			labels = torch.cat(labels)
			
			self.encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}
		
		return {key: tensor[itemIndexInSample] for key, tensor in self.encodings.items()}
	
def createDataLoader(tokenizer, paths, numberOfDataFiles, pathIndexMin, pathIndexMax):

	dataFileIndexList = list(range(pathIndexMin, pathIndexMax))
	print("dataFileIndexList = ", dataFileIndexList)
	
	numberOfDocuments = numberOfDataFiles*numberOfSamplesPerDataFile	#equivalent number of documents (assuming it were loading data files)
	
	dataset = DatasetHDD(numberOfDocuments, dataFileIndexList, paths)

	loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False)	#shuffle not supported by DatasetHDD

	return loader

def continueTrainingModel():
	continueTrain = False
	if((trainStartEpoch > 0) or (trainStartDataFile > 0)):
		continueTrain = True	#if trainStartEpoch=0 and trainStartDataFile=0 will recreate model, if trainStartEpoch>0 or trainStartDataFile>0 will load existing model
	return continueTrain	

def trainDataset(tokenizer, paths):

	excluded_token_set = generateProsodyExcludedTokenSet(tokenizer)
	
	if(continueTrainingModel()):
		print("loading existing model")
		model = RobertaLM.from_pretrained(modelFolderName, local_files_only=True)
	else:
		print("creating new model")
		config = RobertaConfig(
			vocab_size=vocabularySize,  #sync with tokenizer vocab_size
			max_position_embeddings=(sequenceMaxNumTokens+2),
			hidden_size=hiddenLayerSize,
			num_attention_heads=numberOfAttentionHeads,
			num_hidden_layers=numberOfHiddenLayers,
			intermediate_size=intermediateSize,
			type_vocab_size=1,
			position_embedding_type=positionEmbeddingType,
			is_decoder=True
		)
		model = RobertaLM(config)
	
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	model.to(device)

	model.train()
	if(useTrainWarmup):
		learningRateCurrent = warmupLearningRateStart
	else:
		learningRateCurrent = learningRate
	optim = AdamW(model.parameters(), lr=learningRateCurrent)
	
	pathIndexMin = trainStartDataFile
	pathIndexMax = pathIndexMin+trainNumberOfDataFiles
	loader = createDataLoader(tokenizer, paths, trainNumberOfDataFiles, pathIndexMin, pathIndexMax)
	
	model.save_pretrained(modelFolderName)
	
	for epoch in range(trainStartEpoch, trainStartEpoch+trainNumberOfEpochs):
		loop = tqdm(loader, leave=True)
		for batchIndex, batch in enumerate(loop):
			optim.zero_grad()
			
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)
			
			#print("input_ids = ", input_ids)
			#print("attention_mask = ", attention_mask)
			
			outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
						
			accuracy = getAccuracy(input_ids, attention_mask, labels, outputs, excluded_token_set)
			loss = outputs.loss
			
			loss.backward()
			optim.step()

			if(useTrainWarmup):
				if(epoch == trainStartEpoch):
					if(batchIndex < warmupSteps):
						learningRateCurrent += warmupLearningRateIncrement
						for param_group in optim.param_groups:
							param_group['lr'] = learningRateCurrent
			
			loop.set_description(f'Epoch {epoch}')
			loop.set_postfix(batchIndex=batchIndex, loss=loss.item(), accuracy=accuracy)
		
			if(batchIndex % modelSaveNumberOfBatches == 0):
				model.save_pretrained(modelFolderName)
		model.save_pretrained(modelFolderName)

def testDataset(tokenizer, paths):

	excluded_token_set = generateProsodyExcludedTokenSet(tokenizer)

	if(usePretrainedModelDebug):
		model = RobertaLM.from_pretrained("roberta-base")
	else:
		model = RobertaLM.from_pretrained(modelFolderName, local_files_only=True)

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	model.to(device)

	model.eval()
	
	if(legacyDataloaderCode1):
		numberOfDataFiles = len(paths)
		pathIndexMin = int(numberOfDataFiles*trainSplitFraction)
	else:
		pathIndexMin = int(datasetNumberOfDataFiles*trainSplitFraction)
	pathIndexMax = pathIndexMin+testNumberOfDataFiles		
	loader = createDataLoader(tokenizer, paths, testNumberOfDataFiles, pathIndexMin, pathIndexMax)
		
	for epoch in range(trainStartEpoch, trainStartEpoch+trainNumberOfEpochs):
		loop = tqdm(loader, leave=True)
		
		averageAccuracy = 0.0
		averageLoss = 0.0
		batchCount = 0
		
		for batchIndex, batch in enumerate(loop):
			
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)
						
			outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
			
			accuracy = getAccuracy(input_ids, attention_mask, labels, outputs, excluded_token_set)
			loss = outputs.loss
			loss = loss.detach().cpu().numpy()
			
			if(not math.isnan(accuracy)):	#required for usePretrainedModelDebug only
				averageAccuracy = averageAccuracy + accuracy
				averageLoss = averageLoss + loss
				batchCount = batchCount + 1

			loop.set_description(f'Epoch {epoch}')
			loop.set_postfix(batchIndex=batchIndex, loss=loss, accuracy=accuracy)

		averageAccuracy = averageAccuracy/batchCount
		averageLoss = averageLoss/batchCount
		print("averageAccuracy = ", averageAccuracy)
		print("averageLoss = ", averageLoss)
		
def getTokenizerLength(tokenizer):
	return len(tokenizer)	#Size of the full vocabulary with the added token	#https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils.py

def getAccuracy(input_ids, attention_mask, labels, outputs, excluded_token_set=None):
	if(useMaskedLM):
		return getAccuracyMaskedLM(input_ids, labels, outputs, excluded_token_set)
	else:
		return getAccuracyCausalLM(input_ids, outputs, attention_mask, excluded_token_set)

def read_token_ids(file_path):
	with open(file_path, 'r') as file:
		token_ids = [int(line.strip()) for line in file]
	return token_ids
	
def generateProsodyExcludedTokenSet(tokenizer):
	excluded_token_set = None
	if(prosodyDelimitedData):
		if(prosodyDelimitedType=="controlTokens" or prosodyDelimitedType=="txtDebug"):
			excludedTokensIntList = []
		elif(prosodyDelimitedType=="uniqueTokens"):
			excludedTokensIntList = read_token_ids("prosodyTokenIDs.txt")
			print("excludedTokensIntList = ", excludedTokensIntList)
			#excludedTokensStringList = read_token_ids("prosodyTokens.txt")
			#excludedTokensIntList = [tokenizer.encode(token, add_special_tokens=False)[0] for token in excludedTokensStringList]
		elif(prosodyDelimitedType=="repeatTokens"):
			excludedTokensIntList = []	#TODO
		excluded_token_set = set(excludedTokensIntList)
	return excluded_token_set
				
def removeProsodyTokensFromPredictionMask(predictionMask, labels, excluded_token_set):
	if(prosodyDelimitedData):
		if(prosodyDelimitedType=="uniqueTokens"):	#FUTURE: or prosodyDelimitedType=="repeatTokens"
			for token in excluded_token_set:
				predictionMask &= (labels != token)
	return predictionMask
				
def getAccuracyMaskedLM(input_ids, labels, outputs, excluded_token_set=None):
	predictionMask = torch.where(input_ids==customMaskTokenID, 1.0, 0.0)	#maskTokenIndexFloat = maskTokenIndex.float()		#orig: maskTokenIndex
	###predictionMask = removeProsodyTokensFromPredictionMask(predictionMask, labels, excluded_token_set)
	tokenLogits = (outputs.logits).detach()
	accuracy = getAccuracyWithPredictionMask(labels, tokenLogits, predictionMask)
	return accuracy
	
def getAccuracyCausalLM(inputs, outputs, attention_mask, excluded_token_set=None):	
	#based on SBNLPpt_data:getAccuracyMaskedLM
	predictionMask = attention_mask[:, 1:]
	logits = outputs.logits.detach()
	# Shift so that tokens < n predict n
	shift_labels = inputs[..., 1:].contiguous()
	shift_logits = logits[..., :-1, :].contiguous()
	###predictionMask = removeProsodyTokensFromPredictionMask(predictionMask, shift_labels, excluded_token_set)
	accuracy = getAccuracyWithPredictionMask(shift_labels, shift_logits, predictionMask)
	accuracy = accuracy.item()
	#print("accuracy = ", accuracy)
	return accuracy

def getAccuracyWithPredictionMask(labels, tokenLogits, predictionMask):	
	tokenLogitsTopIndex = torch.topk(tokenLogits, accuracyTopN).indices	#get highest n scored entries from dictionary	#tokenLogitsTopIndex.shape = batchSize, sequenceMaxNumTokens, accuracyTopN
	#print("tokenLogitsTopIndex = ", tokenLogitsTopIndex)
	#print("labels = ", labels)
	#print("predictionMask = ", predictionMask)
	if(accuracyTopN == 1):
		tokenLogitsTopIndex = torch.squeeze(tokenLogitsTopIndex)	#tokenLogitsTopIndex[:, :, 1] -> #tokenLogitsTopIndex[:, :]
		#print("tokenLogitsTopIndex = ", tokenLogitsTopIndex)
		comparison = (tokenLogitsTopIndex == labels).float()
		comparisonMasked = torch.multiply(comparison, predictionMask)
		accuracy = (torch.sum(comparisonMasked)/torch.sum(predictionMask)).cpu().numpy()	#accuracy.item()
	else:
		labelsExpanded = torch.unsqueeze(labels, dim=2)
		labelsExpanded = labelsExpanded.expand(-1, -1, tokenLogitsTopIndex.shape[2])	#labels broadcasted to [batchSize, sequenceMaxNumTokens, accuracyTopN]
		comparison = (tokenLogitsTopIndex == labelsExpanded).float()
		predictionMaskExpanded = torch.unsqueeze(predictionMask, dim=2)
		predictionMaskExpanded = predictionMaskExpanded.expand(-1, -1, tokenLogitsTopIndex.shape[2])	#predictionMask broadcasted to [batchSize, sequenceMaxNumTokens, accuracyTopN]
		comparisonMasked = torch.multiply(comparison, predictionMaskExpanded)	#predictionMask broadcasted to [batchSize, sequenceMaxNumTokens, accuracyTopN]
		accuracy = (torch.sum(comparisonMasked)/torch.sum(predictionMask)).cpu().numpy() 	#or torch.sum(comparisonMasked)/(torch.sum(predictionMaskExpanded)/accuracyTopN)	#accuracy.item()
	return accuracy

def getPaths(dataPathName):
	if(Path(dataPathName).exists()):
		pathsGlob = Path(dataPathName).glob('**/*' + dataPreprocessedFileNameEnd)
		if(sortDataFilesByName):
			pathsGlob = sorted(pathsGlob, key=os.path.getmtime)	#key required because path names indices are not padded with 0s
		paths = [str(x) for x in pathsGlob]
	else:
		print("main error: Path does not exist, dataPathName = ", dataPathName)
		exit()
	return paths
	
if(__name__ == '__main__'):
	if(statePreprocessDataset):
		dataset = downloadDataset()
		preprocessDataset(dataset)
	paths = getPaths(dataFolder)  #[str(x) for x in Path(dataFolder).glob('**/*.txt')]
	if(usePretrainedModelDebug):
		tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
		testDataset(tokenizer, paths)
	else:
		if(stateTrainTokenizer):
			trainTokenizer(paths)
		if(stateTrainDataset or stateTestDataset):
			tokenizer = loadTokenizer()
		if(stateTrainDataset):
			trainDataset(tokenizer, paths)
		if(stateTestDataset):
			testDataset(tokenizer, paths)
			
def printe(str):
	print(str)
	exit()
