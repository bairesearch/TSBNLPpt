"""SBNLPpt_data.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SBNLPpt_main.py

# Usage:
see SBNLPpt_main.py

# Description:
SBNLPpt data

"""

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizer
import os
from SBNLPpt_globalDefs import *

specialTokens = ['<s>', '<pad>', '</s>', '<unk>', '<mask>']
specialTokenPadding = '<pad>'
specialTokenMask = '<mask>'

if(useFullwordTokenizer):
	if(useFullwordTokenizerNLTK):
		import nltk
	else:
		from transformers import DistilBertTokenizer
		tokenizerFullword = DistilBertTokenizer.from_pretrained('distilbert-base-cased')		
	
	if(useFullwordTokenizerPretrained):
		if(useFullwordTokenizerPretrainedAuto):
			from transformers import AutoTokenizer	#alternate method using a pretrained tokenizer
	else:
		if(useFullwordTokenizerFast):
			from transformers import TokenizerFast	#TokenizerFast does not support save_pretrained/from_pretrained, so it requires a new save function
		import json
		
if(not useLovelyTensors):
	torch.set_printoptions(profile="full")

#store models to large datasets partition cache folder (not required)
#os.environ['TRANSFORMERS_CACHE'] = '/media/user/datasets/models/'	#select partition with 3TB+ disk space


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

class TokenizerOutputEmulate():
	def __init__(self, inputID, maskID):
		self.input_ids = inputID
		self.attention_mask = maskID	#artificial mask; padding and mask tokens only

class TokenizerBasic():
	def __init__(self):
		self.dict = {}
		self.list = []


def tokenize(lines, tokenizer):
	if(useFullwordTokenizerClass):
		sample = tokenizer(lines, max_length=sequenceMaxNumTokens, padding='max_length', truncation=True, return_tensors='pt')
	else:
		#print("lines = ", lines)
		inputIDlist = []
		maskIDlist = []
		for lineIndex, line in enumerate(lines):
			#print("lineIndex = ", lineIndex)
			tokens = fullwordTokenizeLine(line)
			tokensLength = len(tokens)
			tokensLengthTruncated = min(tokensLength, sequenceMaxNumTokens)
			inputTokens = tokens[0:tokensLengthTruncated] 
			paddingLength = sequenceMaxNumTokens-tokensLengthTruncated
			maskTokens = [specialTokenMask]*tokensLengthTruncated
			#print("total length = ", tokensLengthTruncated+paddingLength)
			if(paddingLength > 0):
				maskPadding = [specialTokenPadding]*paddingLength
				inputs = inputTokens + maskPadding
				mask = maskTokens + maskPadding
			else:
				inputs = inputTokens
				mask = maskTokens
			inputID = [tokenizer.dict[x] for x in inputs]
			maskID = [tokenizer.dict[x] for x in mask]
			#print("inputID = ", inputID)
			#print("maskID = ", maskID)
			tokenIDtensor = torch.Tensor(inputID).to(torch.long)	#dtype=torch.int32
			maskIDtensor = torch.Tensor(maskID).to(torch.long)	#dtype=torch.int32
			inputIDlist.append(tokenIDtensor)
			maskIDlist.append(maskIDtensor)
		inputID = torch.stack(inputIDlist)
		maskID = torch.stack(maskIDlist)
		sample = TokenizerOutputEmulate(inputID, maskID)
	return sample


def trainTokenizer(paths, vocabSize):	
	if(useFullwordTokenizer):
		trainTokenizerFullwords(paths, vocabularySize)	#default method (vocabSize used by GIA word2vec model will be greater than numberOfTokens in tokenizer)
		#trainTokenizerSubwords(paths, vocabularySize)	#alternate method (TODO: verify does not still split words into subwords even with large vocabularySize)
	else:
		trainTokenizerSubwords(paths, vocabularySize)
				
def trainTokenizerSubwords(paths, vocabSize):	
	#subword tokenizer
	if(useSmallTokenizerTrainNumberOfFiles):
		trainTokenizerNumberOfFilesToUse = trainTokenizerNumberOfFilesToUseSmall
	else:
		trainTokenizerNumberOfFilesToUse = len(paths)

	tokenizer = ByteLevelBPETokenizer()

	tokenizer.train(files=paths[:trainTokenizerNumberOfFilesToUse], vocab_size=vocabSize, min_frequency=2, special_tokens=specialTokens)

	#os.mkdir(modelFolderName)

	tokenizer.save_model(modelFolderName)
		
	return tokenizer

def trainTokenizerFullwords(paths, vocabSize):
	if(useFullwordTokenizerPretrained):
		if(useFullwordTokenizerPretrainedAuto):
			tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
		else:
			tokenizer = ByteLevelBPETokenizer()
	else:
		if(useFullwordTokenizerFast):
			tokenizer = TokenizerFast()
		
	if(useSmallTokenizerTrainNumberOfFiles):
		trainTokenizerNumberOfFilesToUse = trainTokenizerNumberOfFilesToUseSmall
	else:
		trainTokenizerNumberOfFilesToUse = len(paths)
	
	#tokensList = []
	tokensSet = set()
	numberOfTokensNonUnique = 0
	for dataFileIndex in range(trainTokenizerNumberOfFilesToUse):
		path = paths[dataFileIndex]
		print("dataFileIndex = ", dataFileIndex)
		with open(path, 'r', encoding='utf-8') as fp:
			lines = fp.read().split('\n')
			for lineIndex, line in enumerate(lines):
				#print("lineIndex = ", lineIndex)
				tokens = fullwordTokenizeLine(line)
				#print("tokens = ", tokens)
				numberOfTokensNonUnique = numberOfTokensNonUnique + len(tokens)
				#tokensList.extend(tokens)
				tokensSet.update(tokens)
	
	tokensList = list(tokensSet)
	if(useFullwordTokenizerPretrained):
		#tokensList.extend(specialTokens)
		#tokenizer.train_on_texts(tokensList)
		tokenizer.add_tokens(tokensList)
		tokenizer.add_tokens(specialTokens)
		if(useFullwordTokenizerPretrainedAuto):
			tokenizer.save_pretrained(modelFolderName)
		else:
			tokenizer.save_model(modelFolderName)			
	else:
		if(useFullwordTokenizerFast):
			tokenizer.train(tokensList)
			tokensVocab = tokenizer.get_vocab()
			tokensSpecial = tokenizer.get_special_tokens_mask()
		else:
			tokensVocab = tokensList
			tokensSpecial = specialTokens
			tokenizer = tokensVocab	#for reference
		with open(tokensVocabPathName, "w") as handle:
			json.dump(tokensVocab, handle)
		with open(tokensSpecialPathName, "w") as handle:
			json.dump(tokensSpecial, handle)
		
	numberOfTokens = countNumberOfTokens(tokenizer)
	if(numberOfTokens > vocabSize):
		print("trainTokenizerFullwords error: numberOfTokens > vocabSize")
		print("vocabSize = ", vocabSize)
		print("numberOfTokens = ", numberOfTokens)
		print("numberOfTokensNonUnique = ", numberOfTokensNonUnique)
			
	return tokenizer
	
def fullwordTokenizeLine(line):
	if(useFullwordTokenizerNLTK):
		tokens = nltk.word_tokenize(line)
	else:
		tokens = tokenizerFullword.basic_tokenizer.tokenize(line)
	return tokens
						
def countNumberOfTokens(tokenizer):
	if(useFullwordTokenizerClass):	
		numberOfTokens = len(tokenizer.get_vocab())
	else:
		numberOfTokens = len(tokenizer)
	return numberOfTokens

def loadTokenizer():
	if(useFullwordTokenizer):
		tokenizer = loadTokenizerFullwords()
	else:
		tokenizer = loadTokenizerSubwords()
	return tokenizer
		
def loadTokenizerSubwords():	
	tokenizer = RobertaTokenizer.from_pretrained(modelFolderName, max_len=sequenceMaxNumTokens)
	return tokenizer
	
def loadTokenizerFullwords():	
	if(useFullwordTokenizerPretrained):
		if(useFullwordTokenizerPretrainedAuto):
			tokenizer = AutoTokenizer.from_pretrained(modelFolderName, max_len=sequenceMaxNumTokens)
		else:
			tokenizer = RobertaTokenizer.from_pretrained(modelFolderName, max_len=sequenceMaxNumTokens)
	else:		
		with open(tokensVocabPathName, "r") as handle:
			tokensVocab = json.load(handle)
		with open(tokensSpecialPathName, "r") as handle:
			tokensSpecial = json.load(handle)
		if(useFullwordTokenizerFast):
			tokenizer = TokenizerFast(vocab=tokensVocab, special_tokens_mask=special_tokens_mask)
		else:
			tokenizer = TokenizerBasic()
			tokensVocabDictionaryItems = createDictionaryItemsFromList(tokensVocab, 0)
			tokenizer.dict = dict(tokensVocabDictionaryItems)
			tokensSpecialDictionaryItems = createDictionaryItemsFromList(tokensSpecial, len(tokenizer.dict))
			for i, j in tokensSpecialDictionaryItems:
				tokenizer.dict[i] = j
			tokenizer.list = list(tokenizer.dict.keys())
	return tokenizer

def createDictionaryItemsFromList(lst, startIndex):
	list1 = lst
	list2 = range(startIndex, len(lst)+startIndex)
	dictionaryItems = zip(list1, list2)
	return dictionaryItems
	
def addMaskTokens(useMLM, inputIDs):
	if(useMLM):
		rand = torch.rand(inputIDs.shape)
		mask_arr = (rand < fractionOfMaskedTokens) * (inputIDs > 2)	#or * (inputIDs != 0) * (inputIDs != 1) * (inputIDs != 2)
		for i in range(inputIDs.shape[0]):
			selection = torch.flatten(mask_arr[i].nonzero()).tolist()
			inputIDs[i, selection] = customMaskTokenID
	else:	
		mask_arr = (inputIDs > 2)	#or * (inputIDs != 0) * (inputIDs != 1) * (inputIDs != 2)
		for i in range(inputIDs.shape[0]):
			selection = torch.flatten(mask_arr[i].nonzero()).tolist()
			inputIDs[i, selection] = customMaskTokenID
	return inputIDs

def dataFileIndexListContainsLastFile(dataFileIndexList, paths):
	containsDataFileLastSample = False
	for dataFileIndex in dataFileIndexList:
		path = paths[dataFileIndex]
		if(str(dataFileLastSampleIndex) in path):
			containsDataFileLastSample = True
	return containsDataFileLastSample
	
class DatasetHDD(torch.utils.data.Dataset):
	def __init__(self, useMLM, dataFileIndexList, paths, tokenizer):
		self.useMLM = useMLM
		self.dataFileIndexList = dataFileIndexList
		self.paths = paths
		self.encodings = None
		self.containsDataFileLastSample = dataFileIndexListContainsLastFile(dataFileIndexList, paths)
		self.tokenizer = tokenizer

	def __len__(self):
		numberOfSamples = len(self.dataFileIndexList)*numberOfSamplesPerDataFile
		if(self.containsDataFileLastSample):
			numberOfSamples = numberOfSamples-numberOfSamplesPerDataFile + numberOfSamplesPerDataFileLast
		return numberOfSamples

	def __getitem__(self, i):
	
		loadNextDataFile = False
		sampleIndex = i // numberOfSamplesPerDataFile
		itemIndexInSample = i % numberOfSamplesPerDataFile
		if(itemIndexInSample == 0):
			loadNextDataFile = True	
		dataFileIndex = self.dataFileIndexList[sampleIndex]
					
		if(loadNextDataFile):
			
			path = self.paths[dataFileIndex]

			with open(path, 'r', encoding='utf-8') as fp:
				lines = fp.read().split('\n')

			sample = tokenize(lines, self.tokenizer)
			inputIDs = []
			mask = []
			labels = []
			labels.append(sample.input_ids)
			mask.append(sample.attention_mask)
			sampleInputIDs = (sample.input_ids).detach().clone()
			inputIDs.append(addMaskTokens(self.useMLM, sampleInputIDs))
			inputIDs = torch.cat(inputIDs)
			mask = torch.cat(mask)
			labels = torch.cat(labels)
			
			self.encodings = {'inputIDs': inputIDs, 'attentionMask': mask, 'labels': labels}
		
		return {key: tensor[itemIndexInSample] for key, tensor in self.encodings.items()}

def createDataLoader(useMLM, tokenizer, paths, pathIndexMin, pathIndexMax):

	dataFileIndexList = list(range(pathIndexMin, pathIndexMax))
	print("dataFileIndexList = ", dataFileIndexList)
	
	dataset = DatasetHDD(useMLM, dataFileIndexList, paths, tokenizer)

	loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False)	#shuffle not supported by DatasetHDD

	return loader

def getTokenizerLength(tokenizer):
	return len(tokenizer)	#Size of the full vocabulary with the added token	#https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils.py


def getAccuracy(tokenizer, inputIDs, predictionMask, labels, outputs):
	tokenizerNumberTokens = getTokenizerLength(tokenizer)
	
	tokenLogits = outputs.detach()

	tokenLogitsTopIndex = torch.topk(tokenLogits, accuracyTopN).indices	#get highest n scored entries from dictionary	#tokenLogitsTopIndex.shape = batchSize, sequenceMaxNumTokens, accuracyTopN
	
	if(accuracyTopN == 1):
		tokenLogitsTopIndex = torch.squeeze(tokenLogitsTopIndex)	#tokenLogitsTopIndex[:, :, 1] -> #tokenLogitsTopIndex[:, :] 	

		comparison = (tokenLogitsTopIndex == labels).float()
		comparisonMasked = torch.multiply(comparison, predictionMask)
		accuracy = (torch.sum(comparisonMasked)/torch.sum(predictionMask)).cpu().numpy() 
	else:
		labelsExpanded = torch.unsqueeze(labels, dim=2)
		labelsExpanded = labelsExpanded.expand(-1, -1, tokenLogitsTopIndex.shape[2])	#labels broadcasted to [batchSize, sequenceMaxNumTokens, accuracyTopN]
		comparison = (tokenLogitsTopIndex == labelsExpanded).float()
		maskTokenIndexExpanded = torch.unsqueeze(predictionMask, dim=2)
		maskTokenIndexExpanded = maskTokenIndexExpanded.expand(-1, -1, tokenLogitsTopIndex.shape[2])	#predictionMask broadcasted to [batchSize, sequenceMaxNumTokens, accuracyTopN]
		comparisonMasked = torch.multiply(comparison, maskTokenIndexExpanded)	#predictionMask broadcasted to [batchSize, sequenceMaxNumTokens, accuracyTopN]
		accuracy = (torch.sum(comparisonMasked)/torch.sum(predictionMask)).cpu().numpy() 	#or torch.sum(comparisonMasked)/(torch.sum(maskTokenIndexExpanded)/accuracyTopN)
	
	#accuracy2 = (torch.mean(comparisonMasked)).cpu().numpy()
	
	return accuracy

