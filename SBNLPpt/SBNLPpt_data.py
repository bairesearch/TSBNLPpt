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
from pathlib import Path
from SBNLPpt_globalDefs import *
if(useFullwordTokenizer):
	import SBNLPpt_tokeniserFullword
if(tokeniserOnlyTrainOnDictionary):
	from nltk.corpus import words
		
if(not useLovelyTensors):
	torch.set_printoptions(profile="full")

#store models to large datasets partition cache folder (not required)
#os.environ['TRANSFORMERS_CACHE'] = '/media/user/datasets/models/'	#select partition with 3TB+ disk space

def getNumberOfDataFiles(dataElements):
	#if(usePreprocessedDataset):
	#	paths = dataElements
	#	numberOfDataFiles = len(paths)
	return numberOfDataFiles
	
def loadDataset():
	if(datasetName == 'OSCAR1900'):
		if(usePreprocessedDataset):
			if(useSmallDatasetDebug):
				dataset = load_dataset('nthngdy/oscar-small', "unshuffled_original_en", streaming=False, cache_dir=downloadCachePathName, split='train')	#or unshuffled_deduplicated_en
			else:
				dataset = load_dataset('oscar', "unshuffled_deduplicated_en", streaming=False, cache_dir=downloadCachePathName, split='train')
		else:
			dataset = load_dataset('oscar', "unshuffled_deduplicated_en", streaming=True, cache_dir=downloadCachePathName, split='train')	#https://huggingface.co/docs/datasets/stream
			#Got disconnected from remote data host. Retrying in 5sec [1/20]
		#dataset = dataset['train']
	elif(datasetName == 'OSCAR2201'):
		if(usePreprocessedDataset):
			dataset = load_dataset('oscar-corpus/OSCAR-2201', language='en', use_auth_token=tokenString, streaming=False, cache_dir=downloadCachePathName, split='train')
		else:
			dataset = load_dataset('oscar-corpus/OSCAR-2201', language='en', use_auth_token=tokenString, streaming=True, cache_dir=downloadCachePathName, split='train')	#https://huggingface.co/datasets/oscar-corpus/OSCAR-2201
	else:
		print("loadDataset error: datasetName unknown; ", datasetName)
		
	return dataset

def prepareDataElements(dataset):
	if(usePreprocessedDataset):
		if(Path(dataPathName).exists()):
			pathsGlob = Path(dataPathName).glob('**/*.txt')
			if(sortDataFilesByName):
				pathsGlob = sorted(pathsGlob, key=os.path.getmtime)	#key required because path names indices are not padded with 0s
			paths = [str(x) for x in pathsGlob]
		else:
			print("main error: Path does not exist, dataPathName = ", dataPathName)
			exit()
		dataElements = paths
	else:
		dataElements = dataset
	return dataElements
		
def preprocessDataset(dataset):
	if(usePreprocessedDataset):
		textData = []
		fileCount = 0
		for documentIndex, document in enumerate(tqdm(dataset)):
			documentText = document['text']
			preprocessDocumentText(documentText)
			textData.append(document)
			if(len(textData) == numberOfDocumentsPerDataFile):
				writeDataFile(fileCount, textData)
				textData = []
				fileCount += 1
			if(preprocessLimitNumberDataFiles):
				if(documentIndex == preprocessNumberOfDataFiles):
					break
		writeDataFile(fileCount, textData)	#remaining data file will be < numberOfDocumentsPerDataFile
	else:
		print("preprocessDataset error: statePreprocessDataset and !usePreprocessedDataset")
		exit()
			
def writeDataFile(fileCount, textData):
	fileName = dataPathName + "/text_" + str(fileCount) + ".txt"
	with open(fileName, 'w', encoding='utf-8') as fp:
		fp.write('\n'.join(textData))

def tokenise(lines, tokenizer, maxLength):
	if(useFullwordTokenizerClass):
		if(maxLength is None):
			sample = tokenizer(lines, return_tensors='pt')
		else:
			sample = tokenizer(lines, max_length=maxLength, padding='max_length', truncation=True, return_tensors='pt')
	else:
		sample = SBNLPpt_tokeniserFullword.tokenizeBasic(lines, tokenizer)
	return sample

def trainTokeniser(dataElements, vocabSize):	
	if(useFullwordTokenizer):
		SBNLPpt_tokeniserFullword.trainTokenizerFullwords(dataElements, vocabularySize)	#default method (vocabSize used by GIA word2vec model will be greater than numberOfTokens in tokenizer)
	else:
		trainTokeniserSubwords(dataElements, vocabularySize)

def createDictionaryFile():
	dictionaryList = words.words() 
	print("len(dictionaryList) = ", len(dictionaryList))
	fileName = modelPathName + "/dictionary.txt"
	with open(fileName, 'w', encoding='utf-8') as fp:
		fp.write(' '.join(dictionaryList))
	return fileName
						
def trainTokeniserSubwords(dataElements, vocabSize):	
	trainTokeniserFromDataFiles = usePreprocessedDataset
	
	if(tokeniserOnlyTrainOnDictionary):
		min_frequency = 1
		trainTokenizerNumberOfFilesToUse = 1
		path = createDictionaryFile()
		paths = []
		paths.append(path)
		trainTokeniserFromDataFiles = True
	else:
		min_frequency = 2
		if(useSmallTokenizerTrainNumberOfFiles):
			trainTokenizerNumberOfFilesToUse = trainTokenizerNumberOfFilesToUseSmall
		else:
			trainTokenizerNumberOfFilesToUse = getNumberOfDataFiles(dataElements)

	tokenizer = ByteLevelBPETokenizer()

	if(trainTokeniserFromDataFiles):
		tokenizer.train(files=paths[:trainTokenizerNumberOfFilesToUse], vocab_size=vocabSize, min_frequency=1, special_tokens=specialTokens)
	else:
		tokenizer.train_from_iterator(dataset, length=trainTokenizerNumberOfFilesToUse, vocab_size=vocabSize, min_frequency=1, special_tokens=specialTokens)
	
	#os.mkdir(modelPathName)

	tokenizer.save_model(modelPathName)
		
	return tokenizer

def loadTokeniser():
	if(useFullwordTokenizer):
		tokenizer = SBNLPpt_tokeniserFullword.loadTokenizerFullwords()
	else:
		tokenizer = loadTokenizerSubwords()
	return tokenizer
		
def loadTokenizerSubwords():	
	tokenizer = RobertaTokenizer.from_pretrained(modelPathName, max_len=sequenceMaxNumTokens)
	return tokenizer

	
def addMaskTokensBatch(useMLM, inputIDs):
	for i in range(inputIDs.shape[0]):
		inputIDs[i] = addMaskTokensSample(inputIDs[i])
	return inputIDs

def addMaskTokensSample(useMLM, inputIDs):
	if(useMLM):
		rand = torch.rand(inputIDs.shape)
		mask_arr = (rand < fractionOfMaskedTokens) * (inputIDs > 2)	#or * (inputIDs != 0) * (inputIDs != 1) * (inputIDs != 2)
	else:	
		mask_arr = (inputIDs > 2)	#or * (inputIDs != 0) * (inputIDs != 1) * (inputIDs != 2)
	selection = torch.flatten(mask_arr.nonzero()).tolist()
	inputIDs[selection] = customMaskTokenID
	return inputIDs
	

def dataFileIndexListContainsLastDocument(dataFileIndexList):
	containsDataFileLastDocument = False
	for dataFileIndex in dataFileIndexList:
		if(str(dataFileIndex) == dataFileLastIndex):
			containsDataFileLastDocument = True
	return containsDataFileLastDocument
	
class DatasetHDD(torch.utils.data.Dataset):
	def __init__(self, useMLM, dataFileIndexList, dataElements, tokenizer):
		self.useMLM = useMLM
		self.dataFileIndexList = dataFileIndexList
		self.paths = dataElements
		self.tokenizer = tokenizer
		self.containsDataFileLastDocument = dataFileIndexListContainsLastDocument(dataFileIndexList)
		self.documentIndexInDataFile = 0
		self.datasetNumberOfDocuments = len(self)
		if(createOrderedDataset):
			self.dataFileLinesIterator = None
			self.documentSamplesBatched = None
			self.sampleIndexInDocument = 0
			self.batchIndexInSample = 0
		else:
			self.encodings = None

	def __len__(self):
		datasetNumberOfDocuments = getDatasetNumberOfDocuments(self.dataFileIndexList, self.containsDataFileLastDocument)
		return datasetNumberOfDocuments

	def __getitem__(self, i):
		loadNextDataFile = False
		dataFileIndexRelative = i // numberOfDocumentsPerDataFile
		if(not createOrderedDataset):
			self.documentIndexInDataFile = i % numberOfDocumentsPerDataFile
		if(self.documentIndexInDataFile == 0):
			loadNextDataFile = True	
		dataFileIndex = self.dataFileIndexList[dataFileIndexRelative]
					
		if(loadNextDataFile):
			path = self.paths[dataFileIndex]
			with open(path, 'r', encoding='utf-8') as fp:
				lines = fp.read().split('\n')
			if(createOrderedDataset):
				self.documentIndexInDataFile = 0
				self.sampleIndexInDocument = 0
				self.batchIndexInSample = 0
				self.dataFileLinesIterator = iter(lines)
			else:
				sample = tokenise(lines, self.tokenizer, sequenceMaxNumTokens)
				self.encodings = getSampleEncodings(self.useMLM, sample.input_ids, sample.attention_mask, True)

		if(createOrderedDataset):
			batchSample, self.documentSamplesBatched, self.documentIndexInDataFile, self.batchIndexInSample, self.sampleIndexInDocument = getOrderedBatchSample(
				self.documentSamplesBatched, self.documentIndexInDataFile, self.batchIndexInSample, self.sampleIndexInDocument, self.tokenizer, self.dataFileLinesIterator, self.useMLM, self.datasetNumberOfDocuments
			)
		else:		
			batchSample = {key: tensor[self.documentIndexInDataFile] for key, tensor in self.encodings.items()}	
		return batchSample
			
class DatasetInternet(torch.utils.data.Dataset):
	def __init__(self, useMLM, dataFileIndexList, dataElements, tokenizer):
		self.useMLM = useMLM
		self.dataFileIndexList = dataFileIndexList
		self.datasetIterator = iter(dataElements)
		self.tokenizer = tokenizer
		self.containsDataFileLastDocument = dataFileIndexListContainsLastDocument(dataFileIndexList)
		self.datasetNumberOfDocuments = len(self)

		self.documentIndex = 0
		if(createOrderedDataset):
			self.documentSamplesBatched = None
			self.sampleIndexInDocument = 0
			self.batchIndexInSample = 0
			
	def __len__(self):
		datasetNumberOfDocuments = getDatasetNumberOfDocuments(self.dataFileIndexList, self.containsDataFileLastDocument)
		return datasetNumberOfDocuments

	def __getitem__(self, i):
		if(createOrderedDataset):
			batchSample, self.documentSamplesBatched, self.documentIndex, self.batchIndexInSample, self.sampleIndexInDocument = getOrderedBatchSample(
				self.documentSamplesBatched, self.documentIndex, self.batchIndexInSample, self.sampleIndexInDocument, self.tokenizer, self.datasetIterator, self.useMLM, self.datasetNumberOfDocuments
			)
		else:
			documentText, reachedEndOfDataset = getNextDocument(self.datasetIterator)
			documentText = preprocessDocumentText(documentText)
			documentTokens = tokenise(documentText, self.tokenizer, sequenceMaxNumTokens)
			encodings = getSampleEncodings(self.useMLM, documentTokens.input_ids[0], documentTokens.attention_mask[0], False)
			batchSample = encodings
			self.documentIndex+=1
			
			if(reachedEndOfDataset):
				if(self.documentIndex != self.datasetNumberOfDocuments):
					print("DatasetInternet: error reachedEndOfDataset and self.documentIndex != len(self)")
					exit()
				
		return batchSample

def preprocessDocumentText(documentText):
	if(preprocessRemoveNewLineCharacters):
		documentText = documentText.replace('\n', '')
	return documentText


def getDatasetNumberOfDocuments(dataFileIndexList, containsDataFileLastDocument):
	datasetNumberOfDocuments = len(dataFileIndexList)*numberOfDocumentsPerDataFile
	if(containsDataFileLastDocument):
		datasetNumberOfDocuments = datasetNumberOfDocuments-numberOfDocumentsPerDataFile + numberOfSamplesPerDataFileLast
	return datasetNumberOfDocuments
		
def getSampleEncodings(useMLM, input_ids, attention_mask, batched):
	#print("input_ids = ", input_ids)
	#print("attention_mask = ", attention_mask)
	inputIDs = []
	mask = []
	labels = []
	labels.append(input_ids)
	mask.append(attention_mask)
	sampleInputIDs = (input_ids).detach().clone()
	if(batched):
		sampleInputIDsMasked = addMaskTokensBatch(useMLM, sampleInputIDs)
	else:
		sampleInputIDsMasked = addMaskTokensSample(useMLM, sampleInputIDs)
	inputIDs.append(sampleInputIDsMasked)
	inputIDs = torch.cat(inputIDs)
	mask = torch.cat(mask)
	labels = torch.cat(labels)
	encodings = {'inputIDs': inputIDs, 'attentionMask': mask, 'labels': labels}
	return encodings
	
def getOrderedBatchSample(documentSamplesBatched, documentIndex, batchIndexInSample, sampleIndexInDocument, tokenizer, dataFileLinesIterator, useMLM, datasetNumberOfDocuments):
	if(sampleIndexInDocument == 0):
		documentSamplesBatched, documentIndex, reachedEndOfDataFile = getBatchOrderedSamples(dataFileLinesIterator, documentIndex, tokenizer)
		if(reachedEndOfDataFile):
			if(usePreprocessedDataset):
				documentIndex = 0
				if(documentIndex != numberOfDocumentsPerDataFile):
					print("DatasetHDD error: reachedEndOfDataFile && documentIndex != numberOfDocumentsPerDataFile")
					exit()
			else:
				if(documentIndex != datasetNumberOfDocuments):
					print("DatasetInternet: error reachedEndOfDataset and self.documentIndex != datasetNumberOfDocuments")
					exit()
	documentSample = documentSamplesBatched[sampleIndexInDocument][batchIndexInSample]
	input_ids = documentSample
	attention_mask = generateAttentionMask(tokenizer, input_ids)
	encodings = getSampleEncodings(useMLM, input_ids, attention_mask, False)
	batchIndexInSample+=1
	sampleIndexInDocument+=1
	if(batchIndexInSample == batchSize):
		batchIndexInSample = 0
	if(sampleIndexInDocument == orderedDatasetDocNumberSamples):
		sampleIndexInDocument = 0
	batchSample = encodings
	return batchSample, documentSamplesBatched, documentIndex, batchIndexInSample, sampleIndexInDocument

def getBatchOrderedSamples(datasetIterator, documentIndex, tokenizer):
	reachedEndOfDataset = False
	stillGettingbatchDocumentSamples = True
	batchDocumentSamples = []
	batchDocumentSampleIndex = 0
	while(stillGettingbatchDocumentSamples):
		documentText, reachedEndOfDataset = getNextDocument(datasetIterator)
		documentText = preprocessDocumentText(documentText)
		documentIndex+=1
		if(reachedEndOfDataset):
			stillGettingbatchDocumentSamples = False
		if(len(documentText) > orderedDatasetDocMinSizeCharacters):
			documentTokens = tokenise(documentText, tokenizer, None)
			documentTokensIDs = documentTokens.input_ids[0]
			batchDocumentSampleIndex = splitDocumentTokens(documentTokensIDs, batchDocumentSamples, batchDocumentSampleIndex)
		if(batchDocumentSampleIndex == batchSize):
			stillGettingbatchDocumentSamples = False
	
	documentSamplesBatchList = list(map(list, zip(*batchDocumentSamples)))	#transpose list of lists: batchSize*numberOfDocumentSamples -> numberOfDocumentSamples*batchSize
	#printDocumentSamplesBatchList(tokenizer, documentSamplesBatchList)
	
	return documentSamplesBatchList, documentIndex, reachedEndOfDataset

def printDocumentSamplesBatchList(tokenizer, documentSamplesBatchList):
	for sampleIndex1 in range(orderedDatasetDocNumberSamples):
		print("sampleIndex1 = ", sampleIndex1)
		for sampleIndex2 in range(batchSize):
			print("sampleIndex2 = ", sampleIndex2)
			sample_ids = documentSamplesBatchList[sampleIndex1][sampleIndex2]
			sampleString = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(sample_ids))
			print("sample = ", sampleString)
			
def splitDocumentTokens(documentTokens, batchDocumentSamples, batchDocumentSampleIndex):
	if(orderedDatasetSplitDocumentsBySentences):
		print("splitDocumentTokens error: orderedDatasetSplitDocumentsBySentences not yet coded")
		exit()
	else:
		if(documentTokens.shape[0] >= orderedDatasetDocNumberTokens):
			documentTokens = documentTokens[0:orderedDatasetDocNumberTokens]
			#documentSamples = [documentTokens[x:x+sequenceMaxNumTokens] for x in xrange(0, len(documentTokens), sequenceMaxNumTokens)]
			documentSamples = torch.split(documentTokens, split_size_or_sections=sequenceMaxNumTokens, dim=0)
			batchDocumentSamples.append(documentSamples)
			batchDocumentSampleIndex+=1
	return batchDocumentSampleIndex

def generateAttentionMask(tokenizer, input_ids):
	attention_mask = (input_ids > 2).float()	#or * (inputIDs != 0) * (inputIDs != 1) * (inputIDs != 2)	# not in [tokenizer.unk_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
	return attention_mask
	
def getNextDocument(datasetIterator):
	reachedEndOfDataset = False
	document = next(datasetIterator)	
	'''
	try:
		document = next(datasetIterator)
		print("as2")
	except StopIteration:
		reachedEndOfDataset = True
		print("as1")
	'''
	if(usePreprocessedDataset):
		documentText = document
	else:
		documentText = document['text']	
	return documentText, reachedEndOfDataset
			
def getOscar2201DocumentLengthCharacters(document):
	documentLengthCharacters = len(document['text'])	#number of characters
	'''
	meta = document['meta']
	warc_headers = meta['warc_headers']
	content_length = warc_headers['content-length']	#in bytes (characters)
	documentLengthCharacters = content_length
	'''
	return documentLengthCharacters

def createDataLoader(useMLM, tokenizer, dataElements, pathIndexMin, pathIndexMax):

	dataFileIndexList = list(range(pathIndexMin, pathIndexMax))
	print("dataFileIndexList = ", dataFileIndexList)
	
	if(usePreprocessedDataset):
		dataLoaderDatasetObject = DatasetHDD(useMLM, dataFileIndexList, dataElements, tokenizer)
	else:
		dataLoaderDatasetObject = DatasetInternet(useMLM, dataFileIndexList, dataElements, tokenizer)	

	loader = torch.utils.data.DataLoader(dataLoaderDatasetObject, batch_size=batchSize, shuffle=False)	#shuffle not supported by DatasetHDD

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

