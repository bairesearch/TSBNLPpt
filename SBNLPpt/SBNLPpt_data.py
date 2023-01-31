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
			trainTokenizerNumberOfFilesToUse = datasetNumberOfDataFiles

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
		inputIDs[i] = addMaskTokensSample(useMLM, inputIDs[i])
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
	
class DataloaderDatasetHDD(torch.utils.data.Dataset):
	def __init__(self, useMLM, numberOfDocumentsEst, dataFileIndexList, dataElements, tokenizer):
		self.useMLM = useMLM
		self.dataFileIndexList = dataFileIndexList
		self.paths = dataElements
		self.tokenizer = tokenizer
		self.numberOfDocuments = getNumberOfDocuments(numberOfDocumentsEst, dataFileIndexList)
		if(createOrderedDataset):
			self.dataFileIndexRelative = 0
			self.documentIndexInDataFile = 0
			self.dataFileLinesIterator = None
			self.documentSegmentsBatchList = None
			self.segmentIndexInDocument = 0
			self.sampleIndexInBatch = 0
		else:
			self.encodings = None

	def __len__(self):
		return self.numberOfDocuments

	def __getitem__(self, i):
		loadNextDataFile = False
		if(not createOrderedDataset):
			self.dataFileIndexRelative = i // numberOfDocumentsPerDataFile
			self.documentIndexInDataFile = i % numberOfDocumentsPerDataFile
		if(self.documentIndexInDataFile == 0):
			loadNextDataFile = True	
		dataFileIndex = self.dataFileIndexList[self.dataFileIndexRelative]
		
		if(loadNextDataFile):
			#print("loadNextDataFile: dataFileIndex = ", dataFileIndex)
			path = self.paths[dataFileIndex]
			with open(path, 'r', encoding='utf-8') as fp:
				lines = fp.read().split('\n')
			if(createOrderedDataset):
				self.dataFileIndexRelative += 1
				self.documentIndexInDataFile = 0
				self.segmentIndexInDocument = 0
				self.sampleIndexInBatch = 0
				self.dataFileLinesIterator = iter(lines)
			else:
				sample = tokenise(lines, self.tokenizer, sequenceMaxNumTokens)
				self.encodings = getSampleEncodings(self.useMLM, sample.input_ids, sample.attention_mask, True)

		if(createOrderedDataset):
			batchSample, self.documentSegmentsBatchList, self.documentIndexInDataFile, self.sampleIndexInBatch, self.segmentIndexInDocument = getOrderedBatchSample(
				self.documentSegmentsBatchList, self.documentIndexInDataFile, self.sampleIndexInBatch, self.segmentIndexInDocument, self.tokenizer, self.dataFileLinesIterator, self.useMLM
			)
		else:		
			batchSample = {key: tensor[self.documentIndexInDataFile] for key, tensor in self.encodings.items()}	
		
		return batchSample

def getNumberOfDocuments(numberOfDocumentsEst, dataFileIndexList):
	if(createOrderedDataset):
		numberOfDocuments = numberOfDocumentsEst
	else:
		containsDataFileLastDocument = dataFileIndexListContainsLastDocument(dataFileIndexList)
		numberOfDocuments = len(dataFileIndexList)*numberOfDocumentsPerDataFile
		if(containsDataFileLastDocument):
			numberOfDocuments = numberOfDocuments-numberOfDocumentsPerDataFile + datasetNumberOfSamplesPerDataFileLast
	return numberOfDocuments
			
class DataloaderDatasetInternet(torch.utils.data.Dataset):
	def __init__(self, useMLM, numberOfDocuments, dataElements, tokenizer):
		self.useMLM = useMLM
		self.datasetIterator = iter(dataElements)
		self.tokenizer = tokenizer
		self.numberOfDocuments = numberOfDocuments

		self.documentIndex = 0
		if(createOrderedDataset):
			self.documentSegmentsBatchList = None
			self.segmentIndexInDocument = 0
			self.sampleIndexInBatch = 0
			
	def __len__(self):
		return self.numberOfDocuments

	def __getitem__(self, i):	
		if(createOrderedDataset):
			batchSample, self.documentSegmentsBatchList, self.documentIndex, self.sampleIndexInBatch, self.segmentIndexInDocument = getOrderedBatchSample(
				self.documentSegmentsBatchList, self.documentIndex, self.sampleIndexInBatch, self.segmentIndexInDocument, self.tokenizer, self.datasetIterator, self.useMLM
			)
		else:
			documentText = getNextDocument(self.datasetIterator)
			documentText = preprocessDocumentText(documentText)
			documentTokens = tokenise(documentText, self.tokenizer, sequenceMaxNumTokens)
			encodings = getSampleEncodings(self.useMLM, documentTokens.input_ids[0], documentTokens.attention_mask[0], False)
			batchSample = encodings
			self.documentIndex+=1
			
		return batchSample

def preprocessDocumentText(documentText):
	if(preprocessRemoveNewLineCharacters):
		documentText = documentText.replace('\n', '')
	return documentText
		
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
	
def getOrderedBatchSample(documentSegmentsBatchList, documentIndex, sampleIndexInBatch, segmentIndexInDocument, tokenizer, dataFileLinesIterator, useMLM):
	if(segmentIndexInDocument == 0):
		documentSegmentsBatchList, documentIndex, reachedEndOfDataFile = getDocumentSegments(dataFileLinesIterator, documentIndex, tokenizer)
		if(reachedEndOfDataFile):
			if(usePreprocessedDataset):
				if(documentIndex != numberOfDocumentsPerDataFile):
					print("DataloaderDatasetHDD error: reachedEndOfDataFile && documentIndex != numberOfDocumentsPerDataFile")
					exit()
				documentIndex = 0
			else:
				print("getOrderedBatchSample error: !usePreprocessedDataset does not support reachedEndOfDataset flag; next(datasetIterator) will throw StopIteration and batch iteration process will automatically stop")
				exit()
	#print("segmentIndexInDocument = ", segmentIndexInDocument, ", sampleIndexInBatch = ", sampleIndexInBatch)
	documentSample = documentSegmentsBatchList[segmentIndexInDocument][sampleIndexInBatch]
	input_ids = documentSample
	attention_mask = generateAttentionMask(tokenizer, input_ids)
	encodings = getSampleEncodings(useMLM, input_ids, attention_mask, False)
	sampleIndexInBatch+=1
	if(sampleIndexInBatch == batchSize):
		segmentIndexInDocument+=1
		sampleIndexInBatch = 0
	if(segmentIndexInDocument == orderedDatasetDocNumberSamples):
		segmentIndexInDocument = 0
		
	batchSample = encodings
	return batchSample, documentSegmentsBatchList, documentIndex, sampleIndexInBatch, segmentIndexInDocument

def getDocumentSegments(datasetIterator, documentIndex, tokenizer):
	reachedEndOfDataset = False
	stillFindingDocumentSegmentSamples = True
	documentSegmentsSampleList = []
	sampleIndex = 0
	while(stillFindingDocumentSegmentSamples):
		documentText = getNextDocument(datasetIterator)
		documentText = preprocessDocumentText(documentText)
		documentIndex+=1
		if(sampleIndex == batchSize):
			stillFindingDocumentSegmentSamples = False
		if(len(documentText) > orderedDatasetDocMinSizeCharacters):
			documentTokens = tokenise(documentText, tokenizer, None)
			documentTokensIDs = documentTokens.input_ids[0]
			sampleIndex = splitDocumentIntoSegments(documentTokensIDs, documentSegmentsSampleList, sampleIndex)
		if(usePreprocessedDataset):
			if(documentIndex == numberOfDocumentsPerDataFile):
				print("reachedEndOfDataset")
				reachedEndOfDataset = True
				stillFindingDocumentSegmentSamples = False
				while sampleIndex < batchSize:
					#fill remaining documentSegmentsSampleList rows with pad_token_id	#FUTURE implementation; load next data file
					documentTokensIDsIgnore = torch.full([sequenceMaxNumTokens*orderedDatasetDocNumberSamples], fill_value=tokenizer.pad_token_id, dtype=torch.long)
					sampleIndex = splitDocumentIntoSegments(documentTokensIDsIgnore, documentSegmentsSampleList, sampleIndex)

	documentSegmentsBatchList = list(map(list, zip(*documentSegmentsSampleList)))	#transpose list of lists: batchSize*numberOfDocumentSegments -> numberOfDocumentSegments*batchSize
	#printDocumentSegments(tokenizer, documentSegmentsBatchList)
		
	return documentSegmentsBatchList, documentIndex, reachedEndOfDataset

def printDocumentSegments(tokenizer, documentSegmentsBatchList):
	for segmentIndex1 in range(orderedDatasetDocNumberSamples):
		print("segmentIndex1 = ", segmentIndex1)
		for sampleIndex in range(batchSize):
			print("sampleIndex = ", sampleIndex)
			sample_ids = documentSegmentsBatchList[segmentIndex1][sampleIndex]
			sampleString = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(sample_ids))
			print("sample = ", sampleString)
			
def splitDocumentIntoSegments(documentTokens, documentSegmentsSampleList, sampleIndex):
	if(orderedDatasetSplitDocumentsBySentences):
		print("splitDocumentIntoSegments error: orderedDatasetSplitDocumentsBySentences not yet coded")
		exit()
	else:
		if(documentTokens.shape[0] >= orderedDatasetDocNumberTokens):
			documentTokens = documentTokens[0:orderedDatasetDocNumberTokens]
			#documentSegments = [documentTokens[x:x+sequenceMaxNumTokens] for x in xrange(0, len(documentTokens), sequenceMaxNumTokens)]
			documentSegments = torch.split(documentTokens, split_size_or_sections=sequenceMaxNumTokens, dim=0)
			documentSegmentsSampleList.append(documentSegments)
			sampleIndex+=1
	return sampleIndex

def generateAttentionMask(tokenizer, input_ids):
	attention_mask = (input_ids > 2).float()	#or * (inputIDs != 0) * (inputIDs != 1) * (inputIDs != 2)	# not in [tokenizer.unk_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
	return attention_mask
	
def getNextDocument(datasetIterator):
	document = next(datasetIterator)
	'''
	reachedEndOfDataset = False
	try:
		document = next(datasetIterator)
	except StopIteration:
		reachedEndOfDataset = True
	'''
	if(usePreprocessedDataset):
		documentText = document
	else:
		documentText = document['text']
	return documentText
			
def getOscar2201DocumentLengthCharacters(document):
	documentLengthCharacters = len(document['text'])	#number of characters
	'''
	meta = document['meta']
	warc_headers = meta['warc_headers']
	content_length = warc_headers['content-length']	#in bytes (characters)
	documentLengthCharacters = content_length
	'''
	return documentLengthCharacters

def createDataLoader(useMLM, tokenizer, dataElements, numberOfDataFiles, pathIndexMin, pathIndexMax):

	if(usePreprocessedDataset):
		dataFileIndexList = list(range(pathIndexMin, pathIndexMax))
		print("dataFileIndexList = ", dataFileIndexList)	
	numberOfDocuments = numberOfDataFiles*numberOfDocumentsPerDataFile	#equivalent number of documents (assuming it were loading data files)
	print("numberOfDocuments = ", numberOfDocuments)
	
	if(usePreprocessedDataset):
		dataLoaderDataset = DataloaderDatasetHDD(useMLM, numberOfDocuments, dataFileIndexList, dataElements, tokenizer)
	else:
		dataLoaderDataset = DataloaderDatasetInternet(useMLM, numberOfDocuments, dataElements, tokenizer)	

	loader = torch.utils.data.DataLoader(dataLoaderDataset, batch_size=batchSize, shuffle=False)	#shuffle not supported by DataloaderDatasetHDD

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

