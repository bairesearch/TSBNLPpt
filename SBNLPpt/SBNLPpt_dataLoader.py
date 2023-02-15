"""SBNLPpt_dataLoader.py

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
from SBNLPpt_globalDefs import *
if(createOrderedDataset):
	import SBNLPpt_dataLoaderOrdered
import SBNLPpt_dataTokeniser

def createDataLoader(useMLM, tokenizer, dataElements, numberOfDataFiles, pathIndexMin, pathIndexMax):	
	if(usePreprocessedDataset):
		dataFileIndexList = list(range(pathIndexMin, pathIndexMax))
		print("dataFileIndexList = ", dataFileIndexList)
		if(debugPrintPaths):
			print("paths = ", dataElements[0:pathIndexMax-pathIndexMin])	
	
	if(createOrderedDataset):
		numberOfDocuments = numberOfDataFiles*numberOfDocumentsPerDataFile * orderedDatasetDocNumberSegments//10    #//10 to reduce number datafiles parsed (not required)
	else:
		numberOfDocuments = numberOfDataFiles*numberOfDocumentsPerDataFile	#equivalent number of documents (assuming it were loading data files)
	print("numberOfDocuments = ", numberOfDocuments)

	if(usePreprocessedDataset):
		dataLoaderDataset = DataloaderDatasetHDD(useMLM, numberOfDocuments, dataFileIndexList, dataElements, tokenizer)
	else:
		dataLoaderDataset = DataloaderDatasetInternet(useMLM, numberOfDocuments, dataElements, tokenizer)	

	loader = torch.utils.data.DataLoader(dataLoaderDataset, batch_size=batchSize, shuffle=False)	#shuffle not supported by DataloaderDatasetHDD

	return loader	
	
class DataloaderDatasetHDD(torch.utils.data.Dataset):
	def __init__(self, useMLM, numberOfDocumentsEst, dataFileIndexList, dataElements, tokenizer):
		self.useMLM = useMLM
		self.dataFileIndexList = dataFileIndexList
		self.paths = dataElements
		self.tokenizer = tokenizer
		self.numberOfDocuments = getNumberOfDocumentsHDD(numberOfDocumentsEst, dataFileIndexList)
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
			if(debugPrintDataFileIndex):
				print("loadNextDataFile: dataFileIndex = ", dataFileIndex)
			path = SBNLPpt_dataTokeniser.generateDataFileName(dataFileIndex)	#OLD: self.paths[dataFileIndex] - requires dataPathName to contain all dataFiles up to dataFileIndex
			with open(path, 'r', encoding='utf-8') as fp:
				lines = fp.read().split('\n')
			if(createOrderedDataset):
				self.dataFileIndexRelative += 1
				self.documentIndexInDataFile = 0
				self.segmentIndexInDocument = 0
				self.sampleIndexInBatch = 0
				self.dataFileLinesIterator = iter(lines)
			else:
				sample = SBNLPpt_dataTokeniser.tokenise(lines, self.tokenizer, sequenceMaxNumTokens)
				self.encodings = SBNLPpt_dataTokeniser.getSampleEncodings(self.useMLM, sample.input_ids, sample.attention_mask, True)

		if(createOrderedDataset):
			batchSample, self.documentSegmentsBatchList, self.documentIndexInDataFile, self.sampleIndexInBatch, self.segmentIndexInDocument = SBNLPpt_dataLoaderOrdered.getOrderedBatchSample(
				self.documentSegmentsBatchList, self.documentIndexInDataFile, self.sampleIndexInBatch, self.segmentIndexInDocument, self.tokenizer, self.dataFileLinesIterator, self.useMLM
			)
		else:		
			batchSample = {key: tensor[self.documentIndexInDataFile] for key, tensor in self.encodings.items()}	
		
		return batchSample

def getNumberOfDocumentsHDD(numberOfDocumentsEst, dataFileIndexList):
	if(createOrderedDataset):
		numberOfDocuments = numberOfDocumentsEst
	else:
		containsDataFileLastDocument = dataFileIndexListContainsLastDocument(dataFileIndexList)
		numberOfDocuments = len(dataFileIndexList)*numberOfDocumentsPerDataFile
		if(containsDataFileLastDocument):
			numberOfDocuments = numberOfDocuments-numberOfDocumentsPerDataFile + datasetNumberOfSamplesPerDataFileLast
	return numberOfDocuments

def dataFileIndexListContainsLastDocument(dataFileIndexList):
	containsDataFileLastDocument = False
	for dataFileIndex in dataFileIndexList:
		if(str(dataFileIndex) == dataFileLastIndex):
			containsDataFileLastDocument = True
	return containsDataFileLastDocument
				
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
			batchSample, self.documentSegmentsBatchList, self.documentIndex, self.sampleIndexInBatch, self.segmentIndexInDocument = SBNLPpt_dataLoaderOrdered.getOrderedBatchSample(
				self.documentSegmentsBatchList, self.documentIndex, self.sampleIndexInBatch, self.segmentIndexInDocument, self.tokenizer, self.datasetIterator, self.useMLM
			)
		else:
			documentText = SBNLPpt_dataTokeniser.getNextDocument(self.datasetIterator)
			documentText = SBNLPpt_dataTokeniser.preprocessDocumentText(documentText)
			documentTokens = SBNLPpt_dataTokeniser.tokenise(documentText, self.tokenizer, sequenceMaxNumTokens)
			encodings = SBNLPpt_dataTokeniser.getSampleEncodings(self.useMLM, documentTokens.input_ids[0], documentTokens.attention_mask[0], False)
			batchSample = encodings
			self.documentIndex+=1
			
		return batchSample
		
