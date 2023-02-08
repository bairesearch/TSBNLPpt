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
import SBNLPpt_dataTokeniser
from SBNLPpt_globalDefs import *

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
	attention_mask = SBNLPpt_dataTokeniser.generateAttentionMask(tokenizer, input_ids)
	encodings = SBNLPpt_dataTokeniser.getSampleEncodings(useMLM, input_ids, attention_mask, False)
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
		documentText = SBNLPpt_dataTokeniser.getNextDocument(datasetIterator)
		documentText = SBNLPpt_dataTokeniser.preprocessDocumentText(documentText)
		documentIndex+=1
		if(sampleIndex == batchSize):
			stillFindingDocumentSegmentSamples = False
		if(len(documentText) > orderedDatasetDocMinSizeCharacters):
			documentTokens = SBNLPpt_dataTokeniser.tokenise(documentText, tokenizer, None)
			documentTokensIDs = documentTokens.input_ids[0]
			sampleIndex = splitDocumentIntoSegments(documentTokensIDs, documentSegmentsSampleList, sampleIndex)
		if(usePreprocessedDataset):
			if(documentIndex == numberOfDocumentsPerDataFile):
				reachedEndOfDataset = True
				stillFindingDocumentSegmentSamples = False
				while sampleIndex < batchSize:
					#fill remaining documentSegmentsSampleList rows with pad_token_id	#FUTURE implementation; load next data file
					documentTokensIDsIgnore = torch.full([sequenceMaxNumTokens*orderedDatasetDocNumberSamples], fill_value=tokenizer.pad_token_id, dtype=torch.long)
					sampleIndex = splitDocumentIntoSegments(documentTokensIDsIgnore, documentSegmentsSampleList, sampleIndex)

	documentSegmentsBatchList = list(map(list, zip(*documentSegmentsSampleList)))	#transpose list of lists: batchSize*numberOfDocumentSegments -> numberOfDocumentSegments*batchSize
	#printDocumentSegments(tokenizer, documentSegmentsBatchList)
		
	return documentSegmentsBatchList, documentIndex, reachedEndOfDataset
			
def splitDocumentIntoSegments(documentTokensIDs, documentSegmentsSampleList, sampleIndex):
	if(orderedDatasetSplitDocumentsBySentences):
		print("splitDocumentIntoSegments error: orderedDatasetSplitDocumentsBySentences not yet coded")
		exit()
	else:
		if(documentTokensIDs.shape[0] >= orderedDatasetDocNumberTokens):
			documentTokensIDs = documentTokensIDs[0:orderedDatasetDocNumberTokens]
			#documentSegments = [documentTokensIDs[x:x+sequenceMaxNumTokens] for x in xrange(0, len(documentTokensIDs), sequenceMaxNumTokens)]
			documentSegments = torch.split(documentTokensIDs, split_size_or_sections=sequenceMaxNumTokens, dim=0)
			documentSegmentsSampleList.append(documentSegments)
			sampleIndex+=1
	return sampleIndex
	
def printDocumentSegments(tokenizer, documentSegmentsBatchList):
	for segmentIndex1 in range(orderedDatasetDocNumberSamples):
		print("segmentIndex1 = ", segmentIndex1)
		for sampleIndex in range(batchSize):
			print("sampleIndex = ", sampleIndex)
			sample_ids = documentSegmentsBatchList[segmentIndex1][sampleIndex]
			sampleString = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(sample_ids))
			print("sample = ", sampleString)

def createDatasetLargeDocuments(tokenizer, dataElements):

	paths = dataElements
	pathIndexMin = trainStartDataFile
	pathIndexMax = pathIndexMin+int(trainNumberOfDataFiles*dataFilesFeedMultiplier)
	
	dataFileIndexList = list(range(pathIndexMin, pathIndexMax))
	for dataFileIndex in dataFileIndexList:
		path = paths[dataFileIndex]
		with open(path, 'r', encoding='utf-8') as fp:
			lines = fp.read().split('\n')

		linesLargeDocuments = []
		for documentIndex, documentText in enumerate(lines):
			if(len(documentText) > orderedDatasetDocMinSizeCharacters):
				documentTokens = SBNLPpt_dataTokeniser.tokenise(documentText, tokenizer, None)
				documentTokensIDs = documentTokens.input_ids[0]
				if(documentTokensIDs.shape[0] >= orderedDatasetDocNumberTokens):
					linesLargeDocuments.append(documentText)
				else:
					linesLargeDocuments.append("SMALL_DOCUMENT_PLACER")
			else:
				linesLargeDocuments.append("SMALL_DOCUMENT_PLACER")

		pathLargeDocuments = path.replace('/'+dataFolderName+'/', '/'+dataFolderNameLargeDocuments+'/')
		print("pathLargeDocuments = ", pathLargeDocuments)
		with open(pathLargeDocuments, 'w', encoding='utf-8') as fp:
			fp.write('\n'.join(linesLargeDocuments))

	exit()

