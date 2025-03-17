"""TSBNLPpt_dataTokeniser.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see TSBNLPpt_main.py

# Usage:
see TSBNLPpt_main.py

# Description:
TSBNLPpt data tokeniser

"""

import torch
from TSBNLPpt_globalDefs import *
if(useFullwordTokenizer):
	import TSBNLPpt_dataTokeniserFullword
else:
	from tokenizers import ByteLevelBPETokenizer
	if(tokeniserOnlyTrainOnDictionary):
		from nltk.corpus import words
	if(useSubwordTokenizerFast):
		from transformers import RobertaTokenizerFast as RobertaTokenizer
	else:
		from transformers import RobertaTokenizer as RobertaTokenizer


def tokenise(lines, tokenizer, maxLength):			
	if(useFullwordTokenizerClass):
		return_offsets_mapping = False
		if(useSubwordTokenizerFast):
			return_offsets_mapping = True
		if(maxLength is None):
			sample = tokenizer(lines, return_tensors='pt', return_offsets_mapping=return_offsets_mapping)
		else:
			sample = tokenizer(lines, max_length=maxLength, padding='max_length', truncation=True, return_tensors='pt', return_offsets_mapping=return_offsets_mapping)
	else:
		sample = TSBNLPpt_dataTokeniserFullword.tokenizeBasic(lines, tokenizer)
	return sample

def trainTokeniser(dataElements, vocabSize):	
	if(useFullwordTokenizer):
		TSBNLPpt_dataTokeniserFullword.trainTokenizerFullwords(dataElements, vocabularySize)	#default method (vocabSize used by GIA word2vec model will be greater than numberOfTokens in tokenizer)
	else:
		trainTokeniserSubwords(dataElements, vocabularySize)
	
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
		#print("dataElements = ", dataElements)
		tokenizer.train(files=dataElements[:trainTokenizerNumberOfFilesToUse], vocab_size=vocabSize, min_frequency=1, special_tokens=specialTokens)
	else:
		tokenizer.train_from_iterator(dataset, length=trainTokenizerNumberOfFilesToUse, vocab_size=vocabSize, min_frequency=1, special_tokens=specialTokens)

	#os.mkdir(modelPathName)

	tokenizer.save_model(modelPathName)
		
	return tokenizer

def createDictionaryFile():
	dictionaryList = words.words() 
	print("len(dictionaryList) = ", len(dictionaryList))
	fileName = modelPathName + "/dictionary.txt"
	with open(fileName, 'w', encoding='utf-8') as fp:
		fp.write(' '.join(dictionaryList))
	return fileName
					
def loadTokeniser():
	if(usePretrainedRobertaTokenizer):
		tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
	else:	
		if(useFullwordTokenizer):
			tokenizer = TSBNLPpt_dataTokeniserFullword.loadTokenizerFullwords()
		else:
			tokenizer = loadTokenizerSubwords()
	
	return tokenizer
		
def loadTokenizerSubwords():	
	tokenizer = RobertaTokenizer.from_pretrained(modelPathName, max_len=sequenceMaxNumTokens)
	return tokenizer

def getTokenizerLength(tokenizer):
	return len(tokenizer)	#Size of the full vocabulary with the added token	#https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils.py

def printSpecialTokenIDs(tokenizer):
	#pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
	print("tokenizer.cls_token_id = ", tokenizer.cls_token_id)	#0 [CLS]
	print("tokenizer.pad_token_id = ", tokenizer.pad_token_id)	#1 [PAD]
	print("tokenizer.sep_token_id = ", tokenizer.sep_token_id)	#2 [SEP]
	print("tokenizer.unk_token_id = ", tokenizer.unk_token_id)	#3 [UNK]

	

#common data loader functions:

def getSampleEncodings(useMLM, input_ids, attention_mask, offset_mapping, batched):
	#print("input_ids = ", input_ids)
	#print("attention_mask = ", attention_mask)
	inputIDs = []
	mask = []
	labels = []
	if(legacyDataloaderCode2):
		labels.append(input_ids)
	else:
		labels.append(addLabelsPredictionMaskTokens(input_ids))
	mask.append(attention_mask)
	sampleInputIDs = (input_ids).detach().clone()
	if(useMaskedLM):
		if(batched):
			sampleInputIDsMasked = addMaskTokensBatch(useMLM, sampleInputIDs)
		else:
			sampleInputIDsMasked = addMaskTokensSample(useMLM, sampleInputIDs)
	else:
		sampleInputIDsMasked = sampleInputIDs
	inputIDs.append(sampleInputIDsMasked)
	inputIDs = torch.cat(inputIDs)
	mask = torch.cat(mask)
	if(multiTokenPrediction):
		labels = build_multi_token_labels_matrix(inputIDs, multiTokenPredictionNumFutureTokens)
	else:
		labels = torch.cat(labels)
	if(useSubwordTokenizerFast):
		offsets = []
		offsets.append(offset_mapping)
		offsets = torch.cat(offsets)
	else:
		offsets = torch.zeros_like(inputIDs)	#'None' is not supported	#not used
	encodings = {'inputIDs': inputIDs, 'attentionMask': mask, 'labels': labels, 'offsets': offsets}
	return encodings


def build_multi_token_labels_matrix(input_ids: torch.LongTensor, num_future_tokens: int) -> torch.LongTensor:
    """
    Create multi-token labels of shape (B, S, K) from input_ids (B, S).
    For each position t, store the tokens [t+1, t+2, ..., t+K],
    or -100 if out of range.
    """
    device = input_ids.device
    B, S = input_ids.size()
    
    # 1) Build positions[t, j] = t+1+j
    #    shape => (S, K)
    row_indices = torch.arange(S, device=device).unsqueeze(1)  # => [S, 1]
    col_offsets = torch.arange(num_future_tokens, device=device)  # => [K]
    positions = row_indices + 1 + col_offsets  # => [S, K]
    
    # 2) Identify out-of-range positions
    out_of_range_mask = positions >= S  # => [S, K]
    
    # 3) Clamp positions so we can index them safely. We'll set out-of-range to S-1
    positions_clamped = positions.clone()
    positions_clamped[out_of_range_mask] = S - 1
    
    # 4) Expand positions to shape (B, S, K) so we can index input_ids
    positions_clamped = positions_clamped.unsqueeze(0).expand(B, -1, -1) 
    # => [B, S, K]
    out_of_range_mask = out_of_range_mask.unsqueeze(0).expand(B, -1, -1)
    # => [B, S, K]
    
    # 5) Make a "batch index" array => shape [B, S, K]
    #    For each (b, t, j), we want to pick from input_ids[b, positions_clamped[b,t,j]]
    batch_arange = torch.arange(B, device=device).unsqueeze(1).unsqueeze(2)
    batch_arange = batch_arange.expand(-1, S, num_future_tokens)
    # => [B, S, K]
    
    # 6) Advanced indexing: 
    #    multi_labels[b, t, j] = input_ids[b, positions_clamped[b, t, j]]
    multi_labels = input_ids[batch_arange, positions_clamped]
    # => shape [B, S, K]
    
    # 7) Replace out-of-range positions with -100
    multi_labels[out_of_range_mask] = crossEntropyLossIgnoreIndex
    return multi_labels



def addLabelsPredictionMaskTokens(input_ids):
	mask_arr = (input_ids == paddingTokenID)
	mask_arr = mask_arr*(labelPredictionMaskTokenID-paddingTokenID)
	labels = input_ids + mask_arr
	#print("labels = ", labels)
	return labels
	
def addMaskTokensBatch(useMLM, inputIDs):
	for i in range(inputIDs.shape[0]):
		inputIDs[i] = addMaskTokensSample(useMLM, inputIDs[i])
	return inputIDs

def addMaskTokensSample(useMLM, inputIDs):
	if(useMLM):
		rand = torch.rand(inputIDs.shape)
		mask_arr = (rand < fractionOfMaskedTokens) * notSpecialTokensIDs(inputIDs)
	else:	
		mask_arr = notSpecialTokensIDs(inputIDs)
	selection = torch.flatten(mask_arr.nonzero()).tolist()
	inputIDs[selection] = customMaskTokenID
	return inputIDs
	
def generateAttentionMask(tokenizer, inputIDs):
	attention_mask = notSpecialTokensIDs(inputIDs).float()
	return attention_mask

def notSpecialTokensIDs(inputIDs):
	inputIDsNotSpecialTokens = (inputIDs > 2) #or (inputIDs != 0) * (inputIDs != 1) * (inputIDs != 2)	#inputIDs are not in [tokenizer.cls_token_id, tokenizer.pad_token_id, tokenizer.sep_token_id]
	return inputIDsNotSpecialTokens
	
def preprocessDocumentText(documentText):
	if(preprocessRemoveNewLineCharacters):
		documentText = documentText.replace('\n', '')
	return documentText
	
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
	
def generateDataFileName(fileIndex):
	fileName = dataPathName + dataPreprocessedFileNameStart + str(fileIndex) + dataPreprocessedFileNameEnd
	return fileName
