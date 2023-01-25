"""SBNLPpt_tokeniserFullword

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SBNLPpt_main.py

# Usage:
see SBNLPpt_main.py

# Description:
SBNLPpt tokeniser fullword
	
"""

import torch as pt
import torch

from SBNLPpt_globalDefs import *

if(tokeniserOnlyTrainOnDictionary):
	from nltk.corpus import words as NLTKwords
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

class TokenizerBasicOutputEmulate():
	def __init__(self, inputID, maskID):
		self.input_ids = inputID
		self.attention_mask = maskID	#artificial mask; padding and mask tokens only

class TokenizerBasic():
	def __init__(self):
		self.dict = {}
		self.list = []
	def tokenize(self, word):
		if(word in self.dict.keys()):
			tokenID = self.dict[word]
		else:
			tokenID = vocabularySize-1	#assumes vocabularySize (model one-hot input size) > number of tokens in tokeniser
		return tokenID

def tokenizeBasic(lines, tokenizer):
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
			if(paddingLength > 0):
				maskPadding = [specialTokenPadding]*paddingLength
				inputs = inputTokens + maskPadding
				mask = maskTokens + maskPadding
			else:
				inputs = inputTokens
				mask = maskTokens
			inputID = [tokenizer.tokenize(x) for x in inputs]
			maskID = [tokenizer.tokenize(x) for x in mask]
			tokenIDtensor = torch.Tensor(inputID).to(torch.long)	#dtype=torch.int32
			maskIDtensor = torch.Tensor(maskID).to(torch.long)	#dtype=torch.int32
			inputIDlist.append(tokenIDtensor)
			maskIDlist.append(maskIDtensor)
		inputID = torch.stack(inputIDlist)
		maskID = torch.stack(maskIDlist)
		sample = TokenizerBasicOutputEmulate(inputID, maskID)
	return sample

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
	if(tokeniserOnlyTrainOnDictionary):
		tokensSet = trainTokenizerFullwordsDictionary()
	else:
		 tokensSet = trainTokenizerFullwordsDatafiles(paths)
		 
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
			
	return tokenizer

def trainTokenizerFullwordsDictionary(tokensSet):
	posWordListAll = getAllNLTKwords()
	tokensSet = set(posWordListAll)
	return tokensSet
	
def trainTokenizerFullwordsDatafiles(paths):
	tokensSet = set()
	for dataFileIndex in range(trainTokenizerNumberOfFilesToUse):
		path = paths[dataFileIndex]
		print("dataFileIndex = ", dataFileIndex)
		with open(path, 'r', encoding='utf-8') as fp:
			lines = fp.read().split('\n')
			for lineIndex, line in enumerate(lines):
				tokens = fullwordTokenizeLine(line)
				tokensSet.update(tokens)
	return tokensSet
		
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
			tokensVocabDictionaryItems = SBNLPpt_GIAdefinePOSwordLists.createDictionaryItemsFromList(tokensVocab, 0)
			tokenizer.dict = dict(tokensVocabDictionaryItems)
			tokensSpecialDictionaryItems = SBNLPpt_GIAdefinePOSwordLists.createDictionaryItemsFromList(tokensSpecial, len(tokenizer.dict))
			for i, j in tokensSpecialDictionaryItems:
				tokenizer.dict[i] = j
			keysList = list(tokenizer.dict.keys())
			if(len(keysList) < vocabularySize):
				blankList = list(range(0, vocabularySize-len(keysList)))
				keysList = keysList + blankList
			tokenizer.list = keysList
	return tokenizer
