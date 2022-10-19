"""TSBpt-main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
conda create -n transformersenv
source activate transformersenv
conda install python=3.7	[DOESNTWORK: conda install python (python-3.10.6) because transformers not found]
pip install datasets
pip install transfomers==4.23.1
pip install torch

# Usage:
source activate transformersenv
python TSBpt-main.py

# Description:
TSBpt main - Transformer Syntactic Bias (TSB): trains a RoBERTa transformer with a number of syntactic inductive biases:
- sharedLayerWeights
	- Roberta number of layers = 6 supports approximately 2^6 words per sentence (contextual window = 512 tokens)

See RobertaForMaskedLM tutorial; https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6

"""

#user config vars:
useSmallDatasetDebug = False
useSmallTokenizerTrainNumberOfFilesDebug = True	#used during rapid testing only (FUTURE: assign est 80 hours to perform full tokenisation train)
useSmallDataloaderDebug = True	#used during rapid testing only (FUTURE: assign est 1 year to perform full dataset train)

statePreprocessDataset = False	#only required once
stateTrainTokenizer = False	#only required once
stateTrainDataset = True

startEpoch = 0	#start epoch of training (if continuing a training regime set accordingly >0)	#if startEpoch=0 will recreate model, if startEpoch>0 will load existing model
numberOfEpochs = 1	#default: 10	#number of epochs to train (for production typically train 1 epoch at a time)

batch_size = 8  #default: 16	#8 and 16 train at approx same rate (16 uses more GPU ram)	#depends on GPU RAM
learningRate = 1e-4
fractionOfMaskedTokens = 0.15

#dependent vars:
stateTestDataset = False
if(stateTrainDataset):
	stateTestDataset = True

if(useSmallDataloaderDebug):
	useSmallDataloaderDebug1 = False	#limit number of data files trained
	useSmallDataloaderDebug2 = True	#limit number of data files trained
	useSmallDataloaderDebug3 = False	#limit number of data files trained
	if(useSmallDataloaderDebug1):	
		loaderNumberOfDataFiles = 1	#mandatory
	elif(useSmallDataloaderDebug2):
		loaderNumberOfDataFiles = 50	#max ~50	#depends on system RAM (mem=50*35MBfile=~2GB, numberOfLoaderIterations=30424/50=600)
	elif(useSmallDataloaderDebug3):
		loaderNumberOfDataFiles = 1000	#1000	#unlimited (<= numberOfDataFiles*trainSplitFraction)

trainSplitFraction = 0.9	#90% train data, 10% test data

#storage location vars (requires 4TB harddrive);
downloadCacheFolder = '/media/user/datasets/cache'
dataFolder = '/media/user/datasets/data'

modelSaveNumberOfBatches = 1000	#resave model after x training batches

accuracyTopN = 1	#default: 1	#>= 1	#calculates batch accuracy based on top n dictionary predictions

from modeling_roberta_sharedLayerWeights import sharedLayerWeights
from datasets import load_dataset
from tqdm.auto import tqdm
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
import os
from transformers import RobertaTokenizer
import torch
from transformers import RobertaConfig
if(sharedLayerWeights):
	from modeling_roberta_sharedLayerWeights import RobertaForMaskedLM
else:
	from transformers import RobertaForMaskedLM
from transformers import AdamW
from transformers import pipeline

#torch.set_printoptions(threshold=10_000)
torch.set_printoptions(profile="full")

#store models to large datasets partition cache folder (not required)
#os.environ['TRANSFORMERS_CACHE'] = '/media/user/datasets/models/'	#select partition with 3TB+ disk space

transformerMaxNumTokens = 512
customMaskTokenID = 4	#3


def downloadDataset():
	if(useSmallDatasetDebug):
		dataset = load_dataset('nthngdy/oscar-small', 'unshuffled_original_en', cache_dir=downloadCacheFolder)	#unshuffled_deduplicated_en
	else:
		dataset = load_dataset('oscar', 'unshuffled_deduplicated_en', cache_dir=downloadCacheFolder)
	
	return dataset

def preprocessDataset(dataset):
	text_data = []
	file_count = 0

	for sample in tqdm(dataset['train']):
		sample = sample['text'].replace('\n', '')
		text_data.append(sample)
		if len(text_data) == 10_000:
			fileName = dataFolder + "/text_" + str(file_count) + ".txt"
			with open(fileName, 'w', encoding='utf-8') as fp:
				fp.write('\n'.join(text_data))
			text_data = []
			file_count += 1
	fileName = dataFolder + "/text_" + str(file_count) + ".txt"
	with open(fileName, 'w', encoding='utf-8') as fp:
		fp.write('\n'.join(text_data))

	
def trainTokenizer(paths):
	if(useSmallTokenizerTrainNumberOfFilesDebug):
		trainTokenizerNumberOfFilesToUse = 1000	#default 1000	#100: 15 min, 1000: 3.75 hours
	else:
		trainTokenizerNumberOfFilesToUse = len(paths)

	tokenizer = ByteLevelBPETokenizer()

	tokenizer.train(files=paths[:trainTokenizerNumberOfFilesToUse], vocab_size=30_522, min_frequency=2, special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

	os.mkdir('./oscarberto')

	tokenizer.save_model('oscarberto')
	
	tokenizer = loadTokenizer()	#CHECKTHIS
	
	return tokenizer

def loadTokenizer():	
	tokenizer = RobertaTokenizer.from_pretrained('oscarberto', max_len=transformerMaxNumTokens)
	return tokenizer

def addMaskTokens(input_ids):
	rand = torch.rand(input_ids.shape)
	mask_arr = (rand < fractionOfMaskedTokens) * (input_ids > 2)	#or * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
	for i in range(input_ids.shape[0]):
		selection = torch.flatten(mask_arr[i].nonzero()).tolist()
		input_ids[i, selection] = customMaskTokenID
	return input_ids

class DatasetHDD(torch.utils.data.Dataset):
	def __init__(self, dataFileIndexList, paths):
		self.dataFileIndexList = dataFileIndexList
		self.paths = paths

	def __len__(self):
		return len(self.dataFileIndexList)

	def __getitem__(self, i):
		dataFileIndex = self.dataFileIndexList[i]
		path = self.paths[dataFileIndex]
		with open(path, 'r', encoding='utf-8') as fp:
			lines = fp.read().split('\n')
			
		sample = tokenizer(lines, max_length=transformerMaxNumTokens, padding='max_length', truncation=True, return_tensors='pt')

		labels = sample.input_ids
		mask = sample.attention_mask
		input_ids = (sample.input_ids).detach().clone() 
		input_ids = addMaskTokens(input_ids)

		encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}

		return {key: tensor[i] for key, tensor in encodings.items()}

def createDataLoader(tokenizer, paths, splitTrain=True):
	numberOfDataFiles = len(paths)
	if(useSmallDataloaderDebug3):
		if(splitTrain):
			pathIndexMin = 0
			pathIndexMax = loaderNumberOfDataFiles
		else:
			pathIndexMin = int(numberOfDataFiles*trainSplitFraction)	#or numberOfDataFiles
			pathIndexMax = pathIndexMin+(loaderNumberOfDataFiles*trainSplitFraction)
	else:
		if(splitTrain):
			pathIndexMin = 0
			pathIndexMax = trainSplitFraction
		else:
			pathIndexMin = int(numberOfDataFiles*trainSplitFraction)
			pathIndexMax = -1
	
	dataFileIndexList = list(range(pathIndexMin, pathIndexMax))

	dataset = DatasetHDD(dataFileIndexList, paths)

	loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

	return loader
	
class DatasetRAM(torch.utils.data.Dataset):
	def __init__(self, encodings):
		self.encodings = encodings

	def __len__(self):
		return self.encodings['input_ids'].shape[0]

	def __getitem__(self, i):
		return {key: tensor[i] for key, tensor in self.encodings.items()}
		
def createDataLoaderDev1(tokenizer):	
	fileName = dataFolder + "/text_0.txt"
	with open(fileName, 'r', encoding='utf-8') as fp:
		lines = fp.read().split('\n')

	batch = tokenizer(lines, max_length=transformerMaxNumTokens, padding='max_length', truncation=True)
	
	labels = torch.tensor([x for x in batch['input_ids']])
	mask = torch.tensor([x for x in batch['attention_mask']])

	input_ids = labels.detach().clone()
	input_ids = addMaskTokens(input_ids)
	
	encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}

	dataset = DatasetRAM(encodings)

	loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

	return loader

def createDataLoaderDev2(tokenizer, paths, loaderIndex):	
	input_ids = []
	mask = []
	labels = []
	
	pathIndexMin = loaderIndex*loaderNumberOfDataFiles
	pathIndexMax = loaderIndex*loaderNumberOfDataFiles + loaderNumberOfDataFiles
	
	for path in tqdm(paths[pathIndexMin:pathIndexMax]):	
		with open(path, 'r', encoding='utf-8') as fp:
			lines = fp.read().split('\n')
		sample = tokenizer(lines, max_length=transformerMaxNumTokens, padding='max_length', truncation=True, return_tensors='pt')

		labels.append(sample.input_ids)
		mask.append(sample.attention_mask)
		sample_input_ids = (sample.input_ids).detach().clone() 
		#sample_input_ids = labels[-1].detach().clone()
		input_ids.append(addMaskTokens(sample_input_ids))
			
	input_ids = torch.cat(input_ids)
	mask = torch.cat(mask)
	labels = torch.cat(labels)

	input_ids.shape

	encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}

	dataset = DatasetRAM(encodings)

	loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

	return loader


def trainDataset(tokenizer, paths):

	if(startEpoch > 0):
		model = RobertaForMaskedLM.from_pretrained('./oscarberto', local_files_only=True)
	else:
		config = RobertaConfig(
			vocab_size=30_522,  #sync with tokenizer vocab_size
			max_position_embeddings=514,
			hidden_size=768,
			num_attention_heads=12,
			num_hidden_layers=6,
			type_vocab_size=1
		)
		model = RobertaForMaskedLM(config)

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	model.to(device)

	model.train()
	optim = AdamW(model.parameters(), lr=learningRate)
	
	numberOfDataFiles = len(paths)
	if(useSmallDataloaderDebug1):
		numberOfLoaderIterations = 1
		loader = createDataLoaderDev1(tokenizer)
	elif(useSmallDataloaderDebug2):
		loaderIndex = 0
		loader = createDataLoaderDev2(tokenizer, paths, loaderIndex)
	else:
		loader = createDataLoader(tokenizer, paths, True)
			
	for epoch in range(startEpoch, startEpoch+numberOfEpochs):
		loop = tqdm(loader, leave=True)
		for batchIndex, batch in enumerate(loop):
			optim.zero_grad()
			
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)
			
			outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
						
			accuracy = getAccuracy(tokenizer, input_ids, attention_mask, labels, outputs)
			loss = outputs.loss
			
			loss.backward()
			optim.step()

			loop.set_description(f'Epoch {epoch}')
			loop.set_postfix(batchIndex=batchIndex, loss=loss.item(), accuracy=accuracy)
		
			if(batchIndex % modelSaveNumberOfBatches == 0):
				model.save_pretrained('./oscarberto')
		model.save_pretrained('./oscarberto')

def testDataset(tokenizer, paths):

	model = RobertaForMaskedLM.from_pretrained('./oscarberto', local_files_only=True)

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	model.to(device)

	model.eval()
	
	numberOfDataFiles = len(paths)
	if(useSmallDataloaderDebug1):
		numberOfLoaderIterations = 1
		loader = createDataLoaderDev1(tokenizer)
	elif(useSmallDataloaderDebug2):
		loaderIndex = 0
		loader = createDataLoaderDev2(tokenizer, paths, loaderIndex)
	else:
		loader = createDataLoader(tokenizer, paths, False)
			
	for epoch in range(startEpoch, startEpoch+numberOfEpochs):
		loop = tqdm(loader, leave=True)
		for batchIndex, batch in enumerate(loop):
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)
			
			outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
			
			accuracy = getAccuracy(tokenizer, input_ids, attention_mask, labels, outputs)
			loss = outputs.loss

			loop.set_description(f'Epoch {epoch}')
			loop.set_postfix(batchIndex=batchIndex, loss=loss.item(), accuracy=accuracy)

def getTokenizerLength(tokenizer):
	return len(tokenizer)	#Size of the full vocabulary with the added token	#https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils.py

def getAccuracy(tokenizer, input_ids, attention_mask, labels, outputs):
	tokenizerNumberTokens = getTokenizerLength(tokenizer)
	
	tokenLogits = (outputs.logits).detach()

	tokenLogitsTopIndex = torch.topk(tokenLogits, accuracyTopN).indices	#get highest n scored entries from dictionary	#tokenLogitsTopIndex.shape = batchSize, transformerMaxNumTokens, accuracyTopN
	
	maskTokenIndex = torch.where(input_ids==customMaskTokenID, 1.0, 0.0)	#maskTokenIndexFloat = maskTokenIndex.float()	

	if(accuracyTopN == 1):
		tokenLogitsTopIndex = torch.squeeze(tokenLogitsTopIndex)	#tokenLogitsTopIndex[:, :, 1] -> #tokenLogitsTopIndex[:, :] 	

		comparison = (tokenLogitsTopIndex == labels).float()
		comparisonMasked = torch.multiply(comparison, maskTokenIndex)
		accuracy = (torch.sum(comparisonMasked)/torch.sum(maskTokenIndex)).cpu().numpy() 
	else:
		comparison = (tokenLogitsTopIndex == labels).float()	#labels broadcasted to [batchSize, transformerMaxNumTokens, accuracyTopN]
		comparisonMasked = torch.multiply(comparison, maskTokenIndex)	#maskTokenIndex broadcasted to [batchSize, transformerMaxNumTokens, accuracyTopN]
		accuracy = (torch.sum(comparisonMasked)/torch.sum(maskTokenIndex)).cpu().numpy() 	
	
	#accuracy2 = (torch.mean(comparisonMasked)).cpu().numpy()
	
	return accuracy
	
def testWordCompletion():
	fill = pipeline('fill-mask', model='oscarberto', tokenizer='oscarberto')

	fill(f'Hi {fill.tokenizer.mask_token} are you?')
	fill(f'good day, {fill.tokenizer.mask_token} are you?')
	fill(f'hi, where are we {fill.tokenizer.mask_token} to meet this afternoon? ')
	fill(f'what would have happened if {fill.tokenizer.mask_token} had chosen another day?')

if(__name__ == '__main__'):
	if(statePreprocessDataset):
		dataset = downloadDataset()
		preprocessDataset(dataset)
	paths = [str(x) for x in Path(dataFolder).glob('**/*.txt')]
	if(stateTrainTokenizer):
		if(startEpoch > 0):
			tokenizer = loadTokenizer()
		else:
			tokenizer = trainTokenizer(paths)
	else:
		tokenizer = loadTokenizer()
	if(stateTrainDataset):
		trainDataset(tokenizer, paths)
	if(stateTestDataset):
		testWordCompletion()

