"""GPT2pt.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023-2024 Baxter AI (baxterai.com)
	
based on; https://huggingface.co/learn/nlp-course/chapter7/6
https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter7/section6_pt.ipynb

# License:
MIT License

# Installation:
conda create -n transformersenvGPT2
source activate transformersenvGPT2
conda install python=3.7
pip install datasets
pip install 'transformers>=4.23'
pip install torch
pip install lovely-tensors
pip install nltk
pip install torchmetrics
pip install pynvml
pip install accelerate
pip install evaluate

# Usage:
source activate transformersenvGPT2
python GPT2pt.py

# Description:
GPT2 train with codeparrot dataset (local)
Training a causal language model from scratch (PyTorch)

"""

from transformers import AutoTokenizer, AutoConfig
usePretrainedModelDebug = False
if(usePretrainedModelDebug):
	from transformers import GPT2LMHeadModel
else:
	from modeling_gpt2 import GPT2LMHeadModel
from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset
from datasets import load_dataset
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from torch.nn import CrossEntropyLoss
import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler

from modeling_gpt2 import centralSequencePrediction
if(centralSequencePrediction):
	import nltk
	from nltk.tokenize import sent_tokenize
	nltk.download('punkt')
	from tokenizers import ByteLevelBPETokenizer
	from modeling_gpt2 import maxConclusionLength
	from modeling_gpt2 import maxIntroLength
	from modeling_gpt2 import maxCentralLength
	from modeling_gpt2 import maxIntroCentralLength
	centralSequencePredictionConclusionEndToken = '<conclusion_end>'
	centralSequencePredictionIntroStartToken = '<intro_start>'

if(centralSequencePrediction):
	datasetName = 'wikitext'
else:
	datasetName = 'wikitext'	#'codeparrot'
if(datasetName=='wikitext'):
	model_name = "wikitext-accelerate"
	sequenceStartToken = '<s>'	#tokenizer.bos_token
	sequenceEndToken = '</s>'	#tokenizer.eos_token
elif(datasetName=='codeparrot'):
	model_name = "codeparrot-ds-accelerate"


#debug options;
trainPartialDataset = False
if(trainPartialDataset):
	trainPartialDatasetTrainSamples = 5000	#max 606720	#50000
	trainPartialDatasetValidSamples = 3322 #max 3322	#500	#50	
trainWithTrainer = False	#also trains using Trainer API: no eval performance output
if(trainWithTrainer):
	trainLoadFromCheckpoint = False
else:
	trainLoadFromPrevious = False	#optional (continue training) - incomplete (requires dataset training samples sync)
	
stateTrainDataset = True
stateTestDataset = False	#evaluate an existing model (do not train)

#configuration options;
num_train_epochs = 1	#default: 1	#10 - measure max training performance over multiple epochs
shuffleTrainDataset = False	#default: False #orig (< 24 June 2023): True  #False is used for comparative tests
saveTrainedModel = True	#save final model after completing 100% train
batchSize = 4	#16	#default: 16	#orig: 32	#lower batch size for simultaneous testing (ram usage)
numberOfHiddenLayers = 12	#default = 12	#12	#1
from modeling_gpt2 import recursiveLayers
numberOfAttentionHeads = 12
hiddenLayerSizeTransformer = 768
intermediateSizeTransformer = hiddenLayerSizeTransformer*4	#default GPT2 specification	#3072
if(recursiveLayers):
	from modeling_gpt2 import sharedLayerWeightsMLPonly
	from modeling_gpt2 import transformerBlockMLPlayer
	recursiveLayersNormaliseNumParameters = False	#optional
	if(recursiveLayersNormaliseNumParameters):
		recursiveLayersNormaliseNumParametersIntermediateOnly = False	#optional	#only normalise intermediary MLP layer
		if(not transformerBlockMLPlayer):
			numberOfAttentionHeads = 32
			hiddenLayerSizeTransformer = 2048	#model size = 463MB
		elif(recursiveLayersNormaliseNumParametersIntermediateOnly):
			#requires high GPU memory ~24GB (although parameter size is equivalent, memory size is much higher for the purposes of storing gradient calculation data in every layer) 	#trial only; batchSize = 4	 
			if(sharedLayerWeightsMLPonly):
				intermediateLayerSizeMultiplier = 12	#model size = 477MB	#hiddenLayerSize 768, intermediateSize 36864
			else:
				intermediateLayerSizeMultiplier = 18		#model size = 486MB	#hiddenLayerSize 768, intermediateSize 55296
			intermediateSizeTransformer *= intermediateLayerSizeMultiplier
		else:
			recursiveLayersNormaliseNumParametersIntermediate = True	#implied true as intermediate_size = 4*hidden_size
			if(sharedLayerWeightsMLPonly):	
				numberOfAttentionHeads = 18	#16	#20
				hiddenLayerSizeTransformer = 1152	#1024	#1280	#model size = 510MB
			else:
				numberOfAttentionHeads = 28	#24	#32
				hiddenLayerSizeTransformer = 1792	#1536	#2048	#model size = 496MB
	else:
		pass
		#model size = 176MB
else:
	pass
	#model size = 486MB
	
printAccuracy = True
if(printAccuracy):
	import math 
	accuracyTopN = 1	#default: 1	#>= 1	#calculates batch accuracy based on top n dictionary predictions
	
def any_keyword_in_string(string, keywords):
	for keyword in keywords:
		if keyword in string:
			return True
	return False


def filter_streaming_dataset(dataset, filters):
	filtered_dict = defaultdict(list)
	total = 0
	for sample in tqdm(iter(dataset)):
		total += 1
		if any_keyword_in_string(sample["content"], filters):
			for k, v in sample.items():
				filtered_dict[k].append(v)
	print(f"{len(filtered_dict['content'])/total:.2%} of data after filtering.")
	return Dataset.from_dict(filtered_dict)

split = "train"  # "valid"
filters = ["pandas", "sklearn", "matplotlib", "seaborn"]


print("start load dataset")
if(datasetName=='wikitext'):
	ds_train = load_dataset(path="wikitext", name="wikitext-2-raw-v1", split="train", ignore_verifications=True)
	ds_valid = load_dataset(path="wikitext", name="wikitext-2-raw-v1", split="validation", ignore_verifications=True)
elif(datasetName=='codeparrot'):
	ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
	ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")
print("end load dataset")

print("Number of rows in the ds_train:", ds_train.num_rows)
print("Number of rows in the ds_valid:", ds_valid.num_rows)

if(trainPartialDataset):
	#if(shuffleTrainDataset): consider restoring "ds_.shuffle().select"
	raw_datasets = DatasetDict(
		{
			"train": ds_train.select(range(trainPartialDatasetTrainSamples)),
			"valid": ds_valid.select(range(trainPartialDatasetValidSamples))
		}
	)
else:
	raw_datasets = DatasetDict(
		{
			"train": ds_train,
			"valid": ds_valid,
		}
	)
	
if(datasetName=='wikitext'):
	context_length = 512
	context_length_min = 256	#ignore text that does not have sufficient number of sentences
	tokenizer = AutoTokenizer.from_pretrained("roberta-base")
	if(centralSequencePrediction):
		new_special_tokens = [centralSequencePredictionConclusionEndToken, centralSequencePredictionIntroStartToken]
		tokenizer.add_tokens(new_special_tokens)
elif(datasetName=='codeparrot'):
	context_length = 128
	tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")


def tokenize(element):
	if(datasetName=='wikitext'):
		contentRaw = element["text"]
		content = []
		for element in contentRaw:
			if len(element) > context_length_min:
				content.append(element)
		outputs = tokenizer(content, max_length=context_length, padding='max_length', truncation=True, return_tensors='pt')
		tokensList = []
		for input_ids in outputs["input_ids"]:
			tokens = tokenizer.convert_ids_to_tokens(input_ids)
			tokensList.append(tokens)
		batch = {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"], "tokens": tokensList, "content": content}
		return batch
	elif(datasetName=='codeparrot'):
		content = element["content"]
		outputs = tokenizer(content, truncation=True, max_length=context_length, return_overflowing_tokens=True, return_length=True)
		input_batch = []
		for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
			if length == context_length:	#only perform prediction for sufficient context_length (no padding)
				input_batch.append(input_ids)
		batch = {"input_ids": input_batch}
		return batch
		
if(datasetName=='wikitext'):
	if(centralSequencePrediction):
		sequenceStartTokenID = tokenizer.convert_tokens_to_ids(sequenceStartToken)
		centralSequencePredictionConclusionEndTokenID = tokenizer.convert_tokens_to_ids(centralSequencePredictionConclusionEndToken)
		centralSequencePredictionIntroStartTokenID = tokenizer.convert_tokens_to_ids(centralSequencePredictionIntroStartToken)
		sequenceEndTokenID = tokenizer.convert_tokens_to_ids(sequenceEndToken)
		pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
	tokenized_datasets = raw_datasets.map(tokenize, batched=True, remove_columns=raw_datasets["train"].column_names)
elif(datasetName=='codeparrot'):
	tokenized_datasets = raw_datasets.map(tokenize, batched=True, remove_columns=raw_datasets["train"].column_names)
	
config = AutoConfig.from_pretrained(
	"gpt2",
	vocab_size=len(tokenizer),
	n_ctx=context_length,
	bos_token_id=tokenizer.bos_token_id,
	eos_token_id=tokenizer.eos_token_id,
	num_hidden_layers=numberOfHiddenLayers,
	num_attention_heads=numberOfAttentionHeads,
	hidden_size=hiddenLayerSizeTransformer,
	n_inner=intermediateSizeTransformer
)

tokenizer.pad_token = tokenizer.eos_token


if(trainWithTrainer):
	if(trainLoadFromCheckpoint):
		model = GPT2LMHeadModel.from_pretrained("./codeparrot-ds/checkpoint-5000")
	else:
		model = GPT2LMHeadModel(config)
	model_size = sum(t.numel() for t in model.parameters())
	print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

	data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

	'''
	out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
	for key in out:
		print(f"{key} shape: {out[key].shape}")
	'''
	
	args = TrainingArguments(
		output_dir="codeparrot-ds",
		per_device_train_batch_size=batchSize,
		per_device_eval_batch_size=batchSize,
		evaluation_strategy="steps",
		eval_steps=5_000,
		logging_steps=5_000,
		gradient_accumulation_steps=8,
		num_train_epochs=num_train_epochs,
		weight_decay=0.1,
		warmup_steps=1_000,
		lr_scheduler_type="cosine",
		learning_rate=5e-4,
		save_steps=5_000,
		fp16=True,
		push_to_hub=False,
	)

	trainer = Trainer(
		model=model,
		tokenizer=tokenizer,
		args=args,
		data_collator=data_collator,
		train_dataset=tokenized_datasets["train"],
		eval_dataset=tokenized_datasets["valid"],
	)


	print("start train")
	trainer.train()
	print("end train")
	if(saveTrainedModel):
		output_dir = "codeparrot-ds/checkpoint-final"
		model.save_pretrained(output_dir)
		tokenizer.save_pretrained(output_dir)
		
keytoken_ids = []
if(datasetName=='codeparrot'):
	for keyword in [
		"plt",
		"pd",
		"sk",
		"fit",
		"predict",
		" plt",
		" pd",
		" sk",
		" fit",
		" predict",
		"testtest",
	]:
		ids = tokenizer([keyword]).input_ids[0]
		if len(ids) == 1:
			keytoken_ids.append(ids[0])
		else:
			print(f"Keyword has not single token: {keyword}")

def keytoken_weighted_loss(inputs, logits, keytoken_ids, alpha=1.0, attention_mask=None):
	if(datasetName=='wikitext'):
		predictionMask = attention_mask[:, 1:].reshape(-1)
	
	# Shift so that tokens < n predict n
	shift_labels = inputs[..., 1:].contiguous()
	shift_logits = logits[..., :-1, :].contiguous()
	# Calculate per-token loss
	loss_fct = CrossEntropyLoss(reduce=False)
	loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
	if(datasetName=='wikitext'):
		#print("loss.shape = ", loss.shape)
		#print("predictionMask.shape = ", predictionMask.shape)
		loss = torch.multiply(loss, predictionMask)	#set loss to 0 where attention mask==0
		#print("loss = ", loss)
	# Resize and average loss per sample
	loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
	# Calculate and scale weighting
	if(datasetName=='codeparrot'):
		#print("keytoken_ids = ", keytoken_ids)
		#print("inputs = ", inputs)
		weights = torch.stack([(inputs == kt).float() for kt in keytoken_ids]).sum(axis=[0, 2])
		weights = alpha * (1.0 + weights)
	else:
		weights = 1
	# Calculate weighted average
	weighted_loss = (loss_per_sample * weights).mean()
	return weighted_loss

def getAccuracy(inputs, logits, attention_mask=None):	
	if(datasetName=='wikitext'):
		predictionMask = attention_mask[:, 1:]
		
	#based on SBNLPpt_data:getAccuracy
	logits = logits.detach()
	# Shift so that tokens < n predict n
	shift_labels = inputs[..., 1:].contiguous()
	shift_logits = logits[..., :-1, :].contiguous()
	tokenLogitsTopIndex = torch.topk(shift_logits, accuracyTopN).indices	#get highest n scored entries from dictionary	#tokenLogitsTopIndex.shape = batchSize, sequenceMaxNumTokens, accuracyTopN
	if(accuracyTopN == 1):
		tokenLogitsTopIndex = torch.squeeze(tokenLogitsTopIndex)	#tokenLogitsTopIndex[:, :, 1] -> #tokenLogitsTopIndex[:, :] 	
		comparison = (tokenLogitsTopIndex == shift_labels).float()
		if(datasetName=='wikitext'):
			comparisonMasked = torch.multiply(comparison, predictionMask)
			accuracy = (torch.sum(comparisonMasked)/torch.sum(predictionMask)).cpu().numpy()	#accuracy.item()		
		else:
			accuracy = torch.mean(comparison)
	else:
		labelsExpanded = torch.unsqueeze(shift_labels, dim=2)
		labelsExpanded = labelsExpanded.expand(-1, -1, tokenLogitsTopIndex.shape[2])	#labels broadcasted to [batchSize, sequenceMaxNumTokens, accuracyTopN]
		comparison = (tokenLogitsTopIndex == labelsExpanded).float()
		if(datasetName=='wikitext'):
			predictionMaskExpanded = torch.unsqueeze(predictionMask, dim=2)
			predictionMaskExpanded = predictionMaskExpanded.expand(-1, -1, tokenLogitsTopIndex.shape[2])	#predictionMask broadcasted to [batchSize, sequenceMaxNumTokens, accuracyTopN]
			comparisonMasked = torch.multiply(comparison, predictionMaskExpanded)	#predictionMask broadcasted to [batchSize, sequenceMaxNumTokens, accuracyTopN]
			accuracy = (torch.sum(comparisonMasked)/torch.sum(predictionMask)).cpu().numpy() 	#or torch.sum(comparisonMasked)/(torch.sum(predictionMaskExpanded)/accuracyTopN)	#accuracy.item()
		else:
			accuracy = torch.mean(comparison)
	#print("accuracy = ", accuracy)
	return accuracy


def batchLists(sample):
	numberLines = len(sample)
	input_ids_batch = []
	attention_mask_batch = []
	tokens_batch = []
	content_batch = []
	for lineIndex in range(numberLines):
		input_ids_batch.append(sample[lineIndex]["input_ids"])
		attention_mask_batch.append(sample[lineIndex]["attention_mask"])
		tokens_batch.append(sample[lineIndex]["tokens"])
		content_batch.append(sample[lineIndex]["content"])
	input_ids_batch = torch.stack(input_ids_batch, dim=0)
	attention_mask_batch = torch.stack(attention_mask_batch, dim=0)
	#tokens_batch = torch.stack(tokens_batch, dim=0)	#not used
	#content_batch = torch.stack(content_batch, dim=0)	#not used
	batch = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch, "tokens": tokens_batch, "content": content_batch}
	return batch
		
def resizeSubtensor(tokens, maxLength, padTokenID):
	if(tokens.shape[0] > maxLength):
		tokens = tokens[0:maxLength]
	if(tokens.shape[0] < maxLength):
		paddingLength = maxLength-tokens.shape[0]
		tokensPadding = torch.full((paddingLength,), padTokenID, dtype=torch.long)
		tokens = torch.cat((tokens, tokensPadding), dim=0)
	return tokens

def centralSequencePredictionCollateFunction(sample):
	numberLines = len(sample)
	#print("numberLines = ", numberLines)

	tokensNewList = []
	attentionMaskNewList = []
	for lineIndex in range(numberLines):
		line = sample[lineIndex]["content"]
		tokens = sample[lineIndex]["tokens"]
		token_ids = sample[lineIndex]["input_ids"]
		attention_mask = sample[lineIndex]["attention_mask"]
		token_offsets = []
		current_offset = 0
		i = 0
		#print("line = ", line)
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
		
		#print("token_offsets = ", token_offsets)
		
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
		
		#print("token_ids = ", token_ids)
		#print("conclusionFirstTokenIndex = ", conclusionFirstTokenIndex)
		#print("conclusionLastTokenIndex = ", conclusionLastTokenIndex)
		
		tokenidsConclusion = torch.concat((torch.tensor([sequenceStartTokenID]), 
			token_ids[conclusionFirstTokenIndex:conclusionLastTokenIndex+1], 
			torch.tensor([centralSequencePredictionConclusionEndTokenID])), dim=0)
		tokenidsConclusion = resizeSubtensor(tokenidsConclusion, maxConclusionLength, pad_token_id)
		tokenidsIntroCentral = torch.concat((torch.tensor([centralSequencePredictionIntroStartTokenID]), 
			token_ids[introFirstTokenIndex:conclusionFirstTokenIndex], 
			torch.tensor([sequenceEndTokenID])), dim=0)
		tokenidsIntroCentral = resizeSubtensor(tokenidsIntroCentral, maxIntroCentralLength, pad_token_id)

		attentionMaskConclusion = torch.concat((torch.ones(1), attention_mask[conclusionFirstTokenIndex:conclusionLastTokenIndex+1], torch.ones(1)), dim=0)
		attentionMaskConclusion = resizeSubtensor(attentionMaskConclusion, maxConclusionLength, 0)
		attentionMaskIntroCentral = torch.concat((torch.ones(1), attention_mask[introFirstTokenIndex:conclusionFirstTokenIndex], torch.ones(1)), dim=0)
		attentionMaskIntroCentral = resizeSubtensor(attentionMaskIntroCentral, maxIntroCentralLength, 0)

		inputidsNew = torch.cat((tokenidsIntroCentral, tokenidsConclusion), dim=0)
		attentionMaskNew = torch.cat((attentionMaskIntroCentral, attentionMaskConclusion), dim=0)

		#print("inputidsNew.shape = ", inputidsNew.shape)
		#print("attentionMaskNew.shape = ", attentionMaskNew.shape)

		sample[lineIndex]["input_ids"] = inputidsNew
		sample[lineIndex]["attention_mask"] = attentionMaskNew

	batch = batchLists(sample)
	return batch
	
tokenized_datasets.set_format("torch")
if(centralSequencePrediction):
	train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=batchSize, shuffle=shuffleTrainDataset, collate_fn=centralSequencePredictionCollateFunction)
	eval_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=batchSize, collate_fn=centralSequencePredictionCollateFunction)
else:
	train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=batchSize, shuffle=shuffleTrainDataset)
	eval_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=batchSize)

weight_decay = 0.1


def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
	params_with_wd, params_without_wd = [], []
	for n, p in model.named_parameters():
		if any(nd in n for nd in no_decay):
			params_without_wd.append(p)
		else:
			params_with_wd.append(p)
	return [
		{"params": params_with_wd, "weight_decay": weight_decay},
		{"params": params_without_wd, "weight_decay": 0.0},
	]

model = GPT2LMHeadModel(config)

optimizer = AdamW(get_grouped_params(model), lr=5e-4)

#accelerator = Accelerator(fp16=True)	#TypeError: __init__() got an unexpected keyword argument 'fp16'
accelerator = Accelerator()

def evaluateAndSave(model, accelerator, output_dir, eval_dataloader):
	evaluate(model, eval_dataloader)
	model.train()
	saveModel(model, accelerator, output_dir, eval_dataloader)

def saveModel(model, accelerator, output_dir, eval_dataloader):
	accelerator.wait_for_everyone()
	unwrapped_model = accelerator.unwrap_model(model)
	unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
	if accelerator.is_main_process:
		tokenizer.save_pretrained(output_dir)
		
def evaluate(model, eval_dataloader):	
	model.eval()
	losses = []
	if(printAccuracy):
		accuracies = []
	for step, batch in enumerate(eval_dataloader):
		#print("step = ", step)
		with torch.no_grad():
			if(datasetName=='wikitext'):
				attention_mask=batch["attention_mask"]
			else:
				attention_mask = None
			outputs = model(batch["input_ids"], labels=batch["input_ids"], attention_mask=attention_mask)
		losses.append(accelerator.gather(outputs.loss.reshape(1)))		#OLD outputs.loss
		if(printAccuracy):
			inputs = batch["input_ids"]
			logits = outputs.logits
			accuracy = getAccuracy(inputs, logits, attention_mask)
			#print("accuracy = ", accuracy)
			if(not math.isnan(accuracy)):
				accuracies.append(accuracy.reshape(1))
	loss = torch.mean(torch.cat(losses))
	try:
		perplexity = torch.exp(loss)
	except OverflowError:
		perplexity = float("inf")
	eval_loss = loss.item()
	eval_perplexity = perplexity.item()
	if(printAccuracy):
		accuracy = torch.mean(torch.cat(accuracies))
		eval_accuracy = accuracy.item()
		print({"loss/eval": eval_loss, "perplexity": eval_perplexity, "accuracy/eval":eval_accuracy})
	else:
		print({"loss/eval": eval_loss, "perplexity": eval_perplexity})
	return eval_loss, eval_perplexity

if(stateTrainDataset):
	if(trainLoadFromPrevious):
		model = GPT2LMHeadModel.from_pretrained("./" + model_name)	#local_files_only=True
		model_size = sum(t.numel() for t in model.parameters())
		print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
		model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)
	else:
		model_size = sum(t.numel() for t in model.parameters())
		print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
		model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)

	num_update_steps_per_epoch = len(train_dataloader)
	num_training_steps = num_train_epochs * num_update_steps_per_epoch

	lr_scheduler = get_scheduler(
		name="linear",
		optimizer=optimizer,
		num_warmup_steps=1_000,
		num_training_steps=num_training_steps,
	)

	output_dir = model_name

	samples_per_step = accelerator.state.num_processes * batchSize
	gradient_accumulation_steps = 8
	eval_steps = 5_000	#5_000	#10

	saveModel(model, accelerator, output_dir, eval_dataloader)
	
	model.train()
	completed_steps = 0
	for epoch in range(num_train_epochs):
		for step, batch in tqdm(enumerate(train_dataloader, start=1), total=num_training_steps):
			if(datasetName=='wikitext'):
				attention_mask=batch["attention_mask"]
			else:
				attention_mask = None
			logits = model(batch["input_ids"], attention_mask=attention_mask).logits
			loss = keytoken_weighted_loss(batch["input_ids"], logits, keytoken_ids, attention_mask=attention_mask)
			if step % 100 == 0:
				accelerator.print(
					{
						"lr": lr_scheduler.get_lr(),
						"samples": step * samples_per_step,
						"steps": completed_steps,
						"loss/train": loss.item() * gradient_accumulation_steps,
					}
				)
			loss = loss / gradient_accumulation_steps
			accelerator.backward(loss)
			if step % gradient_accumulation_steps == 0:
				accelerator.clip_grad_norm_(model.parameters(), 1.0)
				optimizer.step()
				lr_scheduler.step()
				optimizer.zero_grad()
				completed_steps += 1
			if (step % (eval_steps * gradient_accumulation_steps)) == 0:
				evaluateAndSave(model, accelerator, output_dir, eval_dataloader)

	if(saveTrainedModel):
		#output_dir = model_name + "-final"	#temp: separate final stage save from intermittent save
		evaluateAndSave(model, accelerator, output_dir, eval_dataloader)

if(stateTestDataset):
	model = GPT2LMHeadModel.from_pretrained("./" + model_name)	#local_files_only=True
	model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)
	evaluate(model, eval_dataloader)
