"""GPT2pt.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)
	
based on; https://huggingface.co/learn/nlp-course/chapter7/6
https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter7/section6_pt.ipynb

# License:
MIT License

# Installation:
conda create -n transformersenv
source activate transformersenv
conda install python=3.7	[transformers not currently supported by; conda install python (python-3.10.6)]
pip install datasets
pip install transfomers==4.23.1
pip install torch
pip install evaluate
pip install accelerate

# Usage:
source activate transformersenv
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
batchSize = 16	#16	#default: 16	#orig: 32
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

'''
#recreate huggingface-course/codeparrot-ds-*:
# This cell will take a very long time to execute
print("start load dataset")
data = load_dataset(f"transformersbook/codeparrot-{split}", split=split, streaming=True)
filtered_data = filter_streaming_dataset(data, filters)
print("end load dataset")
'''

print("start load dataset")
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

context_length = 128
tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")


'''
filters = ["pandas", "sklearn", "matplotlib", "seaborn"]
example_1 = "import numpy as np"
example_2 = "import pandas as pd"
print(any_keyword_in_string(example_1, filters), any_keyword_in_string(example_2, filters))

#raw_datasets

for key in raw_datasets["train"][0]:
	print(f"{key.upper()}: {raw_datasets['train'][0][key][:200]}")

outputs = tokenizer(
	raw_datasets["train"][:2]["content"],
	truncation=True,
	max_length=context_length,
	return_overflowing_tokens=True,
	return_length=True,
)
print(f"Input IDs length: {len(outputs['input_ids'])}")
print(f"Input chunk lengths: {(outputs['length'])}")
print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")
'''


def tokenize(element):
	outputs = tokenizer(
		element["content"],
		truncation=True,
		max_length=context_length,
		return_overflowing_tokens=True,
		return_length=True,
	)
	input_batch = []
	for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
		#print("length = ", length)
		if length == context_length:	#only perform prediction for sufficient context_length (no padding)
			input_batch.append(input_ids)
	return {"input_ids": input_batch}


tokenized_datasets = raw_datasets.map(tokenize, batched=True, remove_columns=raw_datasets["train"].column_names)
#tokenized_datasets
	
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
		
	'''
	import torch
	from transformers import pipeline

	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	pipe = pipeline(
		"text-generation", model="huggingface-course/codeparrot-ds", device=device
	)

	txt = """\
	# create some data
	x = np.random.randn(100)
	y = np.random.randn(100)

	# create scatter plot with x, y
	"""
	print(pipe(txt, num_return_sequences=1)[0]["generated_text"])

	txt = """\
	# create some data
	x = np.random.randn(100)
	y = np.random.randn(100)

	# create dataframe from x and y
	"""
	print(pipe(txt, num_return_sequences=1)[0]["generated_text"])

	txt = """\
	# dataframe with profession, income and name
	df = pd.DataFrame({'profession': x, 'income':y, 'name': z})

	# calculate the mean income per profession
	"""
	print(pipe(txt, num_return_sequences=1)[0]["generated_text"])

	txt = """
	# import random forest regressor from scikit-learn
	from sklearn.ensemble import RandomForestRegressor

	# fit random forest model with 300 estimators on X, y:
	"""
	print(pipe(txt, num_return_sequences=1)[0]["generated_text"])
	'''

keytoken_ids = []
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

def keytoken_weighted_loss(inputs, logits, keytoken_ids, alpha=1.0):
	# Shift so that tokens < n predict n
	shift_labels = inputs[..., 1:].contiguous()
	shift_logits = logits[..., :-1, :].contiguous()
	# Calculate per-token loss
	loss_fct = CrossEntropyLoss(reduce=False)
	loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
	# Resize and average loss per sample
	loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
	# Calculate and scale weighting
	weights = torch.stack([(inputs == kt).float() for kt in keytoken_ids]).sum(axis=[0, 2])
	weights = alpha * (1.0 + weights)
	# Calculate weighted average
	weighted_loss = (loss_per_sample * weights).mean()
	return weighted_loss

def getAccuracy(inputs, logits):	
	#based on SBNLPpt_data:getAccuracy
	logits = logits.detach()
	# Shift so that tokens < n predict n
	shift_labels = inputs[..., 1:].contiguous()
	shift_logits = logits[..., :-1, :].contiguous()
	tokenLogitsTopIndex = torch.topk(shift_logits, accuracyTopN).indices	#get highest n scored entries from dictionary	#tokenLogitsTopIndex.shape = batchSize, sequenceMaxNumTokens, accuracyTopN
	if(accuracyTopN == 1):
		tokenLogitsTopIndex = torch.squeeze(tokenLogitsTopIndex)	#tokenLogitsTopIndex[:, :, 1] -> #tokenLogitsTopIndex[:, :] 	
		comparison = (tokenLogitsTopIndex == shift_labels).float()
		accuracy = torch.mean(comparison)
	else:
		labelsExpanded = torch.unsqueeze(shift_labels, dim=2)
		labelsExpanded = labelsExpanded.expand(-1, -1, tokenLogitsTopIndex.shape[2])	#labels broadcasted to [batchSize, sequenceMaxNumTokens, accuracyTopN]
		comparison = (tokenLogitsTopIndex == labelsExpanded).float()
		accuracy = torch.mean(comparison)
	#print("accuracy = ", accuracy)
	return accuracy
	
tokenized_datasets.set_format("torch")
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
			outputs = model(batch["input_ids"], labels=batch["input_ids"])
		losses.append(accelerator.gather(outputs.loss.reshape(1)))		#OLD outputs.loss
		if(printAccuracy):
			inputs = batch["input_ids"]
			logits = outputs.logits
			accuracy = getAccuracy(inputs, logits)
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
		model = GPT2LMHeadModel.from_pretrained("./codeparrot-ds-accelerate")	#local_files_only=True
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

	model_name = "codeparrot-ds-accelerate"
	output_dir = "codeparrot-ds-accelerate"

	samples_per_step = accelerator.state.num_processes * batchSize
	gradient_accumulation_steps = 8
	eval_steps = 5_000	#5_000	#10

	saveModel(model, accelerator, output_dir, eval_dataloader)
	
	model.train()
	completed_steps = 0
	for epoch in range(num_train_epochs):
		for step, batch in tqdm(enumerate(train_dataloader, start=1), total=num_training_steps):
			logits = model(batch["input_ids"]).logits
			loss = keytoken_weighted_loss(batch["input_ids"], logits, keytoken_ids)
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
		#output_dir = "codeparrot-ds-accelerate-final"	#temp: separate final stage save from intermittent save
		 evaluateAndSave(model, accelerator, output_dir, eval_dataloader)

if(stateTestDataset):
	model = GPT2LMHeadModel.from_pretrained("./codeparrot-ds-accelerate")	#local_files_only=True
	model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)
	evaluate(model, eval_dataloader)
