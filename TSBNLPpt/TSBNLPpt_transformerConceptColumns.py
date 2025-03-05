"""TSBNLPpt_transformerConceptColumns.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see TSBNLPpt_main.py

# Usage:
see TSBNLPpt_main.py

# Description:
TSBNLPpt transformer Concept Columns
	
"""

import torch as pt
from torch import nn
import nncustom

from transformers.activations import ACT2FN 

from TSBNLPpt_globalDefs import *

import spacy
nlp = spacy.load('en_core_web_md')	#spacy.load('en_core_web_lg')
import nltk
#nltk.download('wordnet')	#Make sure you have downloaded the WordNet corpus:
from nltk.corpus import wordnet as wn

def initialise_dictionary():
	global noun_dict
	noun_dict = build_noun_dictionary()
	if(debugDetectLocalConceptColumns):
		localConceptColumnExpertsTotal = 1000	#use small number of experts
	else:
		localConceptColumnExpertsTotal = len(noun_dict)
	print(f"Number of noun lemmas: {localConceptColumnExpertsTotal}")
	#sample_items = list(noun_dict.items())[:20]	# Inspect a few random entries:
	#print(sample_items)
	return localConceptColumnExpertsTotal

def build_noun_dictionary():
	"""
	Build a dictionary of (lowercased noun lemma) -> (unique integer ID)
	using WordNet from NLTK.
	"""
	noun_dict = {}
	next_id = 1

	# All synsets for nouns only (pos='n').
	for synset in wn.all_synsets(pos='n'):
		for lemma in synset.lemmas():
			lemma_name = lemma.name().lower()
			# If we've not seen this lemma before, assign a new ID.
			if lemma_name not in noun_dict:
				noun_dict[lemma_name] = next_id
				next_id += 1

	return noun_dict
	

# A helper to fetch the dictionary index for a noun
def get_concept_id(token):
	lemma_lower = token.lemma_.lower()
	if lemma_lower in noun_dict:
		if(debugDetectLocalConceptColumns):
			index = list(noun_dict.keys()).index(lemma_lower)
			if(index < debugDetectLocalConceptColumnsMaxExperts):
				concept_id = noun_dict[lemma_lower] + 1	#assumes localConceptColumnExpertsNoDictionaryNounID==0
			else:
				concept_id = localConceptColumnExpertsNoColumnID
		else:
			concept_id = noun_dict[lemma_lower] + 1	#assumes localConceptColumnExpertsNoDictionaryNounID==0
	else:
		concept_id = localConceptColumnExpertsNoDictionaryNounID
	return concept_id
	
def generateConceptColumnIndices(device, tokenizer, batch_input_ids, batch_offsets, identify_type="identify_both_columns"):
	"""
	Generates the following for each sample in the batch:
	1) conceptColumnStartIndices: A tensor indicating, for each token, the start index of its concept column.
	2) conceptColumnEndIndices: A tensor indicating, for each token, the end index of its concept column.
	3) conceptColumnIDs: A tensor of noun IDs (from a given common dictionary) for each token.
		- If a token is recognized as a noun, conceptColumnIDs[token_idx] = noun_dict[lowercased_token_text]+1.
		- If the noun isn't in the dictionary, or if it's not a noun, conceptColumnIDs[token_idx] = 0.
	"""
	
	noun_pos_tags = {'NOUN', 'PROPN'}
	non_noun_pos_tags = {'ADJ', 'ADV', 'VERB', 'ADP', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NUM', 'PART', 'PRON', 'SCONJ', 'SYM', 'X'}

	batch_concept_start_indices = []
	batch_concept_end_indices = []
	batch_concept_ids = []
	
	batch_size = len(batch_input_ids)
	for batch_index in range(batch_size):
		input_ids_sample = batch_input_ids[batch_index]
		offsets_sample = batch_offsets[batch_index]
		seq_length = input_ids_sample.shape[0]
		
		tokens = tokenizer.decode(input_ids_sample, skip_special_tokens=True)	
	
		  # Process the text with spaCy
		doc = nlp(tokens)

		# Create a mapping from character positions to spaCy tokens
		char_to_token = []
		for idx, (start_char, end_char) in enumerate(offsets_sample):
			if start_char == end_char:
				# Special tokens (e.g., <s>, </s>)
				char_to_token.append(None)
			else:
				# Find the corresponding spaCy token
				for token in doc:
					if token.idx <= start_char and token.idx + len(token) >= end_char:
						char_to_token.append(token)
						break
				else:
					char_to_token.append(None)

		# Initialize lists for concept columns
		if(identify_type=="identify_both_columns"):
			conceptColumnStartIndexList = [0]
		elif(identify_type=="identify_previous_column"):
			conceptColumnStartIndexList = [0]
		elif(identify_type=="identify_next_column"):
			conceptColumnStartIndexList = []	#a concept column will not be assigned for these first indices (before first noun)
		conceptColumnEndIndexList = []

		# Track conceptColumnIDs for each token
		concept_indices_list = []  # Indices of concept words in tokens
		token_concept_ids = pt.zeros(seq_length, dtype=pt.long)
		first_noun_idx = 0
		
		for idx in range(seq_length):
			token = char_to_token[idx]
			if token is not None:
				pos = token.pos_
				if pos in noun_pos_tags:
					first_noun_idx = idx
					if(identify_type=="identify_both_columns"):
						if len(conceptColumnStartIndexList) > 1:
							conceptColumnEndIndexList.append(idx - 1)
						conceptColumnStartIndexList.append(idx + 1)
					elif(identify_type=="identify_previous_column"):
						#identify previous column (preceeding noun);
						conceptColumnEndIndexList.append(idx)	#include current noun
						conceptColumnStartIndexList.append(idx + 1)
					elif(identify_type=="identify_next_column"):
						#identify next column (following noun);
						if len(conceptColumnStartIndexList) > 0:
							conceptColumnEndIndexList.append(idx - 1)
						conceptColumnStartIndexList.append(idx)	#include current noun
					# Set concept ID
					concept_indices_list.append(idx)
					token_concept_ids[idx] = get_concept_id(token)

		# get first_pad_index;
		pad_token_id = tokenizer.pad_token_id	#default=1 #https://huggingface.co/transformers/v2.11.0/model_doc/roberta.html 
		pad_indices = (input_ids_sample == pad_token_id).nonzero(as_tuple=True)[0]
		if len(pad_indices) > 0:
			first_pad_index = pad_indices[0].item()
		else:
			first_pad_index = (seq_length - 1)
		
		if(identify_type=="identify_previous_column"):
			# Remove the last start index as per the pseudocode
			if(len(conceptColumnStartIndexList) > 0):
				conceptColumnStartIndexList.pop()
			#a concept column will not be assigned for these last indices (after last noun)
		elif(identify_type=="identify_next_column"):
			if(len(conceptColumnStartIndexList) > 0):
				conceptColumnEndIndexList.append(first_pad_index)
		elif(identify_type=="identify_both_columns"):
			conceptColumnEndIndexList.append(first_pad_index)
			# Remove the last start index as per the pseudocode
			if(len(conceptColumnStartIndexList) > 1):
				conceptColumnStartIndexList.pop()
		
		'''
		print("identify_type = ", identify_type)
		print("conceptColumnStartIndexList = ", conceptColumnStartIndexList)
		print("conceptColumnEndIndexList = ", conceptColumnEndIndexList)
		print("len conceptColumnStartIndexList = ", len(conceptColumnStartIndexList))
		print("len conceptColumnEndIndexList = ", len(conceptColumnEndIndexList))
		'''
		assert len(conceptColumnStartIndexList) == len(conceptColumnEndIndexList)
		
		# For each token, assign its concept column start and end indices
		token_concept_start_indices = pt.zeros(seq_length, dtype=pt.long)
		token_concept_end_indices = pt.zeros(seq_length, dtype=pt.long)

		# Assign concept columns to tokens
		current_concept_idx = 0
		for idx in range(seq_length):
			if current_concept_idx < len(conceptColumnStartIndexList):
				start_idx = conceptColumnStartIndexList[current_concept_idx]
				end_idx = conceptColumnEndIndexList[current_concept_idx]
				
				validIndex = True
				if(identify_type=="identify_next_column"):
					if(idx < first_noun_idx):
						validIndex = False
				if(validIndex):
					token_concept_start_indices[idx] = start_idx
					token_concept_end_indices[idx] = end_idx
				else:
					token_concept_start_indices[idx] = idx 	#attend to self only
					token_concept_end_indices[idx] = idx 	#attend to self only
					
				if idx == end_idx:
					current_concept_idx += 1
			else:
				# For tokens after the last concept column
				token_concept_start_indices[idx] = idx	#attend to self only	#orig: conceptColumnStartIndexList[-1]
				token_concept_end_indices[idx] = idx	#attend to self only	#orig: conceptColumnEndIndexList[-1]

		# Fill in concept IDs for non-noun tokens using the noun in their column.
		for idx in range(seq_length):
			# If it's still 0, we need to propagate from the column's noun
			if token_concept_ids[idx] == 0:
				s = token_concept_start_indices[idx].item()
				e = token_concept_end_indices[idx].item()
				# Within s..e, there should be exactly one noun token (or none).
				# We'll take the first noun we find there:
				column_id = localConceptColumnExpertsNoColumnID
				for col_idx in range(s, e + 1):
					if token_concept_ids[col_idx] != 0:
						column_id = token_concept_ids[col_idx].item()
						break
				token_concept_ids[idx] = column_id
				
		batch_concept_start_indices.append(token_concept_start_indices)
		batch_concept_end_indices.append(token_concept_end_indices)
		batch_concept_ids.append(token_concept_ids)

	# Stack tensors to create batch tensors
	conceptColumnStartIndices = pt.stack(batch_concept_start_indices)  # Shape: [batch_size, seq_length]
	conceptColumnEndIndices = pt.stack(batch_concept_end_indices)	  # Shape: [batch_size, seq_length]
	conceptColumnIDs = pt.stack(batch_concept_ids)	  # Shape: [batch_size, seq_length]

	conceptColumnStartIndices = conceptColumnStartIndices.to(device)
	conceptColumnEndIndices = conceptColumnEndIndices.to(device)
	conceptColumnIDs = conceptColumnIDs.to(device)

	'''
	if(debugDetectLocalConceptColumns):
		print("batch_input_ids = ", batch_input_ids)
		print("conceptColumnStartIndices = ", conceptColumnStartIndices)
		print("conceptColumnEndIndices = ", conceptColumnEndIndices)
		print("conceptColumnIDs = ", conceptColumnIDs)
	'''
	
	return conceptColumnStartIndices, conceptColumnEndIndices, conceptColumnIDs

if(localConceptColumnAttention):

	def applyLocalConceptColumnAttention(self, hidden_states, attention_mask, head_mask, query_layer, key_layer, value_layer, conceptColumnStartIndices, conceptColumnEndIndices, conceptColumnIDsPrev, conceptColumnIDsNext):
		# Compute attention scores
		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

		# Create local concept column mask
		batch_size, seq_length = hidden_states.size(0), hidden_states.size(1)
		device = hidden_states.device
		positions = torch.arange(seq_length, device=device).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, seq_length]

		# Expand start and end indices to match attention_scores dimensions
		s = conceptColumnStartIndices.unsqueeze(1).unsqueeze(-1)  # Shape: [batch_size, 1, seq_length, 1]
		e = conceptColumnEndIndices.unsqueeze(1).unsqueeze(-1)	# Shape: [batch_size, 1, seq_length, 1]

		# Create mask
		local_mask = (positions >= s) & (positions <= e)  # Shape: [batch_size, 1, seq_length, seq_length]

		# Expand mask for attention heads
		local_mask = local_mask.expand(batch_size, self.num_attention_heads, seq_length, seq_length)

		# Apply mask to attention scores
		attention_scores = attention_scores.masked_fill(~local_mask, -float('inf'))

		# Scale attention scores
		attention_scores = attention_scores / math.sqrt(self.attention_head_size)

		# Apply attention mask (e.g., padding mask)
		if attention_mask is not None:
			attention_scores = attention_scores + attention_mask

		# Normalize the attention scores to probabilities
		attention_probs = nn.functional.softmax(attention_scores, dim=-1)
		attention_probs = self.dropout(attention_probs)

		# Apply head mask if provided
		if head_mask is not None:
			attention_probs = attention_probs * head_mask

		# Compute context layer
		context_layer = torch.matmul(attention_probs, value_layer)

		return attention_probs, context_layer
		
if(localConceptColumnExperts):

	class ExpertIntermediate(nn.Module):
		"""
		Similar to RobertaIntermediate, except it uses a small intermediate size.
		"""
		def __init__(self, config):
			super().__init__()
			# Use a small intermediate dimension, e.g. 10
			# (Assumes config.expert_intermediate_size is set to 10 or similar)
			self.dense = nncustom.Linear(config.hidden_size, config.expert_intermediate_size)
			if isinstance(config.hidden_act, str):
				self.intermediate_act_fn = ACT2FN[config.hidden_act]
			else:
				self.intermediate_act_fn = config.hidden_act

		def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
			hidden_states = self.dense(hidden_states)
			hidden_states = self.intermediate_act_fn(hidden_states)
			return hidden_states


	class ExpertOutput(nn.Module):
		"""
		Similar to RobertaOutput, but from expert_intermediate_size back to hidden_size.
		"""
		def __init__(self, config):
			super().__init__()
			self.dense = nncustom.Linear(config.expert_intermediate_size, config.hidden_size)
			self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
			self.dropout = nn.Dropout(config.hidden_dropout_prob)

		def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
			hidden_states = self.dense(hidden_states)
			hidden_states = self.dropout(hidden_states)
			hidden_states = self.LayerNorm(hidden_states + input_tensor)
			return hidden_states

	def feed_forward_experts(self, hidden_states: torch.Tensor, conceptColumnIDs: torch.LongTensor) -> torch.Tensor:
		"""
		hidden_states: (batch_size, seq_len, hidden_dim)
		conceptColumnIDs: (batch_size, seq_len) with values in [0 .. localConceptColumnExpertsTotal-1]
		"""
		bsz, seq_len, hidden_dim = hidden_states.size()

		# Flatten over batch and sequence so we have shape (B*S, hidden_dim)
		flat_hidden_states = hidden_states.view(-1, hidden_dim)
		flat_concepts = conceptColumnIDs.view(-1)  # (B*S,)

		# Prepare an output buffer of the same shape
		output_buffer = torch.empty_like(flat_hidden_states)

		# 1) Bypass tokens that have conceptColumnIDs == localConceptColumnExpertsNoColumnID
		skip_mask = (flat_concepts == localConceptColumnExpertsNoColumnID)
		# Those tokens will simply copy input -> output
		output_buffer[skip_mask] = flat_hidden_states[skip_mask]

		# 2) Route the remaining tokens to the appropriate expert
		for expert_id in range(self.localConceptColumnExpertsTotal):
			expert_mask = (flat_concepts == expert_id)
			if not expert_mask.any():
				continue  # Skip if no tokens go to this expert

			# Indices of tokens that go to this expert
			expert_indices = expert_mask.nonzero(as_tuple=True)[0]

			# Extract those token embeddings
			expert_input = flat_hidden_states[expert_indices]

			# Forward through expert MLP
			# ExpertIntermediate -> ExpertOutput (with residual in the output)
			intermediate_out = self.expertIntermediates[expert_id](expert_input)
			expert_out = self.expertOutputs[expert_id](intermediate_out, expert_input)

			# Place the expert output back into the correct positions
			output_buffer[expert_indices] = expert_out

		# Reshape back to (batch_size, seq_len, hidden_dim)
		layer_output = output_buffer.view(bsz, seq_len, hidden_dim)
		return layer_output
