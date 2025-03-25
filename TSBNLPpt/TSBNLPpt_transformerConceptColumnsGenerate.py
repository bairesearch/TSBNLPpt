"""TSBNLPpt_transformerConceptColumnsGenerate.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see TSBNLPpt_main.py

# Usage:
see TSBNLPpt_main.py

# Description:
TSBNLPpt transformer Concept Columns Generate
	
"""

import torch as pt

from TSBNLPpt_globalDefs import *

import spacy
from spacy.attrs import POS, ORTH, IDX
nlp = spacy.load('en_core_web_md')	#spacy.load('en_core_web_lg')
import nltk
#nltk.download('wordnet')	#Make sure you have downloaded the WordNet corpus:
from nltk.corpus import wordnet as wn


def initialise_dictionary():
	global noun_dict
	noun_dict = build_noun_dictionary()
	if(localConceptColumnExpertsStoreRAM):
		localConceptColumnExpertsTotal = localConceptColumnExpertsNumber
		print(f"Number of noun lemmas: {len(noun_dict)}")
	else:
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

	if(localConceptColumnExpertsStoreRAM):
		nounFrequencies = {}
		
		for syn in wn.all_synsets('n'):  # 'n' for nouns
			for lemma in syn.lemmas():
				nounFrequencies[lemma.name()] = nounFrequencies.get(lemma.name(), 0) + lemma.count()

		# Sort dictionary by frequency count in descending order
		orderedNounFrequencies = dict(sorted(nounFrequencies.items(), key=lambda item: item[1], reverse=True))
		
		mostCommonLemmasLen = localConceptColumnExpertsNumber-1
		mostCommonLemmas = list(orderedNounFrequencies.keys())[:mostCommonLemmasLen]
		
		noun_dict = {lemma: i for i, lemma in enumerate(mostCommonLemmas)}
		
		print("len(mostCommonLemmas) = ", len(mostCommonLemmas))
		#print(list(orderedNounFrequencies.items())[:20])
	else:
		# All synsets for nouns only (pos='n').
		for synset in wn.all_synsets(pos='n'):
			for lemma in synset.lemmas():
				lemma_name = lemma.name().lower()
				# If we've not seen this lemma before, assign a new ID.
				if lemma_name not in noun_dict:
					noun_dict[lemma_name] = next_id
					next_id += 1

	return noun_dict
	
	
def generateConceptColumnIndicesParallel(device, tokenizer, batch_input_ids, batch_offsets, identify_type="identify_both_columns"):
	"""
	Vectorized version that avoids explicit for-loops over tokens.
	We still do a for-loop over the samples in the batch, because
	typically we can't run spaCy across an entire batch simultaneously
	in a purely tensorized way.

	Each step (#1 to #6) is replaced by the above _Optimized function.
	We then aggregate results in lists, stack them, etc.
	"""
	
	batch_concept_start_indices = []
	batch_concept_end_indices   = []
	batch_concept_ids		   = []

	batch_size = len(batch_input_ids)
	for sample_index in range(batch_size):
		input_ids_sample = batch_input_ids[sample_index]
		offsets_sample   = batch_offsets[sample_index]
		seq_length	   = input_ids_sample.shape[0]

		# Step 1
		tokens = getTokens_Optimized(tokenizer, input_ids_sample)
		
		# Step 2
		doc = processTextWithSpacy_Optimized(nlp, tokens)
		
		# Step 3
		char_to_token, pos_t_noun = createMappingFromCharacterPositionsToSpacyTokens_Optimized(doc, offsets_sample)

		# Step 4a
		conceptColumnStartIndexTensor, conceptColumnEndIndexTensor = populateConceptColumnStartAndEndIndexTensor_Optimized(tokenizer, pos_t_noun, seq_length, identify_type, input_ids_sample)

		# Step 4b
		token_concept_ids, first_noun_idx = populateTokenConceptIdsAndFirstNounIdx_Optimized(noun_dict, char_to_token, pos_t_noun, seq_length, noun_pos_tags)

		# Step 5
		token_concept_start_indices, token_concept_end_indices = populateTokenConceptStartAndEndIndices_Optimized(conceptColumnStartIndexTensor, conceptColumnEndIndexTensor, seq_length, first_noun_idx, identify_type)

		# Step 6
		final_concept_ids = updateTokenConceptIds_Optimized(token_concept_ids, token_concept_start_indices, token_concept_end_indices)

		if(debugDetectLocalConceptColumns):
			exit()

		# Gather
		batch_concept_start_indices.append(token_concept_start_indices)
		batch_concept_end_indices.append(token_concept_end_indices)
		batch_concept_ids.append(final_concept_ids)

	# Stack
	conceptColumnStartIndices = pt.stack(batch_concept_start_indices)  # [batch_size, seq_length]
	conceptColumnEndIndices   = pt.stack(batch_concept_end_indices)	# [batch_size, seq_length]
	conceptColumnIDs		  = pt.stack(batch_concept_ids)			# [batch_size, seq_length]

	conceptColumnStartIndices = conceptColumnStartIndices.to(device)
	conceptColumnEndIndices   = conceptColumnEndIndices.to(device)
	conceptColumnIDs		  = conceptColumnIDs.to(device)

	return conceptColumnStartIndices, conceptColumnEndIndices, conceptColumnIDs



# We need a helper that fetches concept IDs given a batch of lemmas
# We'll vectorize as best we can by looking up many lemmas at once.
# If the lemma is not in noun_dict, we map it to localConceptColumnExpertsNoDictionaryNounID.
def vectorized_get_concept_id(noun_dict, lemmas: list):
	"""
	lemmas: list of lowercased lemmas (strings).
	Returns a LongTensor of shape [len(lemmas)] with the concept IDs.
	"""
	# We'll build the result as a list of ints. 
	# Then convert to a Tensor at the end. 
	result = []
	for lm in lemmas:
		if lm in noun_dict:
			result.append(noun_dict[lm] + 1)
		else:
			result.append(localConceptColumnExpertsNoDictionaryNounID)
	return pt.tensor(result, dtype=pt.long)

########################################
# Step #1: getTokens
########################################

def getTokens_Optimized(tokenizer, input_ids_sample):
	"""
	Original Step 1: get tokens (as a single string) via decode.
	There is no explicit Python for-loop here that iterates 
	over tokens, so we can leave it as is.
	
	The original function just returns a single string of tokens.
	We'll do the same and print the tokens.
	"""
	tokens = tokenizer.decode(input_ids_sample, skip_special_tokens=True)
	
	if(debugDetectLocalConceptColumns):
		print("tokens =", tokens)
		
	return tokens

########################################
# Step #2: processTextWithSpacy
########################################

def processTextWithSpacy_Optimized(nlp, tokens):
	"""
	Original Step 2: spaCy doc creation.
	There's no real way to 'vectorize' spaCy's doc pipeline 
	without losing spaCy objects. We'll keep it as is.
	"""
	doc = nlp(tokens)
	# We'll just print doc.text to confirm
	
	if(debugDetectLocalConceptColumns):
		print("doc.text =", doc.text)
		
	return doc

########################################
# Step #3: createMappingFromCharacterPositionsToSpacyTokens
#		  (Vectorized approach)
########################################

def createMappingFromCharacterPositionsToSpacyTokens_Optimized(doc, offsets_sample):
	"""
	Original Step 3 is a nested Python loop:
		for idx, (start_char, end_char) in enumerate(offsets_sample):
			for token in doc:
				if token.idx <= start_char and ...
					char_to_token.append(token or None)
	
	We can vectorize the 'search for matching token' part:
	  - Convert spaCy tokens to two arrays: doc_starts, doc_ends
	  - Convert offsets_sample to start_chars, end_chars
	  - We'll do a broadcast compare to find which doc tokens could match
	  - We'll pick the *first* doc token that satisfies the condition
	  - If none satisfy, that offset maps to None
	
	We'll end by printing the final Python list of [token or None].
	"""
	# Build arrays for doc tokens: [num_doc_tokens]
	doc_starts = []
	doc_ends = []
	for t in doc:
		doc_starts.append(t.idx)
		doc_ends.append(t.idx + len(t))
	doc_starts_t = pt.tensor(doc_starts, dtype=pt.long)  # shape [num_doc_tokens]
	doc_ends_t   = pt.tensor(doc_ends,   dtype=pt.long)  # shape [num_doc_tokens]

	# Convert offsets_sample to Tensors, if they aren't already
	offsets_t = offsets_sample  # shape [seq_length, 2]
	start_chars = offsets_t[:, 0]  # shape [seq_length]
	end_chars   = offsets_t[:, 1]  # shape [seq_length]

	# Identify which doc token covers each offset. Condition:
	#   doc_starts <= start_char AND doc_ends >= end_char
	# We'll broadcast:
	#   doc_starts_t  shape [num_doc_tokens] -> [1, num_doc_tokens]
	#   start_chars   shape [seq_length]	 -> [seq_length, 1]
	# Similarly for doc_ends_t and end_chars
	cond_starts = doc_starts_t.unsqueeze(0) <= start_chars.unsqueeze(1)  # [seq_length, num_doc_tokens]
	cond_ends   = doc_ends_t.unsqueeze(0)   >= end_chars.unsqueeze(1)	# [seq_length, num_doc_tokens]
	combined	= cond_starts & cond_ends								 # [seq_length, num_doc_tokens]

	# For each row i in [seq_length], we want the first column j for which combined[i,j] == True.
	# We'll do argmax on the boolean, but we must also handle the case "no True found".
	#   - If there's no True in row i, argmax returns 0. We must detect that with an 'any' check.
	any_match = combined.any(dim=1)				  # [seq_length], bool
	first_true_idx = pt.argmax(combined.long(), dim=1)	  # [seq_length], int in [0..num_doc_tokens-1]
	
	# If any_match[i] == False, that means no token covers that offset -> map to -1
	# We can fix that with a simple where:
	first_true_idx = pt.where(any_match, first_true_idx, pt.tensor(-1))

	# Now we store these integer indices, and after that we create the final python list:
	#   [doc[j] if j != -1 else None for j in first_true_idx]
	# That is a final pass in Python to keep the original output structure.
	char_to_token_indices = first_true_idx

	# Special tokens (e.g., <s>, </s>); char_to_token.append(None)
	specialTokens = start_chars==end_chars  
	char_to_token_indices = (char_to_token_indices * (~specialTokens).int()) + (specialTokens.int() * -1)
	
	# Build final char_to_token as a python list
	char_to_token = []
	idx_list = char_to_token_indices.tolist()
	for j in idx_list:
		if j == -1:
			char_to_token.append(None)
		else:
			char_to_token.append(doc[j])  # spaCy Token

	if(debugDetectLocalConceptColumns):
		print("char_to_token = ", char_to_token)

	# 1) Build a pos_mask (1 if noun, 0 if not).
	#	Since char_to_token[i] is a spaCy token or None, we do a comprehension in Python 
	#	to store pos. We'll keep that minimal.
	#	We'll *then* do the logic for concept columns with a mostly vector/tensor approach.
	pos_list = []
	for tok in char_to_token:
		if tok is not None and tok.pos_ in noun_pos_tags:
			pos_list.append(1)
		else:
			pos_list.append(0)
	pos_t_noun = pt.tensor(pos_list, dtype=pt.long)  # shape [seq_length]
			
	return char_to_token, pos_t_noun


########################################
# Step #4a: populateConceptColumnStartAndEndIndexTensor
#		   (Vectorized approach)
########################################

import torch

def _build_both_columns_starts_ends(noun_indices: torch.Tensor, first_pad_index: int):
	"""
	Equivalent to:
	  conceptColumnStartIndexList = [0]
	  for each noun n:
		if len(...) > 1: conceptColumnEndIndexList.append(n-1)
		conceptColumnStartIndexList.append(n+1)
	  conceptColumnEndIndexList.append(first_pad_index)
	  if len(conceptColumnStartIndexList) > 1:
		conceptColumnStartIndexList.pop()
	"""
	N = noun_indices.shape[0]
	
	if N == 0:
		# No nouns at all \u21d2 one big column from 0..pad
		starts = torch.tensor([0], dtype=torch.long)
		ends   = torch.tensor([first_pad_index], dtype=torch.long)
		return starts, ends
	
	# Build the 'starts'
	# The final logic after popping effectively yields:
	#   - If N=1,  -> starts = [0]
	#   - If N=2,  -> starts = [0, noun_indices[0]+1]
	#   - If N=3,  -> starts = [0, n0+1, n1+1], etc.
	# Implementation trick: always cat [0, noun_indices[:N-1]+1]
	#   which yields length=N if N>0.
	if N > 1:
		starts_middle = noun_indices[:N-1] + 1   # shape: (N-1,)
	else:
		starts_middle = torch.empty((0,), dtype=torch.long)
	starts = torch.cat([torch.tensor([0], dtype=torch.long), starts_middle], dim=0)  # shape: N
	
	# Build the 'ends'
	#   ends has length=N as well:
	#   The first N-1 end\u2010points = (n1 - 1, n2 - 1, ..., n_{N-1} - 1)
	#   and we append the first_pad_index as the final end.
	if N > 1:
		ends_middle = noun_indices[1:] - 1	   # shape: (N-1,)
	else:
		ends_middle = torch.empty((0,), dtype=torch.long)
	ends = torch.cat([
		ends_middle,
		torch.tensor([first_pad_index], dtype=torch.long)
	], dim=0)  # shape: N
	
	return starts, ends


def populateConceptColumnStartAndEndIndexTensor_Optimized(
	tokenizer,
	pos_t_noun: torch.Tensor, 
	seq_length: int, 
	identify_type: str, 
	input_ids_sample: torch.Tensor,
):
	"""
	Vectorized approach to build start/end column indices for noun-based slicing,
	eliminating Python loops by using advanced indexing.
	"""
	
	# 1) Identify where the PAD tokens start
	pad_token_id = tokenizer.pad_token_id
	pad_positions = (input_ids_sample == pad_token_id).nonzero(as_tuple=True)[0]
	if len(pad_positions) > 0:
		first_pad_index = pad_positions[0].item()
	else:
		first_pad_index = seq_length - 1
	
	# 2) Collect indices where token is a noun
	noun_indices = pos_t_noun.nonzero(as_tuple=True)[0]  # shape: (num_nouns,)
	
	# 3) Dispatch to the appropriate vector-building logic
	if identify_type == "identify_both_columns":
		starts, ends = _build_both_columns_starts_ends(noun_indices, first_pad_index)
	elif identify_type == "identify_previous_column":
		starts, ends = _build_previous_columns_starts_ends(noun_indices)
	elif identify_type == "identify_next_column":
		starts, ends = _build_next_column_starts_ends(noun_indices, first_pad_index)
	else:
		raise ValueError(f"Unrecognized identify_type: {identify_type}")
	
	# 4) Debug printing
	if debugDetectLocalConceptColumns:
		print("pos_t_noun (1=noun, 0=other) =", pos_t_noun)
		print("conceptColumnStartIndexList =", starts.tolist())
		print("conceptColumnEndIndexList   =", ends.tolist())
	
	# 5) Return as Tensors, ensuring they match in length
	assert starts.shape[0] == ends.shape[0], (
		f"Mismatch: len(starts)={starts.shape[0]}, len(ends)={ends.shape[0]}.\n"
		f"starts={starts}, ends={ends}"
	)
	
	return starts, ends

def _build_previous_columns_starts_ends(noun_indices: torch.Tensor):
	"""
	Equivalent to:
	  conceptColumnStartIndexList = [0]
	  for idx in seq_length:
		  if token is noun -> conceptColumnEndIndexList.append(idx)
							  conceptColumnStartIndexList.append(idx+1)
	  conceptColumnStartIndexList.pop()  # pop last
	"""
	N = noun_indices.shape[0]
	
	if N == 0:
		# No nouns \u21d2 no columns
		return torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long)
	
	# We end up with:
	#   starts candidate = [0, n0+1, n1+1, ..., n_{N-1}+1]  (length N+1)
	#   ends			 = [n0,	   n1,	 ..., n_{N-1}] (length N)
	# Then pop the last start \u21d2 final starts is length=N
	candidate_starts = torch.cat([
		torch.tensor([0], dtype=torch.long),
		noun_indices + 1
	])  # shape: (N+1,)
	
	ends   = noun_indices		  # shape: (N,)
	starts = candidate_starts[:-1] # shape: (N,)
	return starts, ends


def _build_next_column_starts_ends(noun_indices: torch.Tensor, first_pad_index: int):
	"""
	Equivalent to:
	  conceptColumnStartIndexList = []
	  for idx in seq_length:
		  if token is noun:
			  if len(starts) > 0: end.append(idx-1)
			  starts.append(idx)
	  end.append(first_pad_index)
	  # Must have same length as starts, so #columns = len(noun_indices)
	"""
	N = noun_indices.shape[0]
	
	if N == 0:
		# Original code would produce mismatch (ends=[pad], starts=[]).
		# If you truly want to replicate that (which triggers an assertion),
		# just do that. Otherwise, define "no columns" if no nouns:
		return torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long)
	
	# Final shape must be N columns.
	# starts = [n0,	 n1,	 n2, ... n_{N-1}]   (length N)
	# ends   = [n1 - 1, n2 - 1, ... n_{N-1}-1, pad] (length N)
	starts = noun_indices
	if N == 1:
		ends = torch.tensor([first_pad_index], dtype=torch.long)
	else:
		middle_ends = noun_indices[1:] - 1		   # shape: (N-1,)
		ends = torch.cat([
			middle_ends,
			torch.tensor([first_pad_index], dtype=torch.long)
		], dim=0)									# shape: (N,)
	
	return starts, ends




########################################
# Step #4b: populateTokenConceptIdsAndFirstNounIdx
#		   (Vectorized approach)
########################################

def populateTokenConceptIdsAndFirstNounIdx_Optimized(noun_dict, char_to_token, pos_t_noun, seq_length, noun_pos_tags):
	"""
	Original Step 4b:
	  - loop from idx=0..seq_length-1
	  - if token is noun -> store concept ID, track first_noun_idx, etc.
	
	Vectorized approach:
	  1) build a list/tensor of pos tags 
	  2) build a list/tensor of lemma_ for each token
	  3) gather concept ID
	  4) find first noun index if any
	"""
	# 1) Build pos mask and lemma array
	lemma_list = []
	for tok in char_to_token:
		if tok is not None and tok.pos_ in noun_pos_tags:
			lemma_list.append(tok.lemma_.lower())
		else:
			lemma_list.append(None)

	# 2) Gather concept IDs only for the noun positions, else 0
	#	We'll do a two-step approach:
	#	  (a) create an array of concept IDs for each position (0 if not noun)
	#	  (b) for noun positions, fetch concept ID using vectorized_get_concept_id
	#	We'll gather the lemmas for the noun positions
	noun_positions = pos_t_noun.nonzero(as_tuple=True)[0]
	noun_lemmas = []
	for i in noun_positions.tolist():
		# i-th token is a noun
		noun_lemmas.append(lemma_list[i] if lemma_list[i] is not None else "")

	# vectorized get_concept_id for the noun positions
	noun_concept_ids = vectorized_get_concept_id(noun_dict, noun_lemmas)  # shape [num_nouns]

	# Now we scatter these IDs back into a full-length seq tensor
	token_concept_ids = pt.ones(seq_length, dtype=pt.long) * localConceptColumnExpertsNoColumnID
	if noun_positions.shape[0] > 0:
		token_concept_ids = token_concept_ids.scatter(0, noun_positions, noun_concept_ids)

	# 3) find first_noun_idx
	first_noun_idx = 0
	if noun_positions.numel() > 0:
		first_noun_idx = noun_positions[0].item()  # the first noun
	else:
		# remains 0 if no noun was found
		pass

	if(debugDetectLocalConceptColumns):
		print("token_concept_ids = ", token_concept_ids)
		print("first_noun_idx = ", first_noun_idx)
	
	return token_concept_ids, first_noun_idx

########################################
# Step #5: populateTokenConceptStartAndEndIndices
#		  (Vectorized approach)
########################################

def populateTokenConceptStartAndEndIndices_Optimized(
	conceptColumnStartIndexTensor, 
	conceptColumnEndIndexTensor, 
	seq_length, 
	first_noun_idx,
	identify_type
):
	"""
	Original Step 5 logic:
	  - We have conceptColumnStartIndexList, conceptColumnEndIndexList of equal length
	  - We iterate over each token to figure out which column it belongs to
	  - If we've used up all columns, that token attends to itself only, etc.

	Vectorized approach:
	  We'll store for each token an integer 'current_col' that identifies 
	  which concept column they belong to. Then we gather the corresponding 
	  start/end. If no column remains, we default to (idx, idx).

	  There's no "nice" direct vector op for "walk through tokens, 
	  move to next column when we pass its end_idx" but we can do 
	  a search-based approach:

	  Strategy:
		1) We'll treat columns as intervals [startCol[i], endCol[i]].
		2) We'll build a large 1D arange [0..seq_length-1] for token indices.
		3) We'll compare each token index to each column interval. 
		   We'll pick the earliest column i such that index <= endCol[i].
		   (Of course we must also have used up the "previous" columns if index > that end.)
		4) If identify_type == "identify_next_column", then for token i < first_noun_idx, we default to (i,i).

	  For clarity and to keep code closer to the original, we might do a partial Python pass 
	  but *without* a for-loop over each token. We'll do a search approach in PyTorch.

	  We'll return token_concept_start_indices, token_concept_end_indices as Tensors. 
	"""

	num_cols = conceptColumnStartIndexTensor.shape[0]
	# If no columns exist, every token attends to self
	if num_cols == 0:
		token_concept_start_indices = pt.arange(seq_length, dtype=pt.long)
		token_concept_end_indices   = pt.arange(seq_length, dtype=pt.long)
		return token_concept_start_indices, token_concept_end_indices

	start_col_t = conceptColumnStartIndexTensor # shape [num_cols]
	end_col_t   = conceptColumnEndIndexTensor 	# shape [num_cols]

	# We'll build an array [0..seq_length-1]
	token_positions = pt.arange(seq_length, dtype=pt.long)

	# We'll figure out which column index each token belongs to, from left to right.
	# For each token i, find the earliest col j such that i <= end_col_t[j].
	# Once we've found that j, that means i belongs to column j (provided i >= start_col_t[j], 
	# but the code in the original picks the column in a cumulative sense).
	# We'll do a broadcast: compare token_positions.unsqueeze(1) to end_col_t ( shape [num_cols] ).
	# Then we pick the first j that satisfies i <= end_col_t[j].
	# We'll also handle the "next_column" case for i < first_noun_idx.

	# expanded = token_positions.unsqueeze(1).repeat(1, num_cols) <= end_col_t.unsqueeze(0)
	# shape [seq_length, num_cols], True if token i <= end_col_t[j]
	# We'll do argmax over dim=1 to get the first True along j.

	less_equal_end = token_positions.unsqueeze(1) <= end_col_t.unsqueeze(0)  # shape [seq_length, num_cols]
	any_true = less_equal_end.any(dim=1)			 # shape [seq_length]
	first_true_idx = pt.argmax(less_equal_end.long(), dim=1)  # shape [seq_length]; 0 if no True or if first True is col0
	# If no True, we have used up all columns -> attend to self
	first_true_idx = pt.where(any_true, first_true_idx, pt.tensor(num_cols))  
	# so if there's no column that ends after i, we store 'num_cols' as a sentinel 
	# meaning "no column left -> attend to self"

	# Now we must also ensure that i >= the start of the chosen column. 
	# If i < start_col_t[j], that's not valid. So let's refine:
	# We do a second check i >= start_col_t[j]. We'll do the standard approach:
	#   valid_col = (i >= start_col_t) & (i <= end_col_t). 
	#   Then we pick the first j with True. 
	#   If none is True, we do the self case.

	valid_start = token_positions.unsqueeze(1) >= start_col_t.unsqueeze(0)  # shape [seq_length, num_cols]
	valid_end   = token_positions.unsqueeze(1) <= end_col_t.unsqueeze(0)
	valid_range = valid_start & valid_end  # shape [seq_length, num_cols]

	any_match = valid_range.any(dim=1)	 # shape [seq_length]
	chosen_col_idx = pt.argmax(valid_range.long(), dim=1)  # first True along each row
	chosen_col_idx = pt.where(any_match, chosen_col_idx, pt.tensor(num_cols))

	# Now we handle the identify_type="identify_next_column" special rule 
	# (tokens i < first_noun_idx => attend to self).
	if identify_type == "identify_next_column":
		# For i < first_noun_idx, chosen_col_idx = num_cols
		mask_before_first_noun = token_positions < first_noun_idx
		chosen_col_idx = pt.where(mask_before_first_noun, pt.tensor(num_cols), chosen_col_idx)

	# Now gather the actual (start_idx, end_idx) from chosen_col_idx
	# We'll make a new array that is shape [num_cols+1], where the last entry 
	# is an invalid sentinel we interpret as self. 
	# We'll store start_col_t, end_col_t for columns 0..num_cols-1,
	# plus we store an identity index for the sentinel. 
	start_extended = pt.cat([start_col_t, pt.arange(1)])  # shape [num_cols+1]
	end_extended   = pt.cat([end_col_t,   pt.arange(1)])

	# We want the sentinel to be "token index" => so let's set the last entry to -1 for now
	# and we will do a final fix if the chosen column is the sentinel
	start_extended[-1] = -1
	end_extended[-1]   = -1

	token_concept_start_indices = start_extended.gather(0, chosen_col_idx)
	token_concept_end_indices   = end_extended.gather(0, chosen_col_idx)

	# If the chosen_col_idx is the sentinel (== num_cols), that means no column -> attend to self
	sentinel_mask = chosen_col_idx == num_cols
	token_concept_start_indices = pt.where(
		sentinel_mask, token_positions, token_concept_start_indices
	)
	token_concept_end_indices = pt.where(
		sentinel_mask, token_positions, token_concept_end_indices
	)

	if(debugDetectLocalConceptColumns):
		print("token_concept_start_indices = ", token_concept_start_indices)
		print("token_concept_end_indices = ", token_concept_end_indices)

	return token_concept_start_indices, token_concept_end_indices


########################################
# Step #6: updateTokenConceptIds
#		  (Vectorized approach)
########################################

def updateTokenConceptIds_Optimized(
	token_concept_ids, 
	token_concept_start_indices, 
	token_concept_end_indices
):
	"""
	Original Step 6:
	  - if localConceptColumnExpertsApplyToAllTokens,
		for each idx, if token_concept_ids[idx] == localConceptColumnExpertsNoColumnID, 
		we find the noun in [start..end], pick the first noun's ID, and set it.
	
	Vectorized approach:
	  1) Identify all positions that are still 0 (localConceptColumnExpertsNoColumnID).
	  2) For each such position, we look up the range [start..end].
	  3) We find the *first* noun in that range (if any).
	  4) If none is found, it remains 0, else we set the ID.

	True \u201cpure\u201d vectorization for \u201cscan from start..end for the first noun\u201d is tricky. 
	One approach is to create a big block for all positions, do a gather from the 
	range [start..end], and find the first non-zero concept ID. 
	We'll do a somewhat advanced approach with gather + cumulative checks.

	If you do not want to implement a more complex gather+scan logic, you might 
	do a small loop that only loops over the *tokens that need filling* 
	rather than over every token. But below is a demonstration that avoids 
	explicit Python for-loops over tokens.
	"""
	if not localConceptColumnExpertsApplyToAllTokens:
		return token_concept_ids

	# 1) We'll identify which tokens have 0
	zero_mask = (token_concept_ids == localConceptColumnExpertsNoColumnID)
	if not zero_mask.any():
		# No updates needed
		return token_concept_ids

	# Build arrays for each token: we want the segment [s..e].
	s = token_concept_start_indices
	e = token_concept_end_indices
	positions = pt.arange(token_concept_ids.shape[0]).unsqueeze(1)  # shape [seq_length, 1]
	# We'll expand so that for each index i, we can build a range [s_i..e_i].
	# That is tricky in a single shot. Instead, we can do a bounding approach up to max(e).

	# We'll do something like:
	#   max_end = e.max()
	#   big_range = pt.arange(max_end+1).unsqueeze(0).expand(seq_length, max_end+1)
	#   within_range_mask = (big_range >= s.unsqueeze(1)) & (big_range <= e.unsqueeze(1))
	# shape [seq_length, max_end+1]
	max_end = e.max().item()
	big_range = pt.arange(max_end + 1).unsqueeze(0).expand(token_concept_ids.shape[0], -1)
	within_range_mask = (big_range >= s.unsqueeze(1)) & (big_range <= e.unsqueeze(1))
	# For positions i that have e_i < big_range dimension, that will be false automatically above e_i.

	# Next, we gather token_concept_ids for the entire row i, but only up to max_end:
	concept_ids_pad = token_concept_ids[:max_end+1]  # shape [max_end+1]
	# Actually we want a 2D gather. We'll do an index-based gather:
	# but we have row i and column big_range[i, *]. We can flatten?

	# Easiest approach: replicate token_concept_ids in row dimension:
	full_ids_expanded = token_concept_ids.unsqueeze(0).expand(token_concept_ids.shape[0], -1)
	# shape [seq_length, seq_length]
	# But we only need up to max_end, so let's do:
	full_ids_expanded = full_ids_expanded[:, :max_end+1]  # shape [seq_length, max_end+1]

	# Now for row i, `within_range_mask[i]` is True wherever index is in [s_i.. e_i].
	# We'll zero out everything else. Then we look for the first non-zero concept ID 
	# in that row's portion. 
	# We'll do a boolean mask to keep only the elements in the range, then do something 
	# like a cumsum to detect the first non-zero.
	# For the row i, the \u201celement indices\u201d are big_range[i], and the concept IDs are full_ids_expanded[i].

	in_range_ids = full_ids_expanded * within_range_mask.long()  # shape [seq_length, max_end+1]
	# Now we want the first non-zero in each row.
	# We can do "non_zero_mask = in_range_ids != 0" and then "argmax" trick. 
	# But if there's no non-zero, we get 0. 
	nonzero_mask = (in_range_ids != 0)

	any_nonzero = nonzero_mask.any(dim=1)  # shape [seq_length]
	# first_nonzero_idx = argmax(nonzero_mask, dim=1). But that will give the first True index in [0..max_end].
	first_nonzero_idx = pt.argmax(nonzero_mask.long(), dim=1)  # shape [seq_length]

	# If any_nonzero[i] is False => no concept ID in that column => remain 0.
	# Otherwise, the ID is in_range_ids[i, first_nonzero_idx[i]].
	gather_ids = in_range_ids[pt.arange(token_concept_ids.shape[0]), first_nonzero_idx]
	# shape [seq_length]

	# The final ID is either 0 if no noun found, or the ID if found
	final_ids = pt.where(any_nonzero, gather_ids, pt.tensor(localConceptColumnExpertsNoColumnID))

	# Now we only apply this update for positions that originally had zero_mask = True 
	# (i.e. we only override tokens that had 0).
	updated_token_concept_ids = pt.where(zero_mask, final_ids, token_concept_ids)

	if(debugDetectLocalConceptColumns):
		print("token_concept_ids = ", updated_token_concept_ids)
	
	return updated_token_concept_ids



# A helper to fetch the dictionary index for a noun
def get_concept_id(noun_dict, token):
	lemma_lower = token.lemma_.lower()
	if lemma_lower in noun_dict:
		concept_id = noun_dict[lemma_lower] + 1	#assumes localConceptColumnExpertsNoDictionaryNounID==0
	else:
		concept_id = localConceptColumnExpertsNoDictionaryNounID
	return concept_id

def generateConceptColumnIndicesSerial(device, tokenizer, batch_input_ids, batch_offsets, identify_type="identify_both_columns"):
	"""
	Generates the following for each sample in the batch:
	1) conceptColumnStartIndices: A tensor indicating, for each token, the start index of its concept column.
	2) conceptColumnEndIndices: A tensor indicating, for each token, the end index of its concept column.
	3) conceptColumnIDs: A tensor of noun IDs (from a given common dictionary) for each token.
		- If a token is recognized as a noun, conceptColumnIDs[token_idx] = noun_dict[lowercased_token_text]+1.
		- If the noun isn't in the dictionary, or if it's not a noun, conceptColumnIDs[token_idx] = 0.
	"""
	
	batch_concept_start_indices = []
	batch_concept_end_indices = []
	batch_concept_ids = []
	
	batch_size = len(batch_input_ids)
	for sample_index in range(batch_size):
		input_ids_sample = batch_input_ids[sample_index]
		offsets_sample = batch_offsets[sample_index]
		seq_length = input_ids_sample.shape[0]
		
		#Step 1: get tokens
		tokens = getTokens(tokenizer, input_ids_sample)
		
		#Step 2: Process the text with spaCy
		doc = processTextWithSpacy(nlp, tokens)
		
		#Step 3: Create a mapping from character positions to spaCy tokens
		char_to_token = createMappingFromCharacterPositionsToSpacyTokens(doc, offsets_sample)

		#Step 4a: populate conceptColumnStartIndexTensor and conceptColumnEndIndexTensor
		conceptColumnStartIndexTensor, conceptColumnEndIndexTensor = populateConceptColumnStartAndEndIndexTensor(tokenizer, char_to_token, seq_length, noun_pos_tags, identify_type, input_ids_sample)

		#Step 4b: populate token_concept_ids and first_noun_idx
		token_concept_ids, first_noun_idx = populateTokenConceptIdsAndFirstNounIdx(noun_dict, char_to_token, seq_length, noun_pos_tags)
		
		#Step 5: populate token_concept_start_indices and token_concept_end_indices
		token_concept_start_indices, token_concept_end_indices = populateTokenConceptStartAndEndIndices(conceptColumnStartIndexTensor, conceptColumnEndIndexTensor, seq_length, first_noun_idx, identify_type)

		#Step 6: if localConceptColumnExpertsApplyToAllTokens, update token_concept_ids for non-noun tokens using the noun in their column
		token_concept_ids = updateTokenConceptIds(token_concept_ids, token_concept_start_indices, token_concept_end_indices) 
		
		if(debugDetectLocalConceptColumns):
			exit()
		
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

	if(debugDetectLocalConceptColumns):
		print("batch_input_ids = ", batch_input_ids)
		print("conceptColumnStartIndices = ", conceptColumnStartIndices)
		print("conceptColumnEndIndices = ", conceptColumnEndIndices)
		print("conceptColumnIDs = ", conceptColumnIDs)

	return conceptColumnStartIndices, conceptColumnEndIndices, conceptColumnIDs

def getTokens(tokenizer, input_ids_sample):
	#Step 1: get tokens
	tokens = tokenizer.decode(input_ids_sample, skip_special_tokens=True)
	
	if(debugDetectLocalConceptColumns):
		print("tokens =", tokens)
		
	return tokens

def processTextWithSpacy(nlp, tokens):
	#Step 2: Process the text with spaCy
	doc = nlp(tokens)
	
	if(debugDetectLocalConceptColumns):
		print("doc =", doc)
		
	return doc
	
def createMappingFromCharacterPositionsToSpacyTokens(doc, offsets_sample):
	#Step 3: Create a mapping from character positions to spaCy tokens
	
	char_to_token = []
	for idx, (start_char, end_char) in enumerate(offsets_sample):
		if start_char == end_char:
			# Special tokens (e.g., <s>, </s>)
			char_to_token.append(None)
		else:
			# Find the corresponding spaCy token
			foundToken = False
			for token in doc:
				if token.idx <= start_char and token.idx + len(token) >= end_char:
					char_to_token.append(token)
					foundToken = True
					break
			if not foundToken:
				print("not foundToken: start_char = ", start_char, ", end_char = ", end_char)  
				char_to_token.append(None)
	
	if(debugDetectLocalConceptColumns):
		print("char_to_token = ", char_to_token)
	
	return char_to_token

def populateConceptColumnStartAndEndIndexTensor(tokenizer, char_to_token, seq_length, noun_pos_tags, identify_type, input_ids_sample):
	#Step 4a: populate conceptColumnStartIndexTensor and conceptColumnEndIndexTensor

	# Initialize lists for concept columns
	if(identify_type=="identify_both_columns"):
		conceptColumnStartIndexList = [0]
	elif(identify_type=="identify_previous_column"):
		conceptColumnStartIndexList = [0]
	elif(identify_type=="identify_next_column"):
		conceptColumnStartIndexList = []	#a concept column will not be assigned for these first indices (before first noun)
	conceptColumnEndIndexList = []

	# 1) Build a pos_mask (1 if noun, 0 if not).
	#	Since char_to_token[i] is a spaCy token or None, we do a comprehension in Python 
	#	to store pos. We'll keep that minimal.
	#	We'll *then* do the logic for concept columns with a mostly vector/tensor approach.
	pos_list = []
	for tok in char_to_token:
		if tok is not None and tok.pos_ in noun_pos_tags:
			pos_list.append(1)
		else:
			pos_list.append(0)
	pos_t_noun = pt.tensor(pos_list, dtype=pt.long)  # shape [seq_length]
	
	for idx in range(seq_length):
		token = char_to_token[idx]
		if token is not None:
			pos = token.pos_
			if pos in noun_pos_tags:
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

	assert len(conceptColumnStartIndexList) == len(conceptColumnEndIndexList)

	if(debugDetectLocalConceptColumns):
		print("isNoun = ", pos_t_noun)
		print("conceptColumnStartIndexList = ", conceptColumnStartIndexList)
		print("conceptColumnEndIndexList = ", conceptColumnEndIndexList)
	
	return conceptColumnStartIndexList, conceptColumnEndIndexList

def populateTokenConceptIdsAndFirstNounIdx(noun_dict, char_to_token, seq_length, noun_pos_tags):
	#Step 4b: populate token_concept_ids and first_noun_idx
		
	# Track conceptColumnIDs for each token
	concept_indices_list = []  # Indices of concept words in tokens
	token_concept_ids = pt.ones(seq_length, dtype=pt.long)*localConceptColumnExpertsNoColumnID
	first_noun_idx = 0
	foundFirstNoun = False
	
	for idx in range(seq_length):
		token = char_to_token[idx]
		if token is not None:
			pos = token.pos_
			if pos in noun_pos_tags:
				if not foundFirstNoun:
					first_noun_idx = idx
					foundFirstNoun = True
				concept_indices_list.append(idx)
				token_concept_ids[idx] = get_concept_id(noun_dict, token)

	if(debugDetectLocalConceptColumns):
		print("token_concept_ids = ", token_concept_ids)
		print("first_noun_idx = ", first_noun_idx)
					
	return token_concept_ids, first_noun_idx
		
def populateTokenConceptStartAndEndIndices(conceptColumnStartIndexList, conceptColumnEndIndexList, seq_length, first_noun_idx, identify_type):
	#Step 5: populate token_concept_start_indices and token_concept_end_indices

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
	
	if(debugDetectLocalConceptColumns):
		print("token_concept_start_indices = ", token_concept_start_indices)
		print("token_concept_end_indices = ", token_concept_end_indices)
	
	return token_concept_start_indices, token_concept_end_indices

def updateTokenConceptIds(token_concept_ids, token_concept_start_indices, token_concept_end_indices):
	#Step 6: if localConceptColumnExpertsApplyToAllTokens, update token_concept_ids for non-noun tokens using the noun in their column

	if(localConceptColumnExpertsApplyToAllTokens):
		# Fill in concept IDs for non-noun tokens using the noun in their column.
		for idx in range(seq_length):
			# If it's still localConceptColumnExpertsNoColumnID, we need to propagate from the column's noun
			if token_concept_ids[idx] == localConceptColumnExpertsNoColumnID:
				s = token_concept_start_indices[idx].item()
				e = token_concept_end_indices[idx].item()
				# Within s..e, there should be exactly one noun token (or none).
				# We'll take the first noun we find there:
				column_id = localConceptColumnExpertsNoColumnID
				for col_idx in range(s, e + 1):
					if token_concept_ids[col_idx] != localConceptColumnExpertsNoColumnID:
						column_id = token_concept_ids[col_idx].item()
						break
				token_concept_ids[idx] = column_id
		
	if(debugDetectLocalConceptColumns):	
		print("token_concept_ids = ", token_concept_ids)
	
	return token_concept_ids
						



