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

import torch
import torch as pt
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import nncustom
from sortedcontainers import SortedDict
import os

from transformers.activations import ACT2FN 

from TSBNLPpt_globalDefs import *





if(localConceptColumnAttention):

	def applyLocalConceptColumnAttention(self, hidden_states, attention_mask, head_mask, query_layer, key_layer, value_layer, conceptColumnStartIndices, conceptColumnEndIndices):
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

	if(localConceptColumnExpertsStructure=="nnModuleList"):
		class ExpertMLP(nn.Module):
			"""
			Simple feed-forward for one expert:
			   hidden_size -> intermediate_size -> hidden_size
			"""
			def __init__(self, hidden_size, intermediate_size, activation_fn=F.gelu):
				super().__init__()
				self.expert_weight_1 = nn.Parameter(torch.empty(intermediate_size, hidden_size))
				self.expert_bias_1 = nn.Parameter(torch.empty(intermediate_size))
				self.expert_weight_2 = nn.Parameter(torch.empty(hidden_size, intermediate_size))
				self.expert_bias_2 = nn.Parameter(torch.empty(hidden_size))
				#self.dense1 = nn.Linear(hidden_size, intermediate_size)
				#self.dense2 = nn.Linear(intermediate_size, hidden_size)
				
				# Initialize parameters
				self.reset_parameters()

				self.activation_fn = activation_fn
				self.layernorm = nn.LayerNorm(hidden_size)
				self.dropout = nn.Dropout(0.1)

			def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
				# hidden_states: (group_size, hidden_size)
				
				x = torch.addmm(self.expert_bias_1, hidden_states, self.expert_weight_1.transpose(0, 1))
				x = self.activation_fn(x)
				x = torch.addmm(self.expert_bias_2, x, self.expert_weight_2.transpose(0, 1))
				#x = self.dense1(hidden_states)
				#x = self.activation_fn(x)
				#x = self.dense2(x)
				
				x = self.dropout(x)
				# residual connection + layer norm
				x = self.layernorm(x + hidden_states)
				return x
				
			def reset_parameters(self):
				# A simple initialization scheme (like xavier) for demonstration
				nn.init.xavier_uniform_(self.expert_weight_1)
				nn.init.zeros_(self.expert_bias_1)
				nn.init.xavier_uniform_(self.expert_weight_2)
				nn.init.zeros_(self.expert_bias_2)

	
	class ParallelExpertsMLP(nn.Module):
		"""
		A 'parallel' expert MLP block that contains multiple experts' parameters in a single tensor.
		All tokens are processed in one batched pass using advanced indexing.

		Args:
			num_experts: number of experts
			num_experts_cpu: max number of experts available in cpu ram
			hidden_size: original model hidden size
			expert_intermediate_size: the small intermediate dimension of each expert
			no_expert_id: the special ID (e.g. -1) that indicates 'no expert' / skip
			activation: activation function for intermediate layer (e.g. GELU, ReLU)
		
		Description:
			The code applies a concept expert MLPs to the tokens, where each expert comprise two linear layers (emulating ordinary transformer MLP). It uses expert_ids (of shape [batch_size, seq_length]) to determine which expert MLP to use for each token in the sequence. 
			The expert MLP layers have a standard input/output size (of hidden_size features), and an intermediate layer size (of expert_intermediate_size features). 
			All tokens in the sequence are executed in parallel (the necessary experts are loaded from [cpu] parameters experts_weight_1/experts_weight_2/experts_bias_1/experts_bias_2 to W1, b1, W2, b2 and are executed in parallel on the gpu).
			
			if(localConceptColumnExpertsStoreRAM):
				1) Assume approximately 96GB of GPU ram 
			else:
				There are num_experts total expert MLPs stored on disk (num_experts_cpu recently accessed experts available in CPU ram), each with their own unique parameters. 

				1) Assume approximately 1TB of SSD storage space to store all experts (indexed by expert id). These experts will be created and accessed on demand. The location of the SSD folder is called conceptExpertsPathName. Each expert is saved to SSD for a particular transformer block (use layer_index).
				2) Assume approximately 64GB of CPU/GPU ram (depending on localConceptColumnExpertsStoreCPU) to store the most recently accessed expert MLP parameters (experts_weight_1, experts_weight_2, experts_bias_1, experts_bias_2), where each MLP parameter tensor contains num_experts_cpu experts. 
					The most recently accessed experts are the last numerOfRecentlyAccessedExperts (eg last 1000 experts) that were brought into GPU ram to process the sequence token experts, thus num_experts_cpu = numerOfRecentlyAccessedExperts. 
					Each expert stored in CPU ram uses last_access_time to indicate the last time it was brought into GPU ram (where last_access_time is derived from conceptColumnData['batchIndex']). 

				To manage the CPU ram memory the code creates i) a tensor called experts_cpu_map, ii) a multi SortedDict from sortedcontainers called last_access_time_experts, and iii) a dictionary called expert_ids_in_cpu, iv) a variable called num_experts_cpu_currently_loaded, and a dictionary called expert_id_cpu_available;
				i) a pytorch tensor called experts_cpu_map of shape [num_experts_cpu, 2] to record the original expert_id and last_access_time of each cpu expert. experts_cpu_map should be initialised to -1 in all its elements. If there is no expert loaded at a particular row (ie expert_id_cpu) of experts_cpu_map, then assign an expert_id of -1 and a last_access_time of -1.
				ii) a multi SortedDict from sortedcontainers called last_access_time_experts (key: last_access_time, value:[list of expert_id_cpu]). Note that a "multi" dictionary uses a list structure for its value, such that it can contain more than one element for a given key. The value of any given last_access_time key will be a list of expert_id_cpu. last_access_time_experts should be initialised as empty.
				iii) a standard python dict called expert_ids_in_cpu (key: expert_id, value: expert_id_cpu). expert_ids_in_cpu should be initialised as empty.
				iv) a python variable called num_experts_cpu_currently_loaded, which designates the number of cpu experts are currently loaded. num_experts_cpu_currently_loaded should be initialised to 0.
				v) a standard python dict called expert_id_cpu_available (key: expert_id_cpu, value: True). expert_id_cpu_available should be initialised as filled with value True, of size num_experts_cpu.
		"""
		def __init__(
			self,
			num_experts: int,
			num_experts_cpu: int,
			hidden_size: int,
			expert_intermediate_size: int,
			no_expert_id: int = -1,
			activation = F.gelu
		):
			super().__init__()
			self.num_experts = num_experts
			self.num_experts_cpu = num_experts_cpu
			self.hidden_size = hidden_size
			self.expert_intermediate_size = expert_intermediate_size
			self.no_expert_id = no_expert_id
			self.activation = activation

			if(localConceptColumnExpertsStructure=="nnParameterList"):
				self.experts_weight_1 = nn.ParameterList([nn.Parameter(nn.init.xavier_uniform_(torch.empty(expert_intermediate_size, hidden_size))) for _ in range(num_experts_cpu)])
				self.experts_bias_1 = nn.ParameterList([nn.Parameter(nn.init.zeros_(torch.empty(expert_intermediate_size))) for _ in range(num_experts_cpu)])
				self.experts_weight_2 = nn.ParameterList([nn.Parameter(nn.init.xavier_uniform_(torch.empty(hidden_size, expert_intermediate_size))) for _ in range(num_experts_cpu)])
				self.experts_bias_2 = nn.ParameterList([nn.Parameter(nn.init.zeros_(torch.empty(hidden_size))) for _ in range(num_experts_cpu)])
			elif(localConceptColumnExpertsStructure=="nnParameter"):
				# 1) For the first linear: (out_dim = expert_intermediate_size, in_dim = hidden_size)
				# We store all experts in one big tensor of shape: (num_experts_cpu, expert_intermediate_size, hidden_size)
				self.experts_weight_1 = nn.Parameter(torch.empty(num_experts_cpu, expert_intermediate_size, hidden_size))
				self.experts_bias_1 = nn.Parameter(torch.empty(num_experts_cpu, expert_intermediate_size))

				# 2) For the second linear: (out_dim = hidden_size, in_dim = expert_intermediate_size)
				# Stacked in shape: (num_experts_cpu, hidden_size, expert_intermediate_size)
				self.experts_weight_2 = nn.Parameter(torch.empty(num_experts_cpu, hidden_size, expert_intermediate_size))
				self.experts_bias_2 = nn.Parameter(torch.empty(num_experts_cpu, hidden_size))
				
				# Initialize parameters
				self.reset_parameters()	
			elif(localConceptColumnExpertsStructure=="nnModuleList"):
				self.experts = nn.ModuleList([ExpertMLP(hidden_size, expert_intermediate_size) for _ in range(num_experts_cpu)])

			# Optionally, we could add LayerNorm parameters for each expert, or a single LN, etc.
			# but here we only do a final residual + dropout, or you can do LN if you like.

			self.dropout = nn.Dropout(0.1)
			self.layer_norm = nn.LayerNorm(hidden_size)
			
			if(not localConceptColumnExpertsStoreRAM):
				# memory manangement structures
				self.experts_cpu_map = torch.ones((num_experts_cpu, 2), dtype=torch.long) * -1	#shape [num_experts_cpu, 2] to record the original expert_id and last_access_time of each cpu expert
				self.last_access_time_experts = SortedDict()	#key: last_access_time, value:[list of expert_id_cpu]
				self.expert_ids_in_cpu = {}	#key: expert_id, value: expert_id_cpu
				self.num_experts_cpu_currently_loaded = 0
				self.expert_id_cpu_available = {i: True for i in range(num_experts_cpu)}	#key: expert_id_cpu, value: True

		if(localConceptColumnExpertsStoreCPU):
			def to(self, *args, **kwargs):
				# First, apply the default `.to(...)` behavior to everything
				super_result = super().to(*args, **kwargs)
				# Then force all experts back on CPU
				if(localConceptColumnExpertsStructure=="nnModuleList"):
					for expert in self.experts:
						expert.to('cpu')
				else:
					self.experts_weight_1.to('cpu')
					self.experts_weight_2.to('cpu')
					self.experts_bias_1.to('cpu')
					self.experts_bias_2.to('cpu')
				return super_result
		
		if(localConceptColumnExpertsStructure=="nnParameter"):
			def reset_parameters(self):
				# A simple initialization scheme (like xavier) for demonstration
				nn.init.xavier_uniform_(self.experts_weight_1)
				nn.init.zeros_(self.experts_bias_1)
				nn.init.xavier_uniform_(self.experts_weight_2)
				nn.init.zeros_(self.experts_bias_2)
		
		if(localConceptColumnExpertsGroupTokensByExpert):
			if(localConceptColumnExpertsStructure=="nnModuleList"):
				def forward_single_expert(self, x_e: torch.Tensor, expert_id: int) -> torch.Tensor:
					"""
					x_e:	  (group_size, hidden_size)
					expert_id is an integer in [0..num_experts-1]
					"""
					# Simply forward through that expert's MLP
					out_e = self.experts[expert_id](x_e)
					return out_e
			else:
				def forward_single_expert(self, x_e: torch.Tensor, expert_id: int) -> torch.Tensor:
					"""
					x_e: (group_size, hidden_size)
					expert_id: integer in [0..num_experts-1]
					"""
					# Extract that expert's parameters
					# w1_e shape: (intermediate_size, hidden_size)
					w1_e = self.experts_weight_1[expert_id]
					b1_e = self.experts_bias_1[expert_id]
					if(localConceptColumnExpertsStoreCPU):
						w1_e = w1_e.to(device=x_e.device, non_blocking=True)
						b1_e = b1_e.to(device=x_e.device, non_blocking=True)
						
					# w2_e shape: (hidden_size, intermediate_size)
					w2_e = self.experts_weight_2[expert_id]
					b2_e = self.experts_bias_2[expert_id]
					if(localConceptColumnExpertsStoreCPU):
						w2_e = w2_e.to(device=x_e.device, non_blocking=True)
						b2_e = b2_e.to(device=x_e.device, non_blocking=True)
						
					# First matmul: out1 = x_e @ w1_e^T + b1_e
					out1 = torch.addmm(b1_e, x_e, w1_e.transpose(0, 1))
					out1 = self.activation(out1)

					# Second matmul: out2 = out1 @ w2_e^T + b2_e
					out2 = torch.addmm(b2_e, out1, w2_e.transpose(0, 1))

					# Residual + LN
					out2 = self.dropout(out2)
					out2 = self.layer_norm(out2 + x_e)

					return out2

		
		def forward(self, hidden_states: torch.Tensor, expert_ids: torch.LongTensor, layer_index, batchIndex):
			"""
			hidden_states: shape (batch_size, seq_len, hidden_size)
			expert_ids:	shape (batch_size, seq_len), integer in [0..num_experts-1], or = no_expert_id (-1)
			"""

			if(localConceptColumnExpertsStoreRAM):
				expert_ids_cpu = expert_ids
			else:
				#print("offloadLeastRecentlyAccessedExpertsToSSD: num_experts_cpu_currently_loaded = ", self.num_experts_cpu_currently_loaded)
				self.offloadLeastRecentlyAccessedExpertsToSSD(layer_index, batchIndex)
				#print("loadRequiredExpertsFromSSD: num_experts_cpu_currently_loaded = ", self.num_experts_cpu_currently_loaded)
				expert_ids_cpu = self.loadRequiredExpertsFromSSD(expert_ids, layer_index, batchIndex)

			bsz, seq_len, hdim = hidden_states.shape
			N = bsz * seq_len  # Flatten

			# Flatten the hidden states to (N, hidden_size)
			hidden_states_flat = hidden_states.view(N, hdim)
			expert_ids_cpu_flat = expert_ids_cpu.view(-1)  # (N,)

			# We'll create an output buffer of shape (N, hidden_size)
			output_flat = hidden_states_flat.clone()

			# ============== 1) Identify tokens to skip vs. tokens to process  ==============
			skip_mask = (expert_ids_cpu_flat == self.no_expert_id)
			process_mask = ~skip_mask  # Invert

			if process_mask.any():

				# Indices of tokens that go to *some* expert
				process_indices = process_mask.nonzero(as_tuple=True)[0]  # shape ~ (num_process_tokens,)

				# Gather the expert IDs for just those tokens
				token_expert_ids = expert_ids_cpu_flat[process_indices]  # shape (num_process_tokens,)

				# Gather the input states that need processing
				x = hidden_states_flat[process_indices]  # shape (num_process_tokens, hidden_size)
					
				if(localConceptColumnExpertsGroupTokensByExpert):

					sorted_expert_ids, sort_order = torch.sort(token_expert_ids)

					x_sorted = x[sort_order]

					expert_outputs = torch.empty_like(x_sorted)
					current_start = 0

					# We'll iterate over contiguous blocks of x_sorted that share the same expert_id
					import itertools
					for expert_id, group_iter in itertools.groupby(sorted_expert_ids, lambda e: e.item()):
						group_size = len(list(group_iter))
						start = current_start
						end = start + group_size
						current_start = end

						# Subset of tokens for this expert
						x_e = x_sorted[start:end]  # shape (group_size, hidden_size)

						# Single forward pass for these tokens
						out_e = self.forward_single_expert(x_e, expert_id)

						# Place in the final buffer
						expert_outputs[start:end] = out_e

					# unsort to the original order among the processed tokens
					_, unsort_order = torch.sort(sort_order)
					process_output = expert_outputs[unsort_order]

					# scatter back
					output_flat[process_indices] = process_output

				else:					
					num_process_tokens = x.size(0)

					if(localConceptColumnExpertsStructure=="nnParameterList"):
						token_expert_ids_list = token_expert_ids.tolist()
						W1list = [self.experts_weight_1[i] for i in token_expert_ids_list]
						b1list = [self.experts_bias_1[i] for i in token_expert_ids_list]
						W2list = [self.experts_weight_2[i] for i in token_expert_ids_list]
						b2list = [self.experts_bias_2[i] for i in token_expert_ids_list]
						W1 = torch.stack(W1list, dim=0)  # shape (num_process_tokens, EIS, H)
						b1 = torch.stack(b1list, dim=0)	# shape (num_process_tokens, EIS)
						W2 = torch.stack(W2list, dim=0) # shape (num_process_tokens, hidden_size, EIS)
						b2 = torch.stack(b2list, dim=0)	# shape (num_process_tokens, hidden_size)
					elif(localConceptColumnExpertsStructure=="nnParameter"):
						# We want W1 per token, of shape (num_process_tokens, expert_intermediate_size, hidden_size)
						# so we do advanced indexing into self.experts_weight_1 with [token_expert_ids].
						W1 = self.experts_weight_1[token_expert_ids]  # shape (num_process_tokens, EIS, H)
						b1 = self.experts_bias_1[token_expert_ids]	# shape (num_process_tokens, EIS)
						W2 = self.experts_weight_2[token_expert_ids]  # shape (num_process_tokens, hidden_size, EIS)
						b2 = self.experts_bias_2[token_expert_ids]	# shape (num_process_tokens, hidden_size)
					else:
						printe("!localConceptColumnExpertsGroupTokensByExpert does not support any other method of localConceptColumnExpertsStructure")

					if(localConceptColumnExpertsStoreCPU):
						W1 = W1.to(device=hidden_states.device, non_blocking=True)
						b1 = b1.to(device=hidden_states.device, non_blocking=True)
						W2 = W2.to(device=hidden_states.device, non_blocking=True)
						b2 = b2.to(device=hidden_states.device, non_blocking=True)

					# We'll do a batched matmul: (N, 1, H) x (N, H, EIS) -> (N, 1, EIS)
					x_3d = x.unsqueeze(1)	  # (num_process_tokens, 1, hidden_size)
					W1_3d = W1.transpose(1, 2) # (num_process_tokens, hidden_size, expert_intermediate_size)

					mm1 = torch.bmm(x_3d, W1_3d).squeeze(1)  # (num_process_tokens, expert_intermediate_size)
					mm1 = mm1 + b1

					# Apply activation
					mm1 = self.activation(mm1)


					mm1_3d = mm1.unsqueeze(1)	 # (num_process_tokens, 1, EIS)
					W2_3d = W2.transpose(1, 2)	# (num_process_tokens, EIS, hidden_size)

					mm2 = torch.bmm(mm1_3d, W2_3d).squeeze(1)  # (num_process_tokens, hidden_size)
					mm2 = mm2 + b2

					# ~~~~~~~~~~~~~ 4) Residual + LN ~~~~~~~~~~~~~
					# Compare with the original x (the residual). We can use a fresh residual from hidden_states_flat.
					# But usually in a Transformer block, the "input_tensor" is the pre-FFN output. Let's do that:
					mm2 = self.dropout(mm2)
					if(not localConceptColumnExpertsApplyWithSharedMLPthenResidual):
						residual = x
						mm2 = self.layer_norm(mm2*(1-localConceptColumnExpertsResidualRatio) + residual*localConceptColumnExpertsResidualRatio)

					# ~~~~~~~~~~~~~ 5) Scatter results back ~~~~~~~~~~~~~
					output_flat[process_indices] = mm2

					# Meanwhile, tokens with skip_mask==True remain as their original input (no change).
					# (We initialized output_flat as a copy of hidden_states_flat, so it\u2019s already \u201cunchanged\u201d for those.)

			# Reshape output back
			output = output_flat.view(bsz, seq_len, hdim)
			return output

		def loadRequiredExpertsFromSSD(self, expert_ids, layer_index, last_access_time):
			'''
			function loadRequiredExpertsFromSSD(): every expert_id in expert_ids is checked to see whether it is currently available in CPU ram (ie check if expert_id in expert_ids_in_cpu). If the expert is not currently available in CPU ram, then;
			- assign a new expert_id_cpu index using expert_id_cpu_available
			- update all the memory management structures experts_cpu_map, last_access_time_experts, expert_ids_in_cpu, num_experts_cpu_currently_loaded, expert_id_cpu_available. Here is python code to achieve the required functionality (use this);
			- check if the expert is available on SSD (expert_id and layer_index);
			 * If the expert is not available on SSD, initialise the expert parameters (see ParallelExpertsMLP to infer the correct initialisation shapes).
			 * If the expert is available on SSD, execute loadExpertFromSSD(expert_id) to load the expert from SSD (at expert_id) to CPU ram (at expert_id_cpu). Note that expert must be loaded from SSD for a particular expert_id and layer_index (transformer block). 
			- update the parameters; expert_weight_1[expert_id_cpu], expert_weight_2[expert_id_cpu], expert_bias_1[expert_id_cpu], expert_bias_2[expert_id_cpu].
			'''
			
			batch_size = expert_ids.shape[0]
			seq_length = expert_ids.shape[1]
			
			expert_ids_cpu_list = []
			for sample_index in range(batch_size):
				expert_ids_cpu_sample_list = []
				#print("\t sample_index = ", sample_index)
				for expert_id_index in range(seq_length):
					expert_id = expert_ids[sample_index, expert_id_index].item()
					#print("\t\t expert_id_index = ", expert_id_index)
					#print("\t\t expert_id = ", expert_id)
					if expert_id == self.no_expert_id:
						expert_id_cpu = expert_id
					else:
						if expert_id in self.expert_ids_in_cpu:
							expert_id_cpu = self.expert_ids_in_cpu[expert_id]
							if(debugLocalConceptColumnExpertsFileIO):
								#print("debugLocalConceptColumnExpertsFileIO: loadExpertFromSSD")
								self.loadExpertFromSSD(expert_id_cpu, expert_id, layer_index)
						else:
							expert_id_cpu = self.getFirstKeyInDict(self.expert_id_cpu_available)	#assign a new expert_id_cpu index (first key in expert_id_cpu_available)
							#update all the memory management structures experts_cpu_map, last_access_time_experts, expert_ids_in_cpu, num_experts_cpu_currently_loaded, expert_id_cpu_available;
							#print("expert_id = ", expert_id)
							assert expert_id_cpu < self.num_experts_cpu, "expert_id_cpu = " + str(self.expert_id_cpu)
							self.experts_cpu_map[expert_id_cpu][0] = expert_id
							self.experts_cpu_map[expert_id_cpu][1] = last_access_time
							if not last_access_time in self.last_access_time_experts: 
								self.last_access_time_experts[last_access_time] = []	#assign value to an empty list
							self.last_access_time_experts[last_access_time].append(expert_id_cpu)
							self.expert_ids_in_cpu[expert_id] = expert_id_cpu
							self.num_experts_cpu_currently_loaded += 1
							self.expert_id_cpu_available.pop(expert_id_cpu)
							if(self.isExpertAvailableOnSSD(expert_id, layer_index)):
								self.loadExpertFromSSD(expert_id_cpu, expert_id, layer_index)
							else:
								self.initialiseExpertCPU(expert_id_cpu)
								if(debugLocalConceptColumnExpertsFileIO):
									#print("debugLocalConceptColumnExpertsFileIO: saveExpertToSSD")
									self.saveExpertToSSD(expert_id_cpu, expert_id, layer_index)
								
					expert_ids_cpu_sample_list.append(expert_id_cpu)
				expert_ids_cpu_sample = torch.tensor(expert_ids_cpu_sample_list)
				expert_ids_cpu_list.append(expert_ids_cpu_sample)
			expert_ids_cpu = torch.stack(expert_ids_cpu_list, dim=0)
			return expert_ids_cpu
			
		def offloadLeastRecentlyAccessedExpertsToSSD(self, layer_index, last_access_time):
			'''
			function offloadLeastRecentlyAccessedExpertsToSSD():
			- identify the least recently accessed experts, which are contained at the start of multi SortedDict last_access_time_experts (note the lowest value keys are stored at the start of a SortedDict). The number of access time keys (containing lists of expert_id_cpu) to offload depends on the amount of CPU ram currently available. Ensure that there is always enough CPU ram to load the required number of new experts for a given batch (its sequences); i.e. num_experts_cpu_currently_loaded/num_experts_cpu < 0.9 (where 1.0-0.9 = 6GB GPU ram / 60GB CPU ram).
			- save these experts to SSD via function saveExpertToSSD. Note that experts must be saved to SSD for a particular expert_id and layer_index (transformer block).
			- clear these experts from the memory management structures experts_cpu_map, last_access_time_experts, expert_ids_in_cpu, num_experts_cpu_currently_loaded, expert_id_cpu_available;
			- there is no need to clear the parameters; expert_weight_1[expert_id_cpu], expert_weight_2[expert_id_cpu], expert_bias_1[expert_id_cpu], expert_bias_2[expert_id_cpu] (these will be overwritten in the future).
			'''
			while((self.num_experts_cpu_currently_loaded/self.num_experts_cpu > 1.0-ratioOfGPUtoCPUramAvailableForExperts) and self.num_experts_cpu_currently_loaded > 0): 	#depends on ratio of GPU to CPU ram available for experts
				oldest_expert_id_cpu_list_key = self.getFirstKeyInDict(self.last_access_time_experts)
				oldest_expert_id_cpu_list = self.getFirstValueInDict(self.last_access_time_experts)
				atLeastOneElementInList = False
				for old_expert_id_cpu in oldest_expert_id_cpu_list:
					atLeastOneElementInList = True
					old_expert_id = self.experts_cpu_map[old_expert_id_cpu][0].item()
					#print("old_expert_id = ", old_expert_id)
					self.saveExpertToSSD(old_expert_id_cpu, old_expert_id, layer_index)
					self.experts_cpu_map[old_expert_id_cpu][0] = -1
					self.experts_cpu_map[old_expert_id_cpu][1] = -1
					self.expert_ids_in_cpu.pop(old_expert_id)
					self.num_experts_cpu_currently_loaded -= 1
					self.expert_id_cpu_available[old_expert_id_cpu] = True
				self.last_access_time_experts.pop(oldest_expert_id_cpu_list_key)

		def isExpertAvailableOnSSD(self, expert_id, layer_index):
			file_exists = False
			file_name = self.generateExpertSSDfileName(expert_id, layer_index, 0)
			path_name = os.path.join(conceptExpertsPathName, file_name+pytorchTensorFileExtension)
			if os.path.exists(path_name):
				file_exists = True
			return file_exists
			
		def generateExpertSSDfileName(self, expert_id, layer_index, parameter_id):
			if(isinstance(expert_id, torch.Tensor)):
				expert_id = expert_id.item()
			file_name = "conceptColumnMLPexpertTensor_" + "parameter_id_" + str(parameter_id) + "_expert_id_" + str(expert_id) + "_layer_index_" + str(layer_index)
			return file_name
			
		def loadExpertFromSSD(self, expert_id_cpu, expert_id, layer_index):
			expert_weight_1 = self.loadTensor(conceptExpertsPathName, self.generateExpertSSDfileName(expert_id, layer_index, 0))
			expert_weight_2 = self.loadTensor(conceptExpertsPathName, self.generateExpertSSDfileName(expert_id, layer_index, 1))
			expert_bias_1 = self.loadTensor(conceptExpertsPathName, self.generateExpertSSDfileName(expert_id, layer_index, 2))
			expert_bias_2 = self.loadTensor(conceptExpertsPathName, self.generateExpertSSDfileName(expert_id, layer_index, 3))
			self.updateExpertCPU(expert_id_cpu, expert_weight_1, expert_weight_2, expert_bias_1, expert_bias_2)

		def saveExpertToSSD(self, expert_id_cpu, expert_id, layer_index):
			if(localConceptColumnExpertsStructure=="nnModuleList"):
				expert_weight_1 = self.experts[expert_id_cpu].expert_weight_1
				expert_weight_2 = self.experts[expert_id_cpu].expert_weight_2
				expert_bias_1 = self.experts[expert_id_cpu].expert_bias_1
				expert_bias_2 = self.experts[expert_id_cpu].expert_bias_2
			else:
				expert_weight_1 = self.experts_weight_1[expert_id_cpu]
				expert_weight_2 = self.experts_weight_2[expert_id_cpu]
				expert_bias_1 = self.experts_bias_1[expert_id_cpu]
				expert_bias_2 = self.experts_bias_2[expert_id_cpu]
			self.saveTensor(expert_weight_1, conceptExpertsPathName, self.generateExpertSSDfileName(expert_id, layer_index, 0))
			self.saveTensor(expert_weight_2, conceptExpertsPathName, self.generateExpertSSDfileName(expert_id, layer_index, 1))
			self.saveTensor(expert_bias_1, conceptExpertsPathName, self.generateExpertSSDfileName(expert_id, layer_index, 2))
			self.saveTensor(expert_bias_2, conceptExpertsPathName, self.generateExpertSSDfileName(expert_id, layer_index, 3))

		def initialiseExpertCPU(self, expert_id_cpu):
			expert_weight_1 = torch.empty(self.expert_intermediate_size, self.hidden_size)
			expert_weight_2 = torch.empty(self.hidden_size, self.expert_intermediate_size)
			expert_bias_1 = torch.empty(self.expert_intermediate_size)
			expert_bias_2 = torch.empty(self.hidden_size)

			# A simple initialization scheme (like xavier) for demonstration (sync with reset_parameters)
			nn.init.xavier_uniform_(expert_weight_1)
			nn.init.zeros_(expert_bias_1)
			nn.init.xavier_uniform_(expert_weight_2)
			nn.init.zeros_(expert_bias_2)
			
			self.updateExpertCPU(expert_id_cpu, expert_weight_1, expert_weight_2, expert_bias_1, expert_bias_2)
			
		def updateExpertCPU(self, expert_id_cpu, expert_weight_1, expert_weight_2, expert_bias_1, expert_bias_2):
			'''
			#probably redundant (ensure paramaters remain on cpu);
			self.expert_weight_1.data = self.expert_weight_1.data.cpu()
			self.expert_bias_1.data = self.expert_bias_1.data.cpu()
			self.expert_weight_2.data = self.expert_weight_2.data.cpu()
			self.expert_bias_2.data = self.expert_bias_2.data.cpu()
			'''
			with torch.no_grad():
				if(localConceptColumnExpertsStructure=="nnModuleList"):
					self.experts[expert_id_cpu].expert_weight_1.copy_(expert_weight_1)
					self.experts[expert_id_cpu].expert_weight_2.copy_(expert_weight_2)
					self.experts[expert_id_cpu].expert_bias_1.copy_(expert_bias_1)
					self.experts[expert_id_cpu].expert_bias_2.copy_(expert_bias_2)
				else:
					self.experts_weight_1[expert_id_cpu].copy_(expert_weight_1)
					self.experts_weight_2[expert_id_cpu].copy_(expert_weight_2)
					self.experts_bias_1[expert_id_cpu].copy_(expert_bias_1)
					self.experts_bias_2[expert_id_cpu].copy_(expert_bias_2)

		def saveTensor(self, tensor, folderName, fileName):
			pt.save(tensor.clone(), os.path.join(folderName, fileName+pytorchTensorFileExtension))

		def loadTensor(self, folderName, fileName):
			tensor = pt.load(os.path.join(folderName, fileName+pytorchTensorFileExtension))
			return tensor

		def getFirstKeyInDict(self, dictionary):
			return next(iter(dictionary.keys()), None)
			
		def getFirstValueInDict(self, dictionary):
			return next(iter(dictionary.values()), None)
