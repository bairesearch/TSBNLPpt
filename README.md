# SBNLPpt

### Author

Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

### Description

Syntactic Bias natural language processing (SBNLP) for PyTorch - neural architectures with various syntactic inductive biases (recursiveLayers, simulatedDendriticBranches, memoryTraceBias, semanticRelationVectorSpaces, tokenMemoryBank, transformerAttentionHeadPermutations, transformerPOSembeddings) - experimental

### License

MIT License

### Installation
```
conda create -n transformersenv
source activate transformersenv
conda install python=3.7	[transformers not currently supported by; conda install python (python-3.10.6)]
pip install datasets
pip install transfomers==4.23.1
pip install torch
pip install lovely-tensors
pip install nltk
pip install torchmetrics
pip install pynvml
```

### Execution
```
source activate transformersenv
python SBNLPpt_main.py
```
