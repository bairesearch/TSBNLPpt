# SBNLPpt

### Author

Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

### Description

Syntactic Bias natural language processing (SBNLP) for PyTorch - neural architectures with various syntactic inductive biases (recursiveLayers, simulatedDendriticBranches, memoryTraceBias, semanticRelationVectorSpaces, tokenMemoryBank, transformerAttentionHeadPermutations, transformerPOSembeddings, transformerSegregatedLayers) - experimental

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

## recursiveLayers

![recursiveLayers1a.drawio.png](https://github.com/bairesearch/TSBpt/blob/master/graph/recursiveLayers1a.drawio.png?raw=true)

![RobertaRecursiveTransformer.png](https://github.com/bairesearch/TSBpt/blob/master/graph/RobertaRecursiveTransformer.png?raw=true)

![RobertaRecursiveTransformerLayers.png](https://github.com/bairesearch/TSBpt/blob/master/graph/RobertaRecursiveTransformerLayers.png?raw=true)

## transformerSegregatedLayers

![transformerSegregatedLayers1a.drawio.png](https://github.com/bairesearch/TSBpt/blob/master/graph/transformerSegregatedLayers1a.drawio.png?raw=true)

![RobertaRecursiveTransformerSegregatedLayersCodebase2022.png](https://github.com/bairesearch/TSBpt/blob/master/graph/RobertaRecursiveTransformerSegregatedLayersCodebase2022.png?raw=true)

![RobertaRecursiveTransformerSegregatedLayers.png](https://github.com/bairesearch/TSBpt/blob/master/graph/RobertaRecursiveTransformerSegregatedLayers.png?raw=true)

## tokenMemoryBank

![tokenMemoryBank1a.drawio.png](https://github.com/bairesearch/TSBpt/blob/master/graph/tokenMemoryBank1a.drawio.png?raw=true)


