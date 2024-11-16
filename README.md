# TSBNLPpt

### Author

Richard Bruce Baxter - Copyright (c) 2022-2024 Baxter AI (baxterai.com)

### Description

Transformer Syntactic Bias natural language processing (TSBNLP) for PyTorch - transformer with various syntactic inductive biases (recursiveLayers, tokenMemoryBank, memoryTraceBias, semanticRelationVectorSpaces/transformerPOSembeddings, transformerAttentionHeadPermutations, simulatedDendriticBranches, prosodyDelimitedData, attendToLocalConceptColumns) - experimental

### License

MIT License

### Installation
```
conda create -n pytorchsenv
source activate pytorchsenv
conda install python=3.12
pip install networkx
pip install matplotlib
pip install yattag
pip install torch
pip install torch_geometric
pip install nltk 
pip install spacy
pip install benepar
pip install datasets
pip install transfomers
pip install lovely-tensors
pip install torchmetrics
pip install pynvml
python3 -m spacy download en_core_web_md
```

### Execution
```
source activate pytorchsenv
python TSBNLPpt_main.py
```

## recursiveLayers

![recursiveLayers1a.drawio.png](https://github.com/bairesearch/TSBpt/blob/master/graph/recursiveLayers/implementations/recursiveLayersOrigImplementation/recursiveLayers1a.drawio.png?raw=true)
- [recursiveLayers1b.drawio.png](https://github.com/bairesearch/TSBpt/blob/master/graph/recursiveLayers/implementations/recursiveLayersNewImplementation/recursiveLayers1b.drawio.png?raw=true)

### ROBERTApt (Masked LM)

#### positionEmbeddingType relative_key

![RobertaMaskedRecursiveTransformer.png](https://github.com/bairesearch/TSBpt/blob/master/graph/recursiveLayers/trainingResults/positionEmbeddingTypeRelative/RobertaMaskedRecursiveTransformer.png?raw=true)
- [RobertaMaskedRecursiveTransformer-loss.png](https://github.com/bairesearch/TSBpt/blob/master/graph/recursiveLayers/trainingResults/positionEmbeddingTypeRelative/RobertaMaskedRecursiveTransformer-loss.png?raw=true)
- [RobertaMaskedRecursiveTransformerHeads.png](https://github.com/bairesearch/TSBpt/blob/master/graph/recursiveLayers/trainingResults/positionEmbeddingTypeRelative/RobertaMaskedRecursiveTransformerHeads.png?raw=true)

#### positionEmbeddingType absolute

![RobertaMaskedRecursiveTransformer.png](https://github.com/bairesearch/TSBpt/blob/master/graph/recursiveLayers/trainingResults/positionEmbeddingTypeAbsolute/RobertaMaskedRecursiveTransformer.png?raw=true)
- [RobertaMaskedRecursiveTransformer-loss.png](https://github.com/bairesearch/TSBpt/blob/master/graph/recursiveLayers/trainingResults/positionEmbeddingTypeAbsolute/RobertaMaskedRecursiveTransformer-loss.png?raw=true)
- [RobertaMaskedRecursiveTransformerLayers.png](https://github.com/bairesearch/TSBpt/blob/master/graph/recursiveLayers/trainingResults/positionEmbeddingTypeAbsolute/RobertaMaskedRecursiveTransformerLayers.png?raw=true)

### ROBERTApt (Causal LM)

#### positionEmbeddingType relative_key

![RobertaCausalRecursiveTransformer.png](https://github.com/bairesearch/TSBpt/blob/master/graph/recursiveLayers/trainingResults/positionEmbeddingTypeRelative/RobertaCausalRecursiveTransformer.png?raw=true)
- [RobertaCausalRecursiveTransformer-loss.png](https://github.com/bairesearch/TSBpt/blob/master/graph/recursiveLayers/trainingResults/positionEmbeddingTypeRelative/RobertaCausalRecursiveTransformer-loss.png?raw=true)


### GPT2pt (Causal LM)

#### positionEmbeddingType relative_key

![GPT2CausalRecursiveTransformer.png](https://github.com/bairesearch/TSBpt/blob/master/graph/recursiveLayers/trainingResults/positionEmbeddingTypeRelative/GPT2CausalRecursiveTransformer.png?raw=true)
- [GPT2CausalRecursiveTransformer-loss.png](https://github.com/bairesearch/TSBpt/blob/master/graph/recursiveLayers/trainingResults/positionEmbeddingTypeRelative/GPT2CausalRecursiveTransformer-loss.png?raw=true)

#### positionEmbeddingType absolute

![GPT2CausalRecursiveTransformer.png](https://github.com/bairesearch/TSBpt/blob/master/graph/recursiveLayers/trainingResults/positionEmbeddingTypeAbsolute/GPT2CausalRecursiveTransformer.png?raw=true)
- [GPT2CausalRecursiveTransformer-loss.png](https://github.com/bairesearch/TSBpt/blob/master/graph/recursiveLayers/trainingResults/positionEmbeddingTypeAbsolute/GPT2CausalRecursiveTransformer-loss.png?raw=true)

## transformerSuperblocks

#### implementations/recursiveLayersNewImplementation
![recursiveLayers1b.drawio.png](https://github.com/bairesearch/TSBpt/blob/master/graph/recursiveLayers/implementations/recursiveLayersNewImplementation/recursiveLayers1b.drawio.png?raw=true)
![transformerSuperblocks1b.drawio.png](https://github.com/bairesearch/TSBpt/blob/master/graph/recursiveLayers/implementations/recursiveLayersNewImplementation/transformerSuperblocks1b.drawio.png?raw=true)

#### implementations/recursiveLayersOrigImplementation
- [recursiveLayers1a.drawio.png](https://github.com/bairesearch/TSBpt/blob/master/graph/recursiveLayers/implementations/recursiveLayersOrigImplementation/recursiveLayers1a.drawio.png?raw=true)
- [transformerSuperblocks1a.drawio.png](https://github.com/bairesearch/TSBpt/blob/master/graph/recursiveLayers/implementations/recursiveLayersOrigImplementation/transformerSuperblocks1a.drawio.png?raw=true)

## tokenMemoryBank

![tokenMemoryBank1a.drawio.png](https://github.com/bairesearch/TSBpt/blob/master/graph/tokenMemoryBank/tokenMemoryBank1a.drawio.png?raw=true)


