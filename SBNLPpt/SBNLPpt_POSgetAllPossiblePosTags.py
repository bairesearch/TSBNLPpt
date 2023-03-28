"""SBNLPpt_POSgetAllPossiblePosTags.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SBNLPpt_main.py

# Usage:
see SBNLPpt_main.py

# Description:
SBNLPpt POS Get All Possible Pos Tags

"""

import nltk
from nltk.corpus import brown
from collections import Counter, defaultdict

# wordPosTagsDict is a dict which will have the word as key and pos tags as values 
wordPosTagsDict = defaultdict(list)


def main():
	constructPOSdictionary()
	posValues = getAllPossiblePosTags('run')
	print(posValues)
	#['VB', 'NN', 'VBN', 'VBD']
	# 
	# VB: run the procedure
	# NN: the run
	# VBN: (passive) past participle - it was run
	# VBD: ??? - shouldnt this be ran? [https://stackoverflow.com/questions/51722351/get-all-possibles-pos-tags-from-a-single-word]

def constructPOSdictionary():
	for word, pos in brown.tagged_words():
		if pos not in wordPosTagsDict[word]:		# to append one tag only once
			wordPosTagsDict[word].append(pos)
				
def getAllPossiblePosTags(wordTest):
	posValues = wordPosTagsDict[wordTest]
	return posValues

if __name__ == '__main__':
	main()
