import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
y1 = np.array([4.0, 2.730393648147583, 2.502493143081665, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y2 = np.array([4.0, 2.2310760021209717, 1.9306120872497559, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y3 = np.array([4.0, 2.4861814975738525, 2.2334532737731934, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y4 = np.array([4.0, 2.264317274093628, 2.0197391510009766, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y5 = np.array([4.0, 2.928511619567871, 2.7008399963378906, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

'''
perplexity;
y1 = np.array([15.338923454284668, 12.21290397644043, np.nan, np.nan, np.nan])
y2 = np.array([9.309879302978516, 6.893728733062744, np.nan, np.nan, np.nan])
y3 = np.array([12.015307426452637, 9.332036972045898, np.nan, np.nan, np.nan])
y4 = np.array([9.624550819396973, 7.536358833312988, np.nan, np.nan, np.nan])
y5 = np.array([18.699777603149414, 14.89223575592041, np.nan, np.nan, np.nan])
'''

l1, = plt.plot(x, y1, color='green', label='lay=1, hid=768, head=12 (176MB)')
l2, = plt.plot(x, y2, color='blue', label='lay=12, hid=768, head=12 (486MB)')
l3, = plt.plot(x, y3, color='red', label='lay=1r12, hid=768, head=12 (176MB)')
l4, = plt.plot(x, y4, color='pink', label='lay=1r12, hid=1536, head=24 (norm:407MB)')
l5, = plt.plot(x, y5, color='lightgreen', label='lay=1, hid=768, head=12 (norm:407MB)')

plt.xticks(np.arange(min(x), max(x)+1, 0.5))
plt.yticks(np.arange(0, 5.0+0.1, 0.5))

plt.xlabel("number of codeparrot-ds train samples (x50000)")	#x50000*0.92=46000
plt.ylabel("Causal LM test loss")
plt.title("GPT2 Recursive Transformer")

plt.legend(handles=[l2, l3, l4, l1, l5])

plt.show()
