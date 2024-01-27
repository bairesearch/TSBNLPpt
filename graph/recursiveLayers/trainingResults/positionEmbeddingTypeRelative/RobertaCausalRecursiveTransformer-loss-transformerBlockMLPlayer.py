import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3, 4])
y2 = np.array([6.0, 2.521359804382324, np.nan, np.nan, np.nan])
y3 = np.array([6.0, 2.5844951899254323, np.nan, np.nan, np.nan])
y6 = np.array([6.0, 4.073932212734222, np.nan, np.nan, np.nan])
y13 = np.array([6.0, 4.489333348503113, np.nan, np.nan, np.nan])

l2, = plt.plot(x, y2, color='blue', label='lay=6, hid=768, head=12 (256MB) [w/wo warmup]')
l3, = plt.plot(x, y3, color='red', label='lay=1r6, hid=768, head=12 (120MB)')
l6, = plt.plot(x, y6, color='blue', label='lay=6, hid=768, head=12 (256MB) [w warmup]')	#, linestyle='dashed'
l13, = plt.plot(x, y13, color='orange', label='lay=1r6, hid=1690, head=26; !MLPlayer (norm:255MB)')

plt.xticks(np.arange(min(x), max(x)+0.5, 0.5))
plt.yticks(np.arange(0, 6.0+0.1, 0.5))

plt.xlabel("number of oscar en train samples (x1000000)")
plt.ylabel("Causal LM test loss")
plt.title("RoBERTa Recursive Transformer")

plt.legend(handles=[l2, l3, l13])

plt.show()
