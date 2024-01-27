import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3, 4])
y2 = np.array([0, 0.31380176396012305, np.nan, np.nan, np.nan])
y3 = np.array([0, 0.30387136477828025, np.nan, np.nan, np.nan])
y6 = np.array([0, 0.310142416574955, np.nan, np.nan, np.nan])
y13 = np.array([0, 0.275956364222765, np.nan, np.nan, np.nan])

l2, = plt.plot(x, y2, color='blue', label='lay=6, hid=768, head=12 (256MB) [w/wo warmup]')
l3, = plt.plot(x, y3, color='red', label='lay=1r6, hid=768, head=12 (120MB)')
#l6, = plt.plot(x, y6, color='blue', label='lay=6, hid=768, head=12 (256MB) [w/warmup]', linestyle='dashed')
l13, = plt.plot(x, y13, color='orange', label='lay=1r6, hid=1690, head=26; !MLPlayer (norm:255MB)')

plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.yticks(np.arange(0, 0.5+0.1, 0.1))

plt.xlabel("number of oscar en train samples (x1000000)")
plt.ylabel("Causal LM test accuracy (Top-1)")
plt.title("RoBERTa Recursive Transformer")

plt.legend(handles=[l2, l3, l13])

plt.show()
