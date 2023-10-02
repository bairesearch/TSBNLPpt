import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3, 4])
y1 = np.array([0, 0.24316118257284164, np.nan, np.nan, np.nan])
y2 = np.array([0, 0.31380176396012305, np.nan, np.nan, np.nan])
y3 = np.array([0, 0.30387136477828025, np.nan, np.nan, np.nan])
y4 = np.array([0, np.nan, np.nan, np.nan, np.nan])
y5 = np.array([0, 0.2571000176203251, np.nan, np.nan, np.nan])

l1, = plt.plot(x, y1, color='green', label='lay=1, hid=768, head=12 (120MB)')
l2, = plt.plot(x, y2, color='blue', label='lay=6, hid=768, head=12 (256MB)')
l3, = plt.plot(x, y3, color='red', label='lay=1r6, hid=768, head=12 (120MB)')
l4, = plt.plot(x, y4, color='pink', label='lay=1r6, hid=1344, head=21 (norm:249MB)')
l5, = plt.plot(x, y5, color='lightgreen', label='lay=1, hid=1344, head=21 (norm:249MB)')

plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.yticks(np.arange(0, 0.5+0.1, 0.1))

plt.xlabel("number of oscar en train samples (x1000000)")
plt.ylabel("Causal LM test accuracy (Top-1)")
plt.title("RoBERTa Recursive Transformer")

plt.legend(handles=[l2, l3, l4, l1, l5])

plt.show()
