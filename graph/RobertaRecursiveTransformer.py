import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3, 4])
y1 = np.array([0, 0.3200524007320404, np.nan, np.nan, np.nan])
y2 = np.array([0, 0.42890446846485136, np.nan, np.nan, np.nan])
y3 = np.array([0, 0.44220867459774016, np.nan, np.nan, np.nan])
y4 = np.array([0, 0.4473719531536102, np.nan, np.nan, np.nan])

l1, = plt.plot(x, y1, color='green', label='lay=1, hid=768, head=12 (120MB)')
l2, = plt.plot(x, y2, color='blue', label='lay=6, hid=768, head=12 (256MB)')
l3, = plt.plot(x, y3, color='red', label='lay=1r6, hid=768, head=12 (120MB)')
l4, = plt.plot(x, y4, color='turquoise', label='lay=1r6, hid=1536, head=24 (norm:263MB)')

plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.yticks(np.arange(0, 0.5+0.1, 0.1))

plt.xlabel("number of oscar en train samples (x1000000)")
plt.ylabel("Masked LM test accuracy (Top-1)")
plt.title("RoBERTa Recursive Transformer")

plt.legend(handles=[l1, l2, l3, l4])

plt.show()
