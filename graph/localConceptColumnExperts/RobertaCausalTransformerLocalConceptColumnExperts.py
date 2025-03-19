import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 0.4, 1, 2, 3, 4])
y1 = np.array([0, 0.28784531995654106, np.nan, np.nan, np.nan, np.nan])
y2 = np.array([0, 0.2995950029402971, np.nan, np.nan, np.nan, np.nan])

l1, = plt.plot(x, y1, color='blue', label='lay=6, hid=768, head=12 (256MB)')
l2, = plt.plot(x, y2, color='magenta', label='lay=6 (w/ 8k concept experts), hid=768, head=12 (30GB)')

plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.minorticks_on()
plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.1))  # Minor ticks (every 0.1)
plt.yticks(np.arange(0, 0.5+0.1, 0.1))

plt.xlabel("number of oscar en train samples (x1000000)")
plt.ylabel("Causal LM test accuracy (Top-1)")
plt.title("RoBERTa Transformer with Local Concept Column Experts")

plt.legend(handles=[l1, l2])

plt.show()
