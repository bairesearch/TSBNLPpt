import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3, 4])
y1 = np.array([0, 0.3200524007320404, np.nan, np.nan, np.nan])
y2 = np.array([0, 0.4284414293527603, np.nan, np.nan, np.nan])
y3 = np.array([0, 0.42890446846485136, np.nan, np.nan, np.nan])
y4 = np.array([0, 0.44220867459774016, np.nan, np.nan, np.nan])
y5 = np.array([0, 0.37814562149047853, np.nan, np.nan, np.nan])
y6 = np.array([0, 0.3787440519094467, np.nan, np.nan, np.nan])
y7 = np.array([0, 0.4473719531536102, np.nan, np.nan, np.nan])
y8 = np.array([0, 0.3374555701255798, np.nan, np.nan, np.nan])

l1, = plt.plot(x, y1, color='green', label='lay=1, hid=768, head=12 (120MB)')
l2, = plt.plot(x, y2, color='magenta', label='lay=1r3, hid=768, head=12 (120MB)')
l3, = plt.plot(x, y3, color='blue', label='lay=6, hid=768, head=12 (256MB)')
l4, = plt.plot(x, y4, color='red', label='lay=1r6, hid=768, head=12 (120MB)')
l5, = plt.plot(x, y5, color='navy', label='lay=1r6, hid=768, head=1 (123MB)')
l6, = plt.plot(x, y6, color='purple', label='lay=1r6, hid=1024, head=1 (169MB)')
l7, = plt.plot(x, y7, color='pink', label='lay=1r6, hid=1536, head=24 (norm:263MB)')
l8, = plt.plot(x, y8, color='lightgreen', label='lay=1, hid=1536, head=24 (norm:263MB)')

plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.yticks(np.arange(0, 0.5+0.1, 0.1))

plt.xlabel("number of oscar en train samples (x1000000)")
plt.ylabel("Masked LM test accuracy (Top-1)")
plt.title("RoBERTa Recursive Transformer")

plt.legend(handles=[l1, l2, l3, l4, l5, l6, l7, l8])

plt.show()
