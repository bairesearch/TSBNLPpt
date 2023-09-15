import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3, 4])
y2 = np.array([0, 0.42890446846485136, 0.4656440287351608, 0.48165937175750734, 0.4940115956068039])
y3 = np.array([0, 0.44220867459774016, 0.4677618439912796, 0.4787390529155731, 0.49010481104850767])
y8 = np.array([0, 0.3820890604168129, np.nan, np.nan, np.nan])
y9 = np.array([0, 0.38374718098628685, np.nan, np.nan, np.nan])
y10 = np.array([0, 0.06796931902996546, np.nan, np.nan, np.nan])

l2, = plt.plot(x, y2, color='blue', label='lay=6, hid=768, head=12 (256MB)')
l3, = plt.plot(x, y3, color='red', label='lay=1r6, hid=768, head=12 (120MB)')
l8, = plt.plot(x, y8, color='navy', label='lay=1r6, test:1r3, hid=768, head=12 (120MB)')
l9, = plt.plot(x, y9, color='purple', label='lay=1r6, test:1r9, hid=768, head=12 (120MB)')
l10, = plt.plot(x, y10, color='darkviolet', label='lay=1r6, test:1r12, hid=768, head=12 (120MB)')

plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.yticks(np.arange(0, 0.5+0.1, 0.1))

plt.xlabel("number of oscar en train samples (x1000000)")
plt.ylabel("Masked LM test accuracy (Top-1)")
plt.title("RoBERTa Recursive Transformer")

plt.legend(handles=[l2, l3, l8, l9, l10])

plt.show()
