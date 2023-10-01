import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3, 4])
y2 = np.array([0, 0.42890446846485136, 0.4656440287351608, 0.48165937175750734, 0.4940115956068039])
y3 = np.array([0, 0.44220867459774016, 0.4677618439912796, 0.4787390529155731, 0.49010481104850767])
y6 = np.array([0, 0.41028395064234735, np.nan, np.nan, np.nan])
y7 = np.array([0, 0.40962072494506835, np.nan, np.nan, np.nan])
y8 = np.array([0.0, 0.4304612619328499, np.nan, np.nan, np.nan ])
y9 = np.array([0.0, 0.42894357948422435, np.nan, np.nan, np.nan ])

l2, = plt.plot(x, y2, color='blue', label='lay=6 (256MB)')
l3, = plt.plot(x, y3, color='red', label='lay=1r6 (120MB)')
l6, = plt.plot(x, y6, color='orange', label='lay=1r6 !MLPlayer (120MB)')
l7, = plt.plot(x, y7, color='yellow', label='lay=1r6 MLPlayerLast (120MB)')
l8, = plt.plot(x, y9, color='purple', label='lay=1r6 sharedLayerWeightsMLP (167MB)')
l9, = plt.plot(x, y8, color='navy', label='lay=1r6 sharedLayerWeightsMLP+SelfOut (156MB)')

plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.yticks(np.arange(0, 0.5+0.1, 0.1))

plt.xlabel("number of oscar en train samples (x1000000)")
plt.ylabel("Masked LM test accuracy (Top-1)")
plt.title("RoBERTa Recursive Transformer")

plt.legend(handles=[l2, l3, l6, l7, l8, l9])

plt.show()
