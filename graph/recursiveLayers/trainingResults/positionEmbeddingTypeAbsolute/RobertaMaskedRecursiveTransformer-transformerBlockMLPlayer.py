import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3, 4])
y2 = np.array([0, 0.4195622387599945, 0.45833434972524645, 0.477471009747982, 0.4880324535870552])
y3 = np.array([0, 0.4107974250018597, 0.44753125154256823, 0.46352191767930984, 0.4760648959302902])
y6 = np.array([0, np.nan, np.nan, np.nan, np.nan])
y7 = np.array([0, np.nan, np.nan, np.nan, np.nan])
y8 = np.array([0.0, 0.4075560009920597, np.nan, np.nan, np.nan ])
y9 = np.array([0.0, 0.40710742794513705, np.nan, np.nan, np.nan ])
y10 = np.array([0.0, 0.3540379656097293, 0.41716802026569844, np.nan, np.nan ])

l2, = plt.plot(x, y2, color='blue', label='lay=6 (256MB)')
l3, = plt.plot(x, y3, color='red', label='lay=1r6 (120MB)')
l6, = plt.plot(x, y6, color='orange', label='lay=1r6 !MLPlayer (?MB)')
l7, = plt.plot(x, y7, color='yellow', label='lay=1r6 MLPlayerLast (120MB)')
l8, = plt.plot(x, y8, color='purple', label='lay=1r6 sharedLayerWeightsMLP (166MB)')
l9, = plt.plot(x, y9, color='navy', label='lay=1r6 sharedLayerWeightsMLP+SelfOut (156MB)')
l10, = plt.plot(x, y10, color='darkviolet', label='lay=1r6 sharedLayerWeightsMLP (intermediate norm:256MB)')

plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.yticks(np.arange(0, 0.5+0.1, 0.1))

plt.xlabel("number of oscar en train samples (x1000000)")
plt.ylabel("Masked LM test accuracy (Top-1)")
plt.title("RoBERTa Recursive Transformer")

plt.legend(handles=[l2, l3, l8, l9, l10], loc='lower left')

plt.show()
