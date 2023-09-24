import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3, 4])
y2 = np.array([0.5, 0.31453748081862926, 0.28465476580947635, 0.2705436096695065, 0.2619548720240593])
y3 = np.array([0.5, 0.3243083935147524, 0.2944739804047346, 0.28517395846247673, 0.27500040173232554])
y6 = np.array([0.5, np.nan, np.nan, np.nan, np.nan])
y7 = np.array([0.5, np.nan, np.nan, np.nan, np.nan])
y8 = np.array([0.5, 0.32535708522319795, np.nan, np.nan, np.nan ])
y9 = np.array([0.5, 0.32623474962949756, np.nan, np.nan, np.nan ])
y10 = np.array([0.5, 0.36937344963759183, 0.3171483307559043, np.nan, np.nan ])

l2, = plt.plot(x, y2, color='blue', label='lay=6 (256MB)')
l3, = plt.plot(x, y3, color='red', label='lay=1r6 (120MB)')
l6, = plt.plot(x, y6, color='orange', label='lay=1r6 !MLPlayer (?MB)')
l7, = plt.plot(x, y7, color='yellow', label='lay=1r6 MLPlayerLast (120MB)')
l8, = plt.plot(x, y8, color='purple', label='lay=1r6 sharedLayerWeightsMLP (166MB)')
l9, = plt.plot(x, y9, color='navy', label='lay=1r6 sharedLayerWeightsMLP+SelfOut (154MB)')
l10, = plt.plot(x, y10, color='darkviolet', label='lay=1r6 sharedLayerWeightsMLP (intermediate norm:256MB)')

plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.yticks(np.arange(0, 0.5+0.1, 0.1))

plt.xlabel("number of oscar en train samples (x1000000)")
plt.ylabel("Masked LM test loss")
plt.title("RoBERTa Recursive Transformer")

plt.legend(handles=[l2, l3, l8, l9, l10])

plt.show()
