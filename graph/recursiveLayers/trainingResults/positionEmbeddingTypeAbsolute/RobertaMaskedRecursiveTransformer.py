import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3, 4])
y1 = np.array([0, 0.4007114856934547, 0.4485383033657074, 0.4697808350682259, 0.4805689078116417])
y2 = np.array([0, 0.4195622387599945, 0.45833434972524645, 0.477471009747982, 0.4880324535870552])
y3 = np.array([0, 0.32205438960790633, 0.34977506301760675, 0.3640139799118042, 0.3723136756336689])
y4 = np.array([0, 0.4107974250018597, 0.44753125154256823, 0.46352191767930984, 0.4760648959302902])
y5 = np.array([0, 0.30340382710933683, 0.3310104872369766, 0.34348129745364187, 0.35091807383656504])

l1, = plt.plot(x, y1, color='pink', label='lay=1r6 sharedLayerW, hid=1344, head=21 (norm:249MB)')
l2, = plt.plot(x, y2, color='blue', label='lay=6, hid=768, head=12 (256MB)')
l3, = plt.plot(x, y3, color='lightgreen', label='lay=1, hid=1344, head=21 (norm:249MB)')
l4, = plt.plot(x, y4, color='red', label='lay=1r6, hid=768, head=12 (120MB)')
l5, = plt.plot(x, y5, color='green', label='lay=1, hid=768, head=12 (120MB)')


plt.xticks(np.arange(min(x), max(x)+1, 1.0))

plt.xlabel("number of oscar en train samples (x1000000)")
plt.ylabel("Masked LM test accuracy (Top-1)")
plt.title("RoBERTa Recursive Transformer")

plt.legend(handles=[l2, l4, l1, l5, l3], loc='lower left')

plt.show()
