import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3, 4])
y1 = np.array([0.5, 0.33334393852204086, 0.2938384489557147, 0.2779732163208723, 0.26914998671591284])
y2 = np.array([0.5, 0.31453748081862926, 0.28465476580947635, 0.2705436096695065, 0.2619548720240593])
y3 = np.array([0.5, 0.3725783558964729, 0.3484503712660074, 0.3369026028829813, 0.3298485902744532])
y4 = np.array([0.5, 0.3243083935147524, 0.2944739804047346, 0.28517395846247673, 0.27500040173232554])
y5 = np.array([0.5, 0.3887588016808033, 0.36320978958070277, 0.3523632300710678, 0.3456409868645668])

l1, = plt.plot(x, y1, color='pink', label='lay=1r6 sharedLayerW, hid=1344, head=21 (norm:249MB)')
l2, = plt.plot(x, y2, color='blue', label='lay=6, hid=768, head=12 (256MB)')
l3, = plt.plot(x, y3, color='lightgreen', label='lay=1, hid=1344, head=21 (norm:249MB)')
l4, = plt.plot(x, y4, color='red', label='lay=1r6, hid=768, head=12 (120MB)')
l5, = plt.plot(x, y5, color='green', label='lay=1, hid=768, head=12 (120MB)')

plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.yticks(np.arange(0, 0.5+0.1, 0.1))

plt.xlabel("number of oscar en train samples (x1000000)")
plt.ylabel("Masked LM test loss")
plt.title("RoBERTa Recursive Transformer")

plt.legend(handles=[l2, l4, l1, l5, l3])

plt.show()
