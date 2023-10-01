import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3, 4])
y1 = np.array([0.5, 0.3768070258796215, np.nan, np.nan, np.nan])
y2 = np.array([0.5, 0.3088168217420578, np.nan, np.nan, np.nan])
y3 = np.array([0.5, 0.30940675837397574, np.nan, np.nan, np.nan])
y4 = np.array([0.5, 0.30296802766025066, np.nan, np.nan, np.nan])
y5 = np.array([0.5, 0.3428756276369095, np.nan, np.nan, np.nan])
y6 = np.array([0.5, 0.34099548028707505, np.nan, np.nan, np.nan])
y7 = np.array([0.5, 0.2995814057946205, np.nan, np.nan, np.nan])
y8 = np.array([0.5, 0.361517172896862, np.nan, np.nan, np.nan])

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
plt.ylabel("Masked LM test loss")
plt.title("RoBERTa Recursive Transformer")

plt.legend(handles=[l1, l2, l3, l4, l5, l6, l7, l8])

plt.show()
