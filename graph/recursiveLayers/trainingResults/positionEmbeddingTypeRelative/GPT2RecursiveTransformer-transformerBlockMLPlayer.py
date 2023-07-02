import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
y1 = np.array([0.0, 0.42544126510620117, 0.4689004719257355, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y2 = np.array([0.0, 0.5252946615219116, 0.5836884379386902, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y3 = np.array([0.0, 0.49481841921806335, 0.539782702922821, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y4 = np.array([0.0, 0.5054967999458313, 0.5646486878395081, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y5 = np.array([0.0, 0.4526788890361786, 0.5098543167114258, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y6 = np.array([0.0, 0.4629115164279938, 0.5114670395851135, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y7 = np.array([0.0, 0.4622270166873932, 0.5125551819801331, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y8 = np.array([0.0, np.nan, 0.4983603060245514, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y9 = np.array([0.0, np.nan, 0.5337565541267395, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y10 = np.array([0.0, np.nan, 0.5188811421394348, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

l1, = plt.plot(x, y1, color='green', label='lay=1, hid=768, head=12 (176MB)')
l2, = plt.plot(x, y2, color='blue', label='lay=12, hid=768, head=12 (486MB)')
l3, = plt.plot(x, y3, color='red', label='lay=1r12, hid=768, head=12 (176MB)')
l4, = plt.plot(x, y4, color='pink', label='lay=1r12, hid=1792, head=28 (norm:496MB)')
l5, = plt.plot(x, y5, color='lightgreen', label='lay=1, hid=768, head=24 (norm:407MB)')
l6, = plt.plot(x, y6, color='orange', label='lay=1r12 !MLPlayer, hid=768, head=12 (156MB)')
l7, = plt.plot(x, y7, color='yellow', label='lay=1r12 MLPlayerLast, hid=768, head=12 (176MB)')
l8, = plt.plot(x, y8, color='magenta', label='lay=1r12, test:1r6, hid=768, head=12 (176MB)')
l9, = plt.plot(x, y9, color='magenta', label='lay=1r12, test:1r18, hid=768, head=12 (176MB)')
l10, = plt.plot(x, y10, color='magenta', label='lay=1r12, test:1r24, hid=768, head=12 (176MB)')

plt.xticks(np.arange(min(x), max(x)+0.5, 0.5))
plt.yticks(np.arange(0, 0.7+0.1, 0.1))

plt.xlabel("number of codeparrot-ds train samples (x1280000)")
plt.ylabel("Causal LM test accuracy (Top-1)")
plt.title("GPT2 Recursive Transformer")

plt.legend(handles=[l2, l3, l4, l1, l5, l6, l7, l8])

plt.show()
