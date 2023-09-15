import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
y2 = np.array([0.0, 0.5252946615219116, 0.5836884379386902, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y3 = np.array([0.0, 0.49481841921806335, 0.539782702922821, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y8 = np.array([np.nan, 0.4983603060245514, 0.4983603060245514, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y9 = np.array([np.nan, 0.5337565541267395, 0.5337565541267395, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y10 = np.array([np.nan, 0.5188811421394348, 0.5188811421394348, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

l2, = plt.plot(x, y2, color='blue', label='lay=12, hid=768, head=12 (486MB)')
l3, = plt.plot(x, y3, color='red', label='lay=1r12, hid=768, head=12 (176MB)')
l8, = plt.plot(x, y8, color='navy', label='lay=1r12, test:1r6, hid=768, head=12 (176MB)')
l9, = plt.plot(x, y9, color='purple', label='lay=1r12, test:1r18, hid=768, head=12 (176MB)')
l10, = plt.plot(x, y10, color='darkviolet', label='lay=1r12, test:1r24, hid=768, head=12 (176MB)')

plt.xticks(np.arange(min(x), max(x)+0.5, 0.5))
plt.yticks(np.arange(0, 0.7+0.1, 0.1))

plt.xlabel("number of codeparrot-ds train samples (x1280000)")
plt.ylabel("Causal LM test accuracy (Top-1)")
plt.title("GPT2 Recursive Transformer")

plt.legend(handles=[l2, l3, l8, l9, l10])

plt.show()
