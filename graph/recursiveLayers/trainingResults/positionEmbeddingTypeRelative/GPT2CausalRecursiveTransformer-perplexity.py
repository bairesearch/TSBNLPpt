import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
y1 = np.array([25.0, 22.8142147064209, 16.517902374267578, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y2 = np.array([25.0, 12.94555950164795, 8.822596549987793, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y3 = np.array([25.0, 16.30487823486328, 11.679261207580566, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y4 = np.array([25.0, 14.864766120910645, 9.997413635253906, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y5 = np.array([25.0, 18.929838180541992, 12.9867525100708, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

l1, = plt.plot(x, y1, color='green', label='lay=1, hid=768, head=12 (176MB)')
l2, = plt.plot(x, y2, color='blue', label='lay=12, hid=768, head=12 (486MB)')
l3, = plt.plot(x, y3, color='red', label='lay=1r12, hid=768, head=12 (176MB)')
l4, = plt.plot(x, y4, color='pink', label='lay=1r12, hid=1792, head=28 (norm:496MB)')
l5, = plt.plot(x, y5, color='lightgreen', label='lay=1, hid=768, head=24 (norm:407MB)')

plt.xticks(np.arange(min(x), max(x)+0.5, 0.5))
plt.yticks(np.arange(0, 25.0+2.0, 2.0))

plt.xlabel("number of codeparrot-ds train samples (x1280000)")
plt.ylabel("Causal LM test perplexity")
plt.title("GPT2 Recursive Transformer")

plt.legend(handles=[l2, l3, l4, l1, l5])

plt.show()
