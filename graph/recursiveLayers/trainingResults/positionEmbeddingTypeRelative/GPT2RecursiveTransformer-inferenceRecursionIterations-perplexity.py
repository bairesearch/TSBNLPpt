import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
y2 = np.array([25.0, 12.94555950164795, 8.822596549987793, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y3 = np.array([25.0, 16.30487823486328, 11.679261207580566, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y8 = np.array([np.nan, 15.2806396484375, 15.2806396484375, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y9 = np.array([np.nan, 11.984127044677734, 11.984127044677734, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y10 = np.array([np.nan, 13.081633567810059, 13.081633567810059, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

l2, = plt.plot(x, y2, color='blue', label='lay=12, hid=768, head=12 (486MB)')
l3, = plt.plot(x, y3, color='red', label='lay=1r12, hid=768, head=12 (176MB)')
l8, = plt.plot(x, y8, color='navy', label='lay=1r12, test:1r6, hid=768, head=12 (176MB)')
l9, = plt.plot(x, y9, color='purple', label='lay=1r12, test:1r18, hid=768, head=12 (176MB)')
l10, = plt.plot(x, y10, color='darkviolet', label='lay=1r12, test:1r24, hid=768, head=12 (176MB)')

plt.xticks(np.arange(min(x), max(x)+0.5, 0.5))
plt.yticks(np.arange(0, 25.0+2.0, 2.0))

plt.xlabel("number of codeparrot-ds train samples (x1280000)")
plt.ylabel("Causal LM test perplexity")
plt.title("GPT2 Recursive Transformer")

plt.legend(handles=[l2, l3, l8, l9, l10])

plt.show()
