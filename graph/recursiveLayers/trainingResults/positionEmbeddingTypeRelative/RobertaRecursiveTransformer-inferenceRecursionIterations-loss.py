import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3, 4])
y2 = np.array([0.5, 0.30940675837397574, 0.28117460092902186, 0.2687879767358303, 0.2612325044542551])
y3 = np.array([0.5, 0.30296802766025066, 0.28217491447329524, 0.27392400919795035, 0.2656524788528681])
y8 = np.array([0.5, 0.4182064918688293, np.nan, np.nan, np.nan])
y9 = np.array([0.5, 0.38168357314213003, np.nan, np.nan, np.nan])
y10 = np.array([0.5, 5.019635282315201, np.nan, np.nan, np.nan])

l2, = plt.plot(x, y2, color='blue', label='lay=6, hid=768, head=12 (256MB)')
l3, = plt.plot(x, y3, color='red', label='lay=1r6, hid=768, head=12 (120MB)')
l8, = plt.plot(x, y8, color='navy', label='lay=1r6, test:1r3, hid=768, head=12 (120MB)')
l9, = plt.plot(x, y9, color='purple', label='lay=1r6, test:1r9, hid=768, head=12 (120MB)')
l10, = plt.plot(x, y10, color='darkviolet', label='lay=1r6, test:1r12, hid=768, head=12 (120MB)')

plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.yticks(np.arange(0, 0.5+0.1, 0.1))

plt.xlabel("number of oscar en train samples (x1000000)")
plt.ylabel("Masked LM test loss")
plt.title("RoBERTa Recursive Transformer")

plt.legend(handles=[l2, l3, l8, l9, l10])

plt.show()
