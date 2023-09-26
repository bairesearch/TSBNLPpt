import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3, 4])
y1 = np.array([0.5, 0.3768070258796215, 0.35672451157569884, 0.34805026364922526, 0.3422828152537346])
y2 = np.array([0.5, 0.30940675837397574, 0.28117460092902186, 0.2687879767358303, 0.2612325044542551])
y3 = np.array([0.5, 0.30296802766025066, 0.28217491447329524, 0.27392400919795035, 0.2656524788528681])
y4 = np.array([0.5, 0.2995814057946205, 0.27438519304990766, 0.2595572123140097, 0.2537853354871273])
y5 = np.array([0.5, 0.36026279502511027, 0.34200011098980904, 0.3328074170410633, 0.32603148027658463])
y6 = np.array([0.5, 0.2982051469832659, np.nan, np.nan, np.nan])
y7 = np.array([0.5, 0.3598037402355671, np.nan, np.nan, np.nan])

l1, = plt.plot(x, y1, color='green', label='lay=1, hid=768, head=12 (120MB)')
l2, = plt.plot(x, y2, color='blue', label='lay=6, hid=768, head=12 (256MB)')
l3, = plt.plot(x, y3, color='red', label='lay=1r6, hid=768, head=12 (120MB)')
l4, = plt.plot(x, y4, color='pink', label='lay=1r6, hid=1536, head=24 (non-intermediate norm:263MB)', linestyle='dashed')
l5, = plt.plot(x, y5, color='lightgreen', label='lay=1, hid=1536, head=24 (non-intermediate norm:263MB)', linestyle='dashed')
l6, = plt.plot(x, y6, color='pink', label='lay=1r6, hid=1344, head=21 (norm:249MB)')
l7, = plt.plot(x, y7, color='lightgreen', label='lay=1, hid=1344, head=21 (norm:249MB)')

plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.yticks(np.arange(0, 0.5+0.1, 0.1))

plt.xlabel("number of oscar en train samples (x1000000)")
plt.ylabel("Masked LM test loss")
plt.title("RoBERTa Recursive Transformer")

plt.legend(handles=[l2, l3, l4, l6, l1, l5, l7])

plt.show()
