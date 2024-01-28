import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3, 4])
y1 = np.array([1.0, 0.3768070258796215, 0.35672451157569884, 0.34805026364922526, 0.3422828152537346])
y2 = np.array([1.0, 0.30940675837397574, 0.28117460092902186, 0.2687879767358303, 0.2612325044542551])
y3 = np.array([1.0, 0.30296802766025066, 0.28217491447329524, 0.27392400919795035, 0.2656524788528681])
y6 = np.array([1.0, 0.3210689457276463, np.nan, np.nan, np.nan])
y7 = np.array([1.0, 0.3213745343607664, np.nan, np.nan, np.nan])
y8 = np.array([1.0, 0.3079980277708173, np.nan, np.nan, np.nan ])
y9 = np.array([1.0, 0.3090381079044938, np.nan, np.nan, np.nan ])
y14 = np.array([1.0, 0.8107746505975724, np.nan, np.nan, np.nan ])

l1, = plt.plot(x, y1, color='green', label='lay=1 (120MB)')
l2, = plt.plot(x, y2, color='blue', label='lay=6 (256MB)')
l3, = plt.plot(x, y3, color='red', label='lay=1r6 (120MB)')
l6, = plt.plot(x, y6, color='orange', label='lay=1r6 !MLPlayer (120MB)')
l7, = plt.plot(x, y7, color='yellow', label='lay=1r6 MLPlayerLast (120MB)')
l8, = plt.plot(x, y9, color='purple', label='lay=1r6 sharedLayerWeightsMLP (167MB)')
l9, = plt.plot(x, y8, color='navy', label='lay=1r6 sharedLayerWeightsMLP+SelfOut (156MB)')
l14, = plt.plot(x, y14, color='orange', label='lay=1 !MLPlayer (120MB)')

plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.yticks(np.arange(0, 0.5+0.1, 0.1))

plt.xlabel("number of oscar en train samples (x1000000)")
plt.ylabel("Masked LM test loss")
plt.title("RoBERTa Recursive Transformer")

plt.legend(handles=[l1, l2, l3, l6, l7, l8, l9, l14])

plt.show()
