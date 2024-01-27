import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
y2 = np.array([4.0, 2.135882616043091, 1.8487520217895508, 1.7244564294815063, 1.6493819952011108, 1.6002614498138428, 1.5652133226394653, 1.5380713939666748, 1.520236611366272])
y3 = np.array([4.0, 2.434828519821167, 2.187492847442627, 2.080907106399536, 2.009803533554077, 1.9629100561141968, 1.9358490705490112, 1.9002397060394287, 1.8871656656265259])
y6 = np.array([4.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y7 = np.array([4.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y8 = np.array([4.0, 2.728748321533203, 2.3622779846191406, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y13 = np.array([4.0, 2.7425777912139893, 2.425227642059326, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

l2, = plt.plot(x, y2, color='blue', label='lay=12 (486MB)')
l3, = plt.plot(x, y3, color='red', label='lay=1r12 (176MB)')
l6, = plt.plot(x, y6, color='orange', label='lay=1r12 !MLPlayer (156MB)')
l7, = plt.plot(x, y7, color='yellow', label='lay=1r12 MLPlayerLast (176MB)')
l8, = plt.plot(x, y8, color='navy', label='lay=1r12 sharedLayerWeightsMLP (276MB)')
l13, = plt.plot(x, y13, color='orange', label='lay=1r12 !MLPlayer (norm:463MB)')

plt.xticks(np.arange(min(x), max(x)+0.5, 0.5))
plt.yticks(np.arange(0, 4.0+0.1, 0.5))

plt.xlabel("number of codeparrot-ds train samples (x1280000)")
plt.ylabel("Causal LM test loss")
plt.title("GPT2 Recursive Transformer")

plt.legend(handles=[l2, l3, l8, l13])

plt.show()
