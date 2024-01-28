import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
y1 = np.array([0.0, 0.42544126510620117, 0.4689004719257355, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y2 = np.array([0.0, 0.5252946615219116, 0.5836884379386902, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y3 = np.array([0.0, 0.49481841921806335, 0.539782702922821, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y6 = np.array([0.0, 0.4629115164279938, 0.5114670395851135, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y7 = np.array([0.0, 0.4622270166873932, 0.5125551819801331, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y8 = np.array([0.0, 0.512558102607727, 0.5613349080085754, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y9 = np.array([0.0, 0.5035503506660461, 0.5652393102645874, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y10 = np.array([0.0, 0.5022847652435303, 0.5738951563835144, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y11 = np.array([0.0, 0.5243274569511414, 0.5877714157104492, 0.6125690937042236, 0.6370240449905396, 0.6460591554641724, 0.6511420607566833, 0.6658273935317993, np.nan])
y12 = np.array([0.0, 0.5122008323669434, 0.561950147151947, 0.5813300013542175, 0.6017950177192688, 0.6117182374000549, 0.6191267967224121, 0.6318800449371338, 0.6338686943054199])
y14 = np.array([0.0, 0.4097575545310974, 0.4444747269153595, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

l1, = plt.plot(x, y1, color='green', label='lay=1 (176MB)')
l2, = plt.plot(x, y2, color='blue', label='lay=12 (486MB)')
l3, = plt.plot(x, y3, color='red', label='lay=1r12 (176MB)')
l6, = plt.plot(x, y6, color='orange', label='lay=1r12 !MLPlayer (156MB)')
l7, = plt.plot(x, y7, color='yellow', label='lay=1r12 MLPlayerLast (176MB)')
l8, = plt.plot(x, y8, color='navy', label='lay=1r12 sharedLayerWeightsMLP (279MB)')
l9, = plt.plot(x, y9, color='purple', label='lay=1r12 sharedLayerWeightsMLP (norm:510MB)')
l10, = plt.plot(x, y10, color='darkviolet', label='lay=1r12 sharedLayerWeightsMLP (intermediate norm:477MB)')
l11, = plt.plot(x, y11, color='cyan', label='lay=18 (642MB)')
l12, = plt.plot(x, y12, color='thistle', label='lay=1r18 sharedLayerWeightsMLP (312MB)')
l14, = plt.plot(x, y14, color='darkorange', label='lay=1 !MLPlayer (156MB)')

plt.xticks(np.arange(min(x), max(x)+0.5, 0.5))
plt.yticks(np.arange(0, 0.7+0.1, 0.1))

plt.xlabel("number of codeparrot-ds train samples (x1280000)")
plt.ylabel("Causal LM test accuracy (Top-1)")
plt.title("GPT2 Recursive Transformer")

plt.legend(handles=[l1, l2, l3, l6, l14, l7, l8, l9, l10, l11, l12])

plt.show()
