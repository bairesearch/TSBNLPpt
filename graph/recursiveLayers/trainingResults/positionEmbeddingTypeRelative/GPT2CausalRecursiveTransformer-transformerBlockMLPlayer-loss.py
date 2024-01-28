import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
y1 = np.array([4.0, 3.1273837089538574, 2.8044447898864746, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y2 = np.array([4.0, 2.5607528686523438, 2.177316188812256, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y3 = np.array([4.0, 2.791464328765869, 2.4578146934509277, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y6 = np.array([4.0, 2.9628775119781494, 2.6314992904663086, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y7 = np.array([4.0, 2.9707717895507812, 2.626065254211426, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y8 = np.array([4.0, 2.650296449661255, 2.3127589225769043, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y9 = np.array([4.0, 2.7048985958099365, 2.2871809005737305, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y10 = np.array([4.0, 2.7052271366119385, 2.2418935298919678, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y11 = np.array([4.0, 2.5743656158447266, 2.1565041542053223, 2.0030035972595215, 1.8743395805358887, 1.8238499164581299, 1.7853643894195557, 1.7099201679229736, np.nan ])
y12 = np.array([4.0, 2.6640706062316895, 2.3150103092193604, 2.174222230911255, 2.0626494884490967, 2.0101771354675293, 1.9626548290252686, 1.888489007949829, 1.8725013732910156])
y14 = np.array([4.0, 3.2518460750579834, 2.9827890396118164, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

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
plt.yticks(np.arange(0, 4.0+0.1, 0.5))

plt.xlabel("number of codeparrot-ds train samples (x1280000)")
plt.ylabel("Causal LM test loss")
plt.title("GPT2 Recursive Transformer")

plt.legend(handles=[l1, l2, l3, l6, l14, l7, l8, l9, l10, l11, l12], loc='lower left')

plt.show()
