import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
y1 = np.array([4.0, 3.1273837089538574, 2.8044447898864746, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y2 = np.array([4.0, 2.5607528686523438, 2.177316188812256, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y3 = np.array([4.0, 2.791464328765869, 2.4578146934509277, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y4 = np.array([4.0, 2.698993682861328, 2.3023264408111572, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y5 = np.array([4.0, 2.940739393234253, 2.563929796218872, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y6 = np.array([4.0, 2.9628775119781494, 2.6314992904663086, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y7 = np.array([4.0, 2.9707717895507812, 2.626065254211426, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y8 = np.array([4.0, np.nan, 2.7265865802764893, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y9 = np.array([4.0, np.nan, 2.4835829734802246, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y10 = np.array([4.0, np.nan, 2.571209192276001, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

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
plt.yticks(np.arange(0, 5.0+0.1, 0.5))

plt.xlabel("number of codeparrot-ds train samples (x1280000)")
plt.ylabel("Causal LM test loss")
plt.title("GPT2 Recursive Transformer")

plt.legend(handles=[l2, l3, l4, l1, l5, l6, l7, l8])

plt.show()
