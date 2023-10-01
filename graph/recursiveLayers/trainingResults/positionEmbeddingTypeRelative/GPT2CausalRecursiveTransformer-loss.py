import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
y1 = np.array([4.0, 3.1273837089538574, 2.8044447898864746, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y2 = np.array([4.0, 2.5607528686523438, 2.177316188812256, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y3 = np.array([4.0, 2.791464328765869, 2.4578146934509277, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y4 = np.array([4.0, 2.698993682861328, 2.3023264408111572, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y5 = np.array([4.0, 2.940739393234253, 2.563929796218872, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

l1, = plt.plot(x, y1, color='green', label='lay=1, hid=768, head=12 (176MB)')
l2, = plt.plot(x, y2, color='blue', label='lay=12, hid=768, head=12 (486MB)')
l3, = plt.plot(x, y3, color='red', label='lay=1r12, hid=768, head=12 (176MB)')
l4, = plt.plot(x, y4, color='pink', label='lay=1r12, hid=1792, head=28 (norm:496MB)')
l5, = plt.plot(x, y5, color='lightgreen', label='lay=1, hid=768, head=24 (norm:407MB)')

plt.xticks(np.arange(min(x), max(x)+0.5, 0.5))
plt.yticks(np.arange(0, 4.0+0.1, 0.5))

plt.xlabel("number of codeparrot-ds train samples (x1280000)")
plt.ylabel("Causal LM test loss")
plt.title("GPT2 Recursive Transformer")

plt.legend(handles=[l2, l3, l4, l1, l5])

plt.show()
