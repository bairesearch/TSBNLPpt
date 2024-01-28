import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
y1 = np.array([0.0, 0.4715276062488556, 0.504429817199707, 0.5217463970184326, 0.5327208638191223, 0.540026068687439, 0.5456399321556091, 0.5492268204689026, 0.5532917976379395])
y2 = np.array([0.0, 0.5848015546798706, 0.6353920698165894, 0.6581273078918457, 0.6719351410865784, 0.6809597015380859, 0.6881480813026428, 0.6926000118255615, 0.6963947415351868])
y3 = np.array([0.0, 0.5370364785194397, 0.5751139521598816, 0.5930986404418945, 0.6051175594329834, 0.6125524044036865, 0.6182177066802979, 0.6243066191673279, 0.6271749138832092])
y6 = np.array([0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y7 = np.array([0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y8 = np.array([0.0, 0.5002896189689636, 0.5535823702812195, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y13 = np.array([0.0, 0.49833741784095764, 0.5456426739692688, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y14 = np.array([0.0, 0.4050893783569336, 0.4393422603607178, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

l1, = plt.plot(x, y1, color='green', label='lay=1 (176MB)')
l2, = plt.plot(x, y2, color='blue', label='lay=12 (486MB)')
l3, = plt.plot(x, y3, color='red', label='lay=1r12 (176MB)')
l6, = plt.plot(x, y6, color='orange', label='lay=1r12 !MLPlayer (156MB)')
l7, = plt.plot(x, y7, color='yellow', label='lay=1r12 MLPlayerLast (176MB)')
l8, = plt.plot(x, y8, color='navy', label='lay=1r12 sharedLayerWeightsMLP (276MB)')
l13, = plt.plot(x, y13, color='orange', label='lay=1r12 !MLPlayer (norm:463MB)')
l14, = plt.plot(x, y14, color='darkorange', label='lay=1 !MLPlayer (158MB)')

plt.xticks(np.arange(min(x), max(x)+0.5, 0.5))
plt.yticks(np.arange(0, 0.7+0.1, 0.1))

plt.xlabel("number of codeparrot-ds train samples (x1280000)")
plt.ylabel("Causal LM test accuracy (Top-1)")
plt.title("GPT2 Recursive Transformer")

plt.legend(handles=[l1, l2, l3, l8, l13, l14])

plt.show()
