import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
y2 = np.array([25.0, 12.94555950164795, 8.822596549987793, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y3 = np.array([25.0, 16.30487823486328, 11.679261207580566, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y6 = np.array([25.0, 19.353580474853516, 13.894586563110352, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y7 = np.array([25.0, 19.506969451904297, 13.819287300109863, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y8 = np.array([25.0, 14.158235549926758, 10.10225772857666, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y9 = np.array([25.0, 13.122990608215332, 8.640877723693848, 7.411283016204834, 6.516513824462891, 6.19566535949707, 5.961752414703369, 5.528520107269287, np.nan ])
y10 = np.array([25.0, 14.354602813720703, 10.125027656555176, 8.795341491699219, 7.866785049438477, 7.464639663696289, 7.118199825286865, 6.609375, 6.504546642303467])

l2, = plt.plot(x, y2, color='blue', label='lay=12 (486MB)')
l3, = plt.plot(x, y3, color='red', label='lay=1r12 (176MB)')
l6, = plt.plot(x, y6, color='orange', label='lay=1r12 !MLPlayer (156MB)')
l7, = plt.plot(x, y7, color='yellow', label='lay=1r12 MLPlayerLast (176MB)')
l8, = plt.plot(x, y8, color='navy', label='lay=1r12 sharedLayerWeightsMLP (279MB)')
l9, = plt.plot(x, y9, color='purple', label='lay=1r18 (174MB)')
l10, = plt.plot(x, y10, color='darkviolet', label='lay=1r18 sharedLayerWeightsMLP (312MB)')

plt.xticks(np.arange(min(x), max(x)+0.5, 0.5))
plt.yticks(np.arange(0, 25.0+2.0, 2.0))

plt.xlabel("number of codeparrot-ds train samples (x1280000)")
plt.ylabel("Causal LM test perplexity")
plt.title("GPT2 Recursive Transformer")

plt.legend(handles=[l2, l3, l6, l7, l8, l9, l10])

plt.show()
