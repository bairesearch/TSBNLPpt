import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3, 4])
y1 = np.array([0.5, 0.3887588016808033, 0.36320978958070277, 0.3523632300710678, 0.3456409868645668])
y2 = np.array([0.5, 0.3547375230026245, np.nan, np.nan, np.nan])
y3 = np.array([0.5, 0.3423201842537522, np.nan, np.nan, np.nan])
y4 = np.array([0.5, 0.32894625853195786, np.nan, np.nan, np.nan])
y5 = np.array([0.5, 0.3337461235207319, np.nan, np.nan, np.nan])
y6 = np.array([0.5, 0.31453748081862926, 0.28465476580947635, 0.2705436096695065, 0.2619548720240593])
y7 = np.array([0.5, 0.3243083935147524, 0.2944739804047346, 0.28517395846247673, 0.27500040173232554])
y8 = np.array([0.5, 0.34144704747617244, np.nan, np.nan, np.nan])

l1, = plt.plot(x, y1, color='green', label='# layers = 1 (120MB)')
l2, = plt.plot(x, y2, color='orange', label='# layers = 2 (147MB)')
l3, = plt.plot(x, y3, color='orange', label='# layers = 2, recursive (120MB)')
l4, = plt.plot(x, y4, color='magenta', label='# layers = 3 (174MB)')
l5, = plt.plot(x, y5, color='magenta', label='# layers = 3, recursive (120MB)')
l6, = plt.plot(x, y6, color='blue', label='# layers = 6 (256MB)')
l7, = plt.plot(x, y7, color='red', label='# layers = 6, recursive (120MB)')
l8, = plt.plot(x, y8, color='purple', label='# layers = 12, recursive (120MB)')


plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.yticks(np.arange(0, 0.5+0.1, 0.1))

plt.xlabel("number of oscar en train samples (x1000000)")
plt.ylabel("Masked LM test loss")
plt.title("RoBERTa Recursive Transformer")

plt.legend(handles=[l1, l2, l3, l4, l5, l6, l7, l8])

plt.show()
