import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
y1 = np.array([30.0, 16.17632293701172, 12.893939971923828, 11.613675117492676, 10.887641906738281, 10.431144714355469, 10.097999572753906, 9.906881332397461, 9.716487884521484])
y2 = np.array([30.0, 8.46451473236084, 6.351888179779053, 5.609471321105957, 5.203763008117676, 4.954327583312988, 4.783695220947266, 4.655602931976318, 4.573307037353516])
y3 = np.array([30.0, 11.413861274719238, 8.91283893585205, 8.011733055114746, 7.461851596832275, 7.120016574859619, 6.929925918579102, 6.687497615814209, 6.60063362121582])
y6 = np.array([30.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y7 = np.array([30.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y8 = np.array([30.0, 15.31370735168457, 10.615104675292969, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y13 = np.array([30.0, 15.526958465576172, 11.304801940917969, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
y14 = np.array([30.0, 26.357221603393555, 20.34063148498535, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

l1, = plt.plot(x, y1, color='green', label='lay=1 (176MB)')
l2, = plt.plot(x, y2, color='blue', label='lay=12 (486MB)')
l3, = plt.plot(x, y3, color='red', label='lay=1r12 (176MB)')
l6, = plt.plot(x, y6, color='orange', label='lay=1r12 !MLPlayer (156MB)')
l7, = plt.plot(x, y7, color='yellow', label='lay=1r12 MLPlayerLast (176MB)')
l8, = plt.plot(x, y8, color='navy', label='lay=1r12 sharedLayerWeightsMLP (276MB)')
l13, = plt.plot(x, y13, color='orange', label='lay=1r12 !MLPlayer (norm:463MB)')
l14, = plt.plot(x, y14, color='darkorange', label='lay=1 !MLPlayer (158MB)')

plt.xticks(np.arange(min(x), max(x)+0.5, 0.5))
plt.yticks(np.arange(0, 25.0+2.0, 2.0))

plt.xlabel("number of codeparrot-ds train samples (x1280000)")
plt.ylabel("Causal LM test perplexity")
plt.title("GPT2 Recursive Transformer")

plt.legend(handles=[l1, l2, l3, l8, l13, l14])

plt.show()
