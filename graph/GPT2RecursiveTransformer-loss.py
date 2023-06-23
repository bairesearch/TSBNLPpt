import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
y1 = np.array([4.0, 2.783548593521118, 2.5567574501037598, 2.452183246612549, 2.3876283168792725, 2.3447959423065186, 2.3123373985290527, 2.293229579925537, 2.2738242149353027])
y2 = np.array([4.0, 2.135882616043091, 1.8487520217895508, 1.7244564294815063, 1.6493819952011108, 1.6002614498138428, 1.5652133226394653, 1.5380713939666748, 1.520236611366272])
y3 = np.array([4.0, 2.434828519821167, 2.187492847442627, 2.080907106399536, 2.009803533554077, 1.9629100561141968, 1.9358490705490112, 1.9002397060394287, 1.8871656656265259])
y4 = np.array([4.0, 2.2140212059020996, 1.9371627569198608, 1.7961446046829224, 1.7178380489349365, 1.6728403568267822, 1.6263391971588135, 1.6075694561004639, 1.5847148895263672])
y5 = np.array([4.0, 2.5302352905273438, 2.2657041549682617, 2.15181040763855, 2.0746896266937256, 2.02394962310791, 1.981958270072937, 1.9619404077529907, 1.9405854940414429])
#y6 = np.array([4.0, 2.264317274093628, 2.0197391510009766, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

l1, = plt.plot(x, y1, color='green', label='lay=1, hid=768, head=12 (176MB)')
l2, = plt.plot(x, y2, color='blue', label='lay=12, hid=768, head=12 (486MB)')
l3, = plt.plot(x, y3, color='red', label='lay=1r12, hid=768, head=12 (176MB)')
l4, = plt.plot(x, y4, color='pink', label='lay=1r12, hid=1792, head=28 (norm:496MB)')
l5, = plt.plot(x, y5, color='lightgreen', label='lay=1, hid=768, head=24 (norm:407MB)')
#l6, = plt.plot(x, y6, color='hotpink', label='lay=1r12, hid=1536, head=24 (norm:407MB)')

plt.xticks(np.arange(min(x), max(x)+0.5, 0.5))
plt.yticks(np.arange(0, 5.0+0.1, 0.5))

plt.xlabel("number of codeparrot-ds train samples (x1280000)")
plt.ylabel("Causal LM test loss")
plt.title("GPT2 Recursive Transformer")

plt.legend(handles=[l2, l3, l4, l1, l5])

plt.show()
