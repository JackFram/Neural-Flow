import numpy as np

import matplotlib.pylab as plt

with open('./results/oshs-emotion-quantize-results.npy', 'rb') as f:
    prev_result = np.load(f)

quant_acc = list(prev_result[0,:])


with open('./results/oshs-emotion-ft-results.npy', 'rb') as f:
    prev_result = np.load(f)

oshs_acc = list(prev_result[0,:])
range = list(prev_result[3,:])

plt.plot(range[:len(quant_acc)], quant_acc, label="pure_quant_acc")
plt.plot(range, oshs_acc, label="oshs_acc")
plt.legend()

plt.savefig("./results/QP_OSHS_ft_emotion.pdf", bbox_inches="tight", dpi=500)