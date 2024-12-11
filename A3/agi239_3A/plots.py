import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
import os
from torchvision import datasets, transforms
from torch import optim, nn, unsqueeze
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Lambda, Compose
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import tarfile, sys

# plt.plot(range(5), [0.0031650410192771232, 0.0028633269980142357, 0.002594966785109864, 0.0022605173577328968, 0.002502553227165231], '-o', label='lr=1e-03')
# plt.plot(range(5), [0.007731382175777981, 0.015254152096401902, 0.006859211771761148, 0.007565116742734852, 0.005825696614252957], '-o',label='lr=1e-02')
# plt.plot(range(5), [0.004654977587856024, 0.004453237637505697, 0.004261726191478602, 0.004093769083383881, 0.00405370544852264], '-o',label='lr=1e-04')
# plt.plot(range(5), [0.00404618132344301, 0.004039355023919262, 0.004034300562935369, 0.00402999843546371, 0.004025004300348815], '-o',label='lr=1e-05')

# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title('Loss over time for validation set for different learning rates')
# plt.legend()
# plt.savefig('loss_lr.png', dpi=300)


# plt.plot(range(5), [0.0031650410192771232, 0.0028633269980142357, 0.002594966785109864, 0.0022605173577328968, 0.002502553227165231] , '-o', label='batch_size=16')
# plt.plot(range(5), [0.01308384236522793, 0.012966860896266414, 0.009789905387962861, 0.010797198192722405, 0.009511369838027452], '-o',label='batch_size=4')
# plt.plot(range(5), [0.006741597526940677, 0.008251433646622024, 0.005145266192427562, 0.0047450756808123115, 0.006215380649445582], '-o',label='batch_size=8')

# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title('Loss over time for validation set for different batch sizes')
# plt.legend()
# plt.savefig('loss_batch_size_val.png', dpi=300)

# plt.plot(range(5), [0.011005034855008125, 0.005670853477343917, 0.004480390001251363, 0.0036298692122101782, 0.003391864772629924], '-o', label='lr=1e-03')
# plt.plot(range(5), [0.007864213012158871, 0.004204366410104558, 0.003499100586818531, 0.003423882230609888, 0.002682394971200847], '-o',label='lr=3e-03')
# plt.plot(range(5), [0.03596488614082336, 0.03376077279448509, 0.026971014842391013, 0.01751457633972168, 0.012357400105148554], '-o',label='lr=1e-04')

# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title('Loss over time for validation set for different learning rates var network')
# plt.legend()
# plt.savefig('loss_lr_vn.png', dpi=300)


# plt.plot(range(5), [0.015536890324461273, 0.00936166596263065, 0.006864355189027265, 0.005654873954512368, 0.005994845116689976], '-o',label='batch_size=8')
# plt.plot(range(5), [0.007864213012158871, 0.004204366410104558, 0.003499100586818531, 0.003423882230609888, 0.002682394971200847] , '-o', label='batch_size=16')
# plt.plot(range(5), [0.004576910277083516, 0.0019504680984653532, 0.0020476998091675342, 0.0013482466971501707, 0.0015965508848894388], '-o',label='batch_size=32')

# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title('Loss over time for validation set for different batch sizes var network')
# plt.legend()
# plt.savefig('loss_batch_size_val_vn.png', dpi=300)


plt.plot(range(5), [0.0160076,0.010845,0.0096386,0.00870442,0.00738941], '-o', label='lr=1e-03')
plt.plot(range(5), [0.01559209,0.010375,0.010509,0.0108099,0.0108099959], '-o',label='lr=3e-03')
plt.plot(range(5), [0.05852960162907839, 0.04137731620967388, 0.032682354348897935, 0.027355985180288554, 0.023787305923178792], '-o',label='lr=1e-04')

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title('Loss over time for validation set for different learning rates fix network')
plt.legend()
plt.savefig('loss_lr_fn.png', dpi=300)


plt.plot(range(5), [0.03316448044905555, 0.020677614410676323, 0.020954178327358748, 0.014549419196069993, 0.015471981947913582], '-o',label='batch_size=8')
plt.plot(range(5), [0.0160076,0.010845,0.0096386,0.00870442,0.00738941] , '-o', label='batch_size=16')
plt.plot(range(5), [0.009267083651572467, 0.006605264805071056, 0.005434715081448667, 0.004694191637076437, 0.004553708301507868], '-o',label='batch_size=32')

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title('Loss over time for validation set for different batch sizes fix network')
plt.legend()
plt.savefig('loss_batch_size_val_fn.png', dpi=300)


