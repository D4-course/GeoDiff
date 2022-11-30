import os
import argparse
import pickle
import torch
from torch_geometric.data import Data
from utils.chem import draw_mol_image
import matplotlib.pyplot as plt

with open('./log-1/model/sample_2022_11_02__12_19_27/samples_all.pkl', 'rb') as f:
    data = pickle.load(f)
    final = data[-1]
    plt.imshow(draw_mol_image(final['rdmol']))
    plt.savefig('test.png')
    plt.show()
