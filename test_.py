import pytest
from utils.chem import *
from utils.common import *
from utils.evaluation.covmat import *
from models.common import *
from models.geometry import *
from rdkit.Chem import AllChem
import numpy as np


# test for set_rdmol_positions
def test_set_rdmol_positions():
    mol = Chem.MolFromSmiles('CC')
    m2=Chem.AddHs(mol)
    cids = AllChem.EmbedMultipleConfs(m2, numConfs=10)
    pos = torch.tensor([[0, 0, 0]])
    #print(mol.type)
    mol = set_rdmol_positions(m2, pos)
    assert np.isclose(mol.GetConformer(0).GetPositions().all(), pos.all())

# test for draw_mol_image
def test_draw_mol_image():
    mol = Chem.MolFromSmiles('CC')
    m2=Chem.AddHs(mol)
    cids = AllChem.EmbedMultipleConfs(m2, numConfs=10)
    img = draw_mol_image(m2)
    assert img is not None

# test for get_atom_symbol
def test_get_atom_symbol():
    symbol = get_atom_symbol(6)
    assert symbol == 'C'

# test for get_atoms_in_ring
def test_get_atoms_in_ring():
    mol = Chem.MolFromSmiles('CC')
    m2=Chem.AddHs(mol)
    cids = AllChem.EmbedMultipleConfs(m2, numConfs=10)
    atoms = get_atoms_in_ring(m2)
    assert atoms is not None

# test for get_2d_mol
def test_get_2d_mol():
    mol = Chem.MolFromSmiles('CC')
    m2=Chem.AddHs(mol)
    cids = AllChem.EmbedMultipleConfs(m2, numConfs=10)
    mol = get_2d_mol(m2)
    assert mol is not None

# test for draw_mol_svg
def test_draw_mol_svg():
    mol = Chem.MolFromSmiles('CC')
    m2=Chem.AddHs(mol)
    cids = AllChem.EmbedMultipleConfs(m2, numConfs=10)
    svg = draw_mol_svg(m2)
    assert svg is not None

class config():
    def __init__(self,name):
        self.type=name
        self.lr=1e-5
        self.beta1=0.9
        self.beta2=0.999
        self.weight_decay=0.1

# test for get_optimizer
def test_get_optimizer():
    model = torch.nn.Linear(10, 10)
    optimizer = get_optimizer(config('adam'), model)
    assert optimizer is not None

class sch_config():
    def __init__(self,name):
        self.type=name
        self.factor=0.1
        self.patience=10
        self.min_lr=1e-4
# # test get_scheduler
def test_get_scheduler():
    model = torch.nn.Linear(10, 10)
    optimizer = get_optimizer(config('adam'), model)
    scheduler = get_scheduler(sch_config('plateau'), optimizer)
    assert scheduler is not None

# test if output dimension is correct for multi-layer perceptron
def test_mlp():
    mlp = MultiLayerPerceptron(1, [100, 100],"relu")
    assert mlp(torch.tensor([1.0])).shape == torch.Size([100])

