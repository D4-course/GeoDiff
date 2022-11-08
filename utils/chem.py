"""
Set of utility functions to visualise and manipulate
an RDKit molecule.
"""

from copy import deepcopy
from typing import List, Tuple
import torch
from torchvision.transforms.functional import to_tensor  # pylint: disable=import-error
import rdkit  # pylint: disable=import-error
import rdkit.Chem.Draw  # pylint: disable=import-error
from rdkit import Chem  # pylint: disable=import-error
from rdkit.Chem import rdDepictor as DP  # pylint: disable=import-error
from rdkit.Chem import PeriodicTable as PT  # pylint: disable=import-error
from rdkit.Chem import rdMolAlign as MA  # pylint: disable=import-error
from rdkit.Chem.rdchem import BondType as BT  # pylint: disable=import-error
from rdkit.Chem.rdchem import Mol, GetPeriodicTable  # pylint: disable=import-error
from rdkit.Chem.Draw import rdMolDraw2D as MD2  # pylint: disable=import-error
from rdkit.Chem.rdmolops import RemoveHs  # pylint: disable=import-error


BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}
BOND_NAMES = {i: t for i, t in enumerate(BT.names.keys())}


def set_conformer_positions(conf, pos):
    """
    Create a molecule with atoms in the given positions
    """
    for i in range(pos.shape[0]):
        conf.SetAtomPosition(i, pos[i].tolist())
    return conf


def draw_mol_image(rdkit_mol, tensor=False):
    """
    Convert the molecule into an image represenation of it
    """
    rdkit_mol.UpdatePropertyCache()
    img = rdkit.Chem.Draw.MolToImage(rdkit_mol, kekulize=False)
    if tensor:
        return to_tensor(img)
    return img


def update_data_rdmol_positions(data):
    """
    Update the positions of the atoms
    """
    for i in range(data.pos.size(0)):
        data.rdmol.GetConformer(0).SetAtomPosition(i, data.pos[i].tolist())
    return data


def update_data_pos_from_rdmol(data):
    """
    Get the positions of atoms as a Tensor
    """
    new_pos = torch.FloatTensor(
        data.rdmol.GetConformer(0).GetPositions()).to(data.pos)
    data.pos = new_pos
    return data


def set_rdmol_positions(rdkit_mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    mol = deepcopy(rdkit_mol)
    set_rdmol_positions_(mol, pos)
    return mol


def set_rdmol_positions_(mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    for i in range(pos.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol


def get_atom_symbol(atomic_number):
    """
    Get atomic symbol on the periodic table
    """
    return PT.GetElementSymbol(GetPeriodicTable(), atomic_number)


def mol_to_smiles(mol: Mol) -> str:
    """
    Convert the molecule into its smile representation
    """
    return Chem.MolToSmiles(mol, allHsExplicit=True)


def mol_to_smiles_without_hs(mol: Mol) -> str:
    'Remove Hydrogens'
    return Chem.MolToSmiles(Chem.RemoveHs(mol))


def remove_duplicate_mols(molecules: List[Mol]) -> List[Mol]:
    'Remove Duplicates from the list of conformations'
    unique_tuples: List[Tuple[str, Mol]] = []

    for molecule in molecules:
        duplicate = False
        smiles = mol_to_smiles(molecule)
        for unique_smiles, _ in unique_tuples:
            if smiles == unique_smiles:
                duplicate = True
                break

        if not duplicate:
            unique_tuples.append((smiles, molecule))

    return [mol for smiles, mol in unique_tuples]


def get_atoms_in_ring(mol):
    'Get atoms in a ring'
    atoms = set()
    for ring in mol.GetRingInfo().AtomRings():
        for atom in ring:
            atoms.add(atom)
    return atoms


def get_2d_mol(mol):
    'Convert the 3D representation into 2D'
    mol = deepcopy(mol)
    DP.Compute2DCoords(mol)
    return mol


def draw_mol_svg(mol, mol_size=(450, 150), kekulize=False):
    'Get the SVG representation of the molecule'
    rdkit_mol = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(rdkit_mol)
        except Exception as error:
            rdkit_mol = Chem.Mol(mol.ToBinary())
    if not rdkit_mol.GetNumConformers():
        DP.Compute2DCoords(rdkit_mol)
    drawer = MD2.MolDraw2DSVG(mol_size[0], mol_size[1])
    drawer.DrawMolecule(rdkit_mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # It seems that the svg renderer used doesn't quite hit the spec.
    # Here are some fixes to make it work in the notebook, although I think
    # the underlying issue needs to be resolved at the generation step
    # return svg.replace('svg:','')
    return svg


def get_best_rmsd(probe, ref):
    'Remove Hydrogens and then compute the least error between the two'
    probe = RemoveHs(probe)
    ref = RemoveHs(ref)
    rmsd = MA.GetBestRMS(probe, ref)
    return rmsd
