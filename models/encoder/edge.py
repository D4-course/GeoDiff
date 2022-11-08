import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, ModuleList, Linear, Embedding
from torch_geometric.nn import MessagePassing, radius_graph
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from math import pi as PI

from utils.chem import BOND_TYPES
from ..common import MeanReadout, SumReadout, MultiLayerPerceptron


class GaussianSmearingEdgeEncoder(Module):
    """Set the number of gaussians needed and the cutoff required.
    Then setup a radial basis function that uses gaussian
    smearing and create an embedding table that can store
    the attributes of edges.
    """
    def __init__(self, num_gaussians=64, cutoff=10.0):
        super().__init__()
        #self.NUM_BOND_TYPES = 22
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        # Larger `stop` to encode more cases
        self.rbf = GaussianSmearing(
            start=0.0, stop=cutoff * 2, num_gaussians=num_gaussians)
        self.bond_emb = Embedding(100, embedding_dim=num_gaussians)

    @property
    def out_channels(self):
        return self.num_gaussians * 2

    def forward(self, edge_length, edge_type):
        """
        Encode the edge attributes using the distance 
        and the bond type
        Input:
            edge_length: The length of edges, shape=(E, 1).
            edge_type: The type pf edges, shape=(E,)
        Returns:
            edge_attr:  The representation of edges. (E, 2 * num_gaussians)
        """
        edge_attr = torch.cat(
            [self.rbf(edge_length), self.bond_emb(edge_type)], dim=1)
        return edge_attr


class MLPEdgeEncoder(Module):
    """Setup the dimensions and the bond type embeddings. Also setup an MLP with an input dimension of 1, 
    hidden dimension specified and a specified activation function (default to RELU)
    """
    def __init__(self, hidden_dim=100, activation='relu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bond_emb = Embedding(100, embedding_dim=self.hidden_dim)
        self.mlp = MultiLayerPerceptron(
            1, [self.hidden_dim, self.hidden_dim], activation=activation)

    @property
    def out_channels(self):
        return self.hidden_dim

    def forward(self, edge_length, edge_type):
        """
        Sends the edge length through the MLP (acts like an RBF)
        Sends the edge attributes through the embedding generated earlier and 
        return the product
        Input:
            edge_length: The length of edges, shape=(E, 1).
            edge_type: The type pf edges, shape=(E,)
        Returns:
            edge_attr:  The representation of edges. (E, 2 * num_gaussians)
        """
        d_emb = self.mlp(edge_length)  # (num_edge, hidden_dim)
        edge_attr = self.bond_emb(edge_type)  # (num_edge, hidden_dim)
        return d_emb * edge_attr  # (num_edge, hidden)


def get_edge_encoder(cfg):
    "Specify if you need an MLP or Gaussian Smearing Edge Encoder"
    if cfg.edge_encoder == 'mlp':
        return MLPEdgeEncoder(cfg.hidden_dim, cfg.mlp_act)
    if cfg.edge_encoder == 'gaussian':
        return GaussianSmearingEdgeEncoder(cfg.hidden_dim // 2, cutoff=cfg.cutoff)
    raise NotImplementedError(
        'Unknown edge encoder: %s' % cfg.edge_encoder)