"""
A set of geometric functions to be used for the models.
"""

import torch
from torch_scatter import scatter_add # pylint: disable=import-error


def get_distance(pos, edge_index):
    """Get the distance between node of the graph
    using their positions
    """
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)


def eq_transform(score_d, pos, edge_index, edge_length):
    """Score a graph on the basis of edge length and position.
    Take the normalised length and then score the graph separately
    on the basis of start and end nodes.
    """
    coord_size = pos.size(0)
    dd_dr = (1. / edge_length) * (pos[edge_index[0]] - pos[edge_index[1]])   # (E, 3)
    score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0, dim_size=coord_size) \
        + scatter_add(- dd_dr * score_d, edge_index[1], dim=0, dim_size=coord_size) # (N, 3)
    return score_pos


def convert_cluster_score_d(cluster_score_d, cluster_pos, cluster_edge_index,\
                            cluster_edge_length, subgraph_index):
    """
    Args:
        cluster_score_d:    (E_c, 1)
        subgraph_index:     (N, )
    """
    cluster_score_pos = eq_transform(cluster_score_d, cluster_pos,\
                                     cluster_edge_index, cluster_edge_length)  # (C, 3)
    score_pos = cluster_score_pos[subgraph_index]
    return score_pos


def get_angle(pos, angle_index):
    """
    Args:
        pos:  (N, 3)
        angle_index:  (3, A), left-center-right.
    Find the cosine of the angle between two edge by first
    computing the edge lengths, normalising them and taking the acos
    """
    left, ctr, right = angle_index   # (A, )
    v1 = pos[left] - pos[ctr] # (A, 3)
    v2 = pos[right] - pos[ctr]
    inner_prod = torch.sum(v1 * v2, dim=-1, keepdim=True)   # (A, 1)
    length_prod = torch.norm(v1, dim=-1, keepdim=True) *\
                  torch.norm(v2, dim=-1, keepdim=True)   # (A, 1)
    angle = torch.acos(inner_prod / length_prod)    # (A, 1)
    return angle


def get_dihedral(pos, dihedral_index):
    """
    Args:
        pos:  (N, 3)
        dihedral:  (4, A)
    Get the dihedral angle by finding the plane on which the 
    edges lie and then finding the angle between the two planes.
    """
    left, ctr1, ctr2, right = dihedral_index # (A, )
    v_ctr = pos[ctr2] - pos[ctr1]   # (A, 3)
    v1 = pos[left] - pos[ctr1]
    v2 = pos[right] - pos[ctr2]
    n1 = torch.cross(v_ctr, v1, dim=-1) # Normal vectors of the two planes
    n2 = torch.cross(v_ctr, v2, dim=-1)
    inner_prod = torch.sum(n1 * n2, dim=1, keepdim=True)    # (A, 1)
    length_prod = torch.norm(n1, dim=-1, keepdim=True) *\
                  torch.norm(n2, dim=-1, keepdim=True)
    dihedral = torch.acos(inner_prod / length_prod)
    return dihedral