import torch
from torch_geometric.typing import SparseTensor
from torch_geometric.nn import radius_graph


def edge_inds(
    x,
    batch,
    loop = True,
    edge_index_only = True,
    full=True,
    triangular_out=True,
    triangular_in=True,
    max_num_neighbors_node=15,#10, #this and next line limits the training to molecules with less than 10 atoms
    max_num_neighbors_edge=225,#100,
):

    edge_index = radius_graph(x, r=float('inf'), batch=batch, loop=loop, max_num_neighbors=max_num_neighbors_node)

    if edge_index_only:
        return edge_index
      
    row, col = edge_index  # j->i
    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=col, col=row, value=value,
                          sparse_sizes=(x.size(0), x.size(0)))

    i_node = adj_t.storage.row()
    j_node = adj_t.storage.col()
    i_edge = adj_t.storage.value()

    if full:
        num_edges = adj_t.set_value(None).sum(dim=1).to(torch.long)
        batch = batch.repeat_interleave(num_edges)
        edge_edge_index = radius_graph(value.view([-1,1]).float(), r=float('inf'), batch=batch, loop=True, max_num_neighbors=max_num_neighbors_edge)
        i_edge_full, j_edge_full = edge_edge_index

    if triangular_out:
        ###outgoing
        adj_t_row = adj_t[adj_t.storage.row()]
        
        idx_j_edge = adj_t_row.storage.value()
        idx_i_edge = adj_t_row.storage.row()
        ###

      ###incoming
    if triangular_in:
        adj_t_t = adj_t.t()
        adj_t_t_row = adj_t_t[adj_t.storage.row()]
        idx_t_j_edge = adj_t_t_row.storage.value()
        dim_row = adj_t_t_row.set_value(None).sum(dim=1).to(torch.long)
        idx_t_i_edge = adj_t_t.storage.value().repeat_interleave(dim_row)
        ###

    if full:
        if triangular_out:
            if triangular_in:
                return i_node, j_node, i_edge, idx_i_edge, idx_j_edge, idx_t_i_edge, idx_t_j_edge, i_edge_full, j_edge_full
            else:
                return i_node, j_node, i_edge, idx_i_edge, idx_j_edge, i_edge_full, j_edge_full
        elif triangular_in:
            return i_node, j_node, i_edge, idx_t_i_edge, idx_t_j_edge, i_edge_full, j_edge_full
        else:
            return i_node, j_node, i_edge, i_edge_full, j_edge_full
    elif triangular_out:
        if triangular_in:
            return i_node, j_node, i_edge, idx_i_edge, idx_j_edge, idx_t_i_edge, idx_t_j_edge
        else:
            return i_node, j_node, i_edge, idx_i_edge, idx_j_edge      
    elif triangular_in:
        return i_node, j_node, i_edge, idx_t_i_edge, idx_t_j_edge
