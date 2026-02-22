"""
Code adapted from https://github.com/snap-stanford/RelGNN/tree/main
Commit https://github.com/snap-stanford/RelGNN/commit/cffdb8b54627e92c7dd112c1243dde739c90d35b

"""

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_geometric.nn import LayerNorm
from torch_geometric.typing import EdgeType, NodeType

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import TransformerConv, SAGEConv
from torch.nn import Embedding, ModuleDict
import warnings

from torch_geometric.typing import NodeType
from torch_geometric.utils.hetero import check_add_self_loops

from collections import defaultdict

from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP

from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder


def get_atomic_routes(edge_type_list):

    src_to_tuples = defaultdict(list)
    for src, rel, dst in edge_type_list:
        if rel.startswith('f2p'):
            if src == dst:
                src = src + '--' + rel
            src_to_tuples[src].append((src, rel, dst))

    atomic_routes_list = []
    get_rev_edge = lambda edge: (edge[2], 'rev_' + edge[1], edge[0])
    for src, tuples in src_to_tuples.items():
        if '--' in src:
            src = src.split('--')[0]
        if len(tuples) == 1:
            _, rel, dst = tuples[0]
            edge = (src, rel, dst)
            atomic_routes_list.append(('dim-dim', ) + edge)
            atomic_routes_list.append(('dim-dim', ) + get_rev_edge(edge))
        else:
            for _, rel_q, dst_q in tuples:
                for _, rel_v, dst_v in tuples:
                    if rel_q != rel_v:
                        edge_q = (src, rel_q, dst_q)
                        edge_v = (src, rel_v, dst_v)
                        atomic_routes_list.append(('dim-fact-dim', ) + edge_q +
                                                  get_rev_edge(edge_v))

    return atomic_routes_list


class RelGNNConv(TransformerConv):

    def __init__(
        self,
        attn_type,
        in_channels,
        out_channels,
        heads,
        aggr,
        simplified_MP=False,
        bias=True,
        mlp_layers=1,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, heads, bias=bias, **kwargs)
        self.attn_type = attn_type
        if attn_type == 'dim-fact-dim':
            self.aggr_conv = SAGEConv(in_channels, out_channels, aggr=aggr)
        self.simplified_MP = simplified_MP
        self.final_proj = Linear(heads * out_channels, out_channels, bias=bias)
        self.final_proj.reset_parameters()

    def forward(
        self,
        x,
        edge_index,
        edge_attr=None,
        return_attention_weights=None,
    ):
        # dim-dim
        if self.attn_type == 'dim-dim':
            if self.simplified_MP and edge_index.shape[1] == 0:
                return None
            out = super().forward(x, edge_index, edge_attr,
                                  return_attention_weights)
            return self.final_proj(out)

        # dim-fact-dim
        edge_attn, edge_aggr = edge_index

        src_aggr, dst_aggr, dst_attn = x

        if self.simplified_MP:
            if edge_attn.shape[1] == 0:
                return None

            if edge_aggr.shape[1] == 0:
                src_attn = dst_aggr
            else:
                src_attn = self.aggr_conv((src_aggr, dst_aggr), edge_aggr)
        else:
            src_attn = self.aggr_conv((src_aggr, dst_aggr), edge_aggr)

        out = super().forward((src_attn, dst_attn), edge_attn, edge_attr,
                              return_attention_weights)

        return self.final_proj(out), src_attn


def group(xs: List[Tensor], aggr: Optional[str]) -> Optional[Tensor]:
    if len(xs) == 0:
        return None
    elif aggr is None:
        return torch.stack(xs, dim=1)
    elif len(xs) == 1:
        return xs[0]
    elif aggr == "cat":
        return torch.cat(xs, dim=-1)
    else:
        out = torch.stack(xs, dim=0)
        out = getattr(torch, aggr)(out, dim=0)
        out = out[0] if isinstance(out, tuple) else out
        return out


class RelGNN_HeteroConv(torch.nn.Module):
    r"""A generic wrapper for computing graph convolution on heterogeneous
    graphs.
    This layer will pass messages from source nodes to target nodes based on
    the bipartite GNN layer given for a specific edge type.
    If multiple relations point to the same destination, their results will be
    aggregated according to :attr:`aggr`.
    In comparison to :meth:`torch_geometric.nn.to_hetero`, this layer is
    especially useful if you want to apply different message passing modules
    for different edge types.

    .. code-block:: python

        hetero_conv = HeteroConv({
            ('paper', 'cites', 'paper'): GCNConv(-1, 64),
            ('author', 'writes', 'paper'): SAGEConv((-1, -1), 64),
            ('paper', 'written_by', 'author'): GATConv((-1, -1), 64),
        }, aggr='sum')

        out_dict = hetero_conv(x_dict, edge_index_dict)

        print(list(out_dict.keys()))
        >>> ['paper', 'author']

    Args:
        convs (Dict[Tuple[str, str, str], MessagePassing]): A dictionary
            holding a bipartite
            :class:`~torch_geometric.nn.conv.MessagePassing` layer for each
            individual edge type.
        aggr (str, optional): The aggregation scheme to use for grouping node
            embeddings generated by different relations
            (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"cat"`, :obj:`None`). (default: :obj:`"sum"`)
    """

    def __init__(
        self,
        convs,
        aggr: Optional[str] = "sum",
        simplified_MP: Optional[bool] = False,
    ):
        super().__init__()

        for edge_type, module in convs.items():
            check_add_self_loops(module, [edge_type])

        src_node_types = {key[0] for key in convs.keys()}
        dst_node_types = {key[-1] for key in convs.keys()}
        if len(src_node_types - dst_node_types) > 0:
            warnings.warn(
                f"There exist node types ({src_node_types - dst_node_types}) "
                f"whose representations do not get updated during message "
                f"passing as they do not occur as destination type in any "
                f"edge type. This may lead to unexpected behavior.")

        # Convert tuple keys to string keys for ModuleDict compatibility
        str_convs = {}
        self.edge_type_mapping = {}
        for edge_type_tuple, module in convs.items():
            edge_type_str = str(edge_type_tuple)
            str_convs[edge_type_str] = module
            self.edge_type_mapping[edge_type_str] = edge_type_tuple

        self.convs = ModuleDict(str_convs)
        self.aggr = aggr
        self.simplified_MP = simplified_MP

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs.values():
            conv.reset_parameters()

    def forward(
        self,
        x_dict,
        edge_index_dict,
    ) -> Dict[NodeType, Tensor]:
        r"""Runs the forward pass of the module.

        Args:
            x_dict (Dict[str, torch.Tensor]): A dictionary holding node feature
                information for each individual node type.
            edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): A
                dictionary holding graph connectivity information for each
                individual edge type, either as a :class:`torch.Tensor` of
                shape :obj:`[2, num_edges]` or a
                :class:`torch_sparse.SparseTensor`.
        """
        out_dict: Dict[str, List[Tensor]] = {}

        def update(out_dict, dst, out):
            if dst not in out_dict:
                out_dict[dst] = [out]
            else:
                out_dict[dst].append(out)

        for edge_type_str, conv in self.convs.items():
            edge_type_info = self.edge_type_mapping[edge_type_str]
            attn_type = edge_type_info[0]

            if attn_type == 'dim-dim':
                src, rel, dst = edge_type_info[1:]
                x = (
                    x_dict.get(src, None),
                    x_dict.get(dst, None),
                )
                edge_index = edge_index_dict[(src, rel, dst)]

                out = conv(x, edge_index)

                if self.simplified_MP and out is None:
                    continue

                update(out_dict, dst, out)

            elif attn_type == 'dim-fact-dim':
                edge_attn, edge_aggr = edge_type_info[1:4], edge_type_info[4:]
                src_attn, _, dst = edge_attn
                src_aggr = edge_aggr[0]
                x = (
                    x_dict[src_aggr],
                    x_dict[src_attn],
                    x_dict[dst],
                )
                edge_index = (
                    edge_index_dict[edge_attn],
                    edge_index_dict[edge_aggr],
                )
                out = conv(x, edge_index)

                if self.simplified_MP and out is None:
                    continue

                out_dst, out_src_attn = out
                update(out_dict, dst, out_dst)
                update(out_dict, src_attn, out_src_attn)

        for key, value in out_dict.items():
            out_dict[key] = group(value, self.aggr)

        if self.simplified_MP:
            for key, value in x_dict.items():
                if key not in out_dict:
                    out_dict[key] = value

        return out_dict

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_relations={len(self.convs)})'


class RelGNN(torch.nn.Module):

    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        aggr: str = "sum",
        num_model_layers: int = 2,
        num_heads: int = 1,
        simplified_MP=False,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_model_layers):
            conv = RelGNN_HeteroConv(
                {
                    edge_type:
                    RelGNNConv(edge_type[0], (channels, channels),
                               channels,
                               num_heads,
                               aggr=aggr,
                               simplified_MP=simplified_MP)
                    for edge_type in edge_types
                },
                aggr=aggr,
                simplified_MP=simplified_MP,
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_model_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[NodeType, Tensor],
        num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
        num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict


class RelGNN_Model(torch.nn.Module):

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_model_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        # List of node types to add shallow embeddings to input
        shallow_list: List[NodeType] = [],
        # ID awareness
        id_awareness: bool = False,
        atomic_routes=None,
        num_heads=None,
        simplified_MP=False,
        mlp_layers=1,
    ):
        super().__init__()

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types
                if "time" in data[node_type]
            ],
            channels=channels,
        )
        self.gnn = RelGNN(
            node_types=data.node_types,
            edge_types=atomic_routes,
            channels=channels,
            aggr=aggr,
            num_model_layers=num_model_layers,
            num_heads=num_heads,
            simplified_MP=simplified_MP,
        )
        self.head = MLP(
            channels,
            hidden_channels=channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=mlp_layers,
        )
        self.embedding_dict = ModuleDict({
            node:
            Embedding(data.num_nodes_dict[node], channels)
            for node in shallow_list
        })

        self.id_awareness_emb = None
        if id_awareness:
            self.id_awareness_emb = torch.nn.Embedding(1, channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict,
                                              batch.batch_dict)

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(
                batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )

        return self.head(x_dict[entity_table][:seed_time.size(0)])

    def forward_dst_readout(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
    ) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError(
                "id_awareness must be set True to use forward_dst_readout")
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        # Add ID-awareness to the root node
        x_dict[entity_table][:seed_time.size(0
                                             )] += self.id_awareness_emb.weight

        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict,
                                              batch.batch_dict)

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(
                batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )

        return self.head(x_dict[dst_table])
