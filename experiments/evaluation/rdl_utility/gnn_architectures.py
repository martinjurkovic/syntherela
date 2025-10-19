from typing import Any, Dict, List, Optional, Callable

import torch
import torch_frame
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_frame.nn.models import ResNet
from torch_geometric.nn import HeteroConv, LayerNorm, PositionalEncoding, GINConv, GraphConv, GATConv, GATv2Conv
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.nn import MLP


class HeteroGNN(torch.nn.Module):
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        conv_factory: Callable[[int, int], torch.nn.Module],
        aggr: str = "mean",
        num_layers: int = 2,
        **conv_kwargs,
    ):
        """
        A flexible heterogeneous GNN that can use different convolution layers.

        Args:
            node_types: List of node types in the heterogeneous graph
            edge_types: List of edge types in the heterogeneous graph
            channels: Number of channels/features
            conv_factory: A callable that takes (in_channels, out_channels) and returns a conv layer
            aggr: Aggregation method for HeteroConv
            num_layers: Number of GNN layers
            **conv_kwargs: Additional keyword arguments passed to conv_factory
        """
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: conv_factory(channels, channels, **conv_kwargs)
                    for edge_type in edge_types
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
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


# Convenience factory functions for common GNN types
def gin_conv_factory(in_channels: int, out_channels: int, aggr: str = "mean", **kwargs):
    """Factory function for GIN convolution layers."""
    mlp = MLP([in_channels, out_channels, out_channels])
    return GINConv(mlp, aggr=aggr, **kwargs)


def graphconv_factory(in_channels: int, out_channels: int, **kwargs):
    """Factory function for GraphConv layers (GCN-like but supports heterogeneous graphs)."""
    # GraphConv doesn't have add_self_loops parameter, so we don't need to set it
    return GraphConv(in_channels, out_channels, **kwargs)


def gat_conv_factory(in_channels: int, out_channels: int, heads: int = 4, concat: bool = True, **kwargs):
    """Factory function for GAT convolution layers."""
    # Set add_self_loops=False for heterogeneous graphs to avoid HeteroConv issues
    kwargs.setdefault('add_self_loops', False)
    if concat:
        # When concatenating heads, each head should output out_channels/heads features
        head_out_channels = out_channels // heads
    else:
        head_out_channels = out_channels
    return GATConv(in_channels, head_out_channels, heads=heads, concat=concat, **kwargs)


def gatv2_conv_factory(in_channels: int, out_channels: int, heads: int = 4, concat: bool = True, **kwargs):
    """Factory function for GAT v2 convolution layers."""
    # Set add_self_loops=False for heterogeneous graphs to avoid HeteroConv issues
    kwargs.setdefault('add_self_loops', False)
    if concat:
        # When concatenating heads, each head should output out_channels/heads features
        head_out_channels = out_channels // heads
    else:
        head_out_channels = out_channels
    return GATv2Conv(in_channels, head_out_channels, heads=heads, concat=concat, **kwargs)


# Usage examples:
"""
# Using GIN convolution (equivalent to your original implementation)
gin_model = HeteroGNN(
    node_types=['node_type_1', 'node_type_2'],
    edge_types=[('node_type_1', 'edge_relation', 'node_type_2')],
    channels=64,
    conv_factory=gin_conv_factory,
    aggr="mean",
    num_layers=2
)

# Using GraphConv convolution (GCN-like but supports heterogeneous graphs)
gcn_model = HeteroGNN(
    node_types=['node_type_1', 'node_type_2'],
    edge_types=[('node_type_1', 'edge_relation', 'node_type_2')],
    channels=64,
    conv_factory=graphconv_factory,
    num_layers=2
)

# Using GAT convolution with 4 attention heads
gat_model = HeteroGNN(
    node_types=['node_type_1', 'node_type_2'],
    edge_types=[('node_type_1', 'edge_relation', 'node_type_2')],
    channels=64,
    conv_factory=gat_conv_factory,
    num_layers=2,
    heads=4,
    concat=True
)

# Using GAT v2 convolution with 4 attention heads
gatv2_model = HeteroGNN(
    node_types=['node_type_1', 'node_type_2'],
    edge_types=[('node_type_1', 'edge_relation', 'node_type_2')],
    channels=64,
    conv_factory=gatv2_conv_factory,
    num_layers=2,
    heads=4,
    concat=True
)

# Using a custom lambda function for more control
custom_model = HeteroGNN(
    node_types=['node_type_1', 'node_type_2'],
    edge_types=[('node_type_1', 'edge_relation', 'node_type_2')],
    channels=64,
    conv_factory=lambda in_ch, out_ch: GraphConv(in_ch, out_ch, improved=True),
    num_layers=3
)
"""
