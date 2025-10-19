from typing import Any, Dict, List, Callable

import torch
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType

from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from gnn_architectures import HeteroGNN, gin_conv_factory, graphconv_factory, gat_conv_factory, gatv2_conv_factory


class Model(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        # List of node types to add shallow embeddings to input
        shallow_list: List[NodeType] = [],
        # ID awareness
        id_awareness: bool = False,
        # GNN factory function - defaults to HeteroGraphSAGE for backward compatibility
        gnn_factory: Callable = None,
        mlp_layers: int = 1,
        **gnn_kwargs,
    ):
        """
        Args:
            data: HeteroData object
            col_stats_dict: Column statistics dictionary
            num_layers: Number of GNN layers
            channels: Number of channels
            out_channels: Output channels for the head
            aggr: Aggregation method
            norm: Normalization method
            shallow_list: List of node types to add shallow embeddings to input
            id_awareness: Whether to use ID awareness
            gnn_factory: Factory function to create GNN. Should accept (node_types, edge_types, channels, aggr, num_layers)
                        and return a GNN module. Defaults to HeteroGraphSAGE if None.
            **gnn_kwargs: Additional keyword arguments passed to gnn_factory
        """
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
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels,
        )

        # Use provided gnn_factory or default to HeteroGraphSAGE
        if gnn_factory is None:
            self.gnn = HeteroGraphSAGE(
                node_types=data.node_types,
                edge_types=data.edge_types,
                channels=channels,
                aggr=aggr,
                num_layers=num_layers,
            )
        else:
            self.gnn = gnn_factory(
                node_types=data.node_types,
                edge_types=data.edge_types,
                channels=channels,
                aggr=aggr,
                num_layers=num_layers,
                **gnn_kwargs,
            )

        self.head = MLP(
            channels,
            hidden_channels=channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=mlp_layers,
        )
        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )

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

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )

        return self.head(x_dict[entity_table][: seed_time.size(0)])

    def forward_dst_readout(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
    ) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError(
                "id_awareness must be set True to use forward_dst_readout"
            )
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        # Add ID-awareness to the root node
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )

        return self.head(x_dict[dst_table])


# Factory functions for different GNN architectures
def create_hetero_gin(node_types, edge_types, channels, aggr, num_layers, **kwargs):
    """Factory function to create HeteroGNN with GIN convolution."""
    return HeteroGNN(
        node_types=node_types,
        edge_types=edge_types,
        channels=channels,
        conv_factory=gin_conv_factory,
        aggr=aggr,
        num_layers=num_layers,
        **kwargs
    )


def create_hetero_graphconv(node_types, edge_types, channels, aggr, num_layers, **kwargs):
    """Factory function to create HeteroGNN with GraphConv convolution."""
    return HeteroGNN(
        node_types=node_types,
        edge_types=edge_types,
        channels=channels,
        conv_factory=graphconv_factory,
        aggr=aggr,
        num_layers=num_layers,
        **kwargs
    )


def create_hetero_gat(node_types, edge_types, channels, aggr, num_layers, **kwargs):
    """Factory function to create HeteroGNN with GAT convolution."""
    return HeteroGNN(
        node_types=node_types,
        edge_types=edge_types,
        channels=channels,
        conv_factory=gat_conv_factory,
        aggr=aggr,
        num_layers=num_layers,
        **kwargs
    )


def create_hetero_gatv2(node_types, edge_types, channels, aggr, num_layers, **kwargs):
    """Factory function to create HeteroGNN with GAT v2 convolution."""
    return HeteroGNN(
        node_types=node_types,
        edge_types=edge_types,
        channels=channels,
        conv_factory=gatv2_conv_factory,
        aggr=aggr,
        num_layers=num_layers,
        **kwargs
    )


# Usage examples:
"""
# Using default HeteroGraphSAGE (backward compatible)
model = Model(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=3,
    channels=64,
    out_channels=1,
    aggr="mean",
    norm="batch_norm"
)

# Using HeteroGNN with GIN convolution
model_gin = Model(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=3,
    channels=64,
    out_channels=1,
    aggr="mean",
    norm="batch_norm",
    gnn_factory=create_hetero_gin
)

# Using HeteroGNN with GraphConv convolution
model_graphconv = Model(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=3,
    channels=64,
    out_channels=1,
    aggr="mean",
    norm="batch_norm",
    gnn_factory=create_hetero_graphconv
)

# Using HeteroGNN with GAT convolution (4 attention heads)
model_gat = Model(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=3,
    channels=64,
    out_channels=1,
    aggr="mean",
    norm="batch_norm",
    gnn_factory=create_hetero_gat,
    heads=4,
    concat=True
)

# Using HeteroGNN with GAT v2 convolution (4 attention heads)
model_gatv2 = Model(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=3,
    channels=64,
    out_channels=1,
    aggr="mean",
    norm="batch_norm",
    gnn_factory=create_hetero_gatv2,
    heads=4,
    concat=True
)

# Using a custom lambda function for more control
# Note: you would need to import GCNConv from torch_geometric.nn
model_custom = Model(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=3,
    channels=64,
    out_channels=1,
    aggr="mean",
    norm="batch_norm",
    gnn_factory=lambda **kwargs: HeteroGNN(
        conv_factory=lambda in_ch, out_ch: GCNConv(in_ch, out_ch, improved=True),
        **kwargs
    )
)
"""
