import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Small residual MLP block used to refine the hidden representation
    without losing the original information.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = self.mlp(x)

        # Keep the original features and add the learned transformation
        x = residual + x

        # Helps keep the activations stable during training
        x = self.norm(x)

        return x


class RoleEncoder(nn.Module):
    """
    Encoder used for one node role in the edge.

    We use different encoders for source and destination nodes because
    their semantic role in a provenance event is not necessarily the same.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        num_residual_blocks: int = 2,
    ):
        super().__init__()

        # A few residual blocks to learn a richer representation
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(
                    dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_residual_blocks)
            ]
        )

        # Gating layer used to learn which hidden features are more useful
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # Maps the hidden representation to the final embedding dimension
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        # Refine the semantic representation
        h = self.residual_blocks(x)

        # Compute an importance weight for each hidden feature
        gate = self.gate(h)

        # Keep useful features and reduce the effect of less useful ones
        h = h * gate

        # Produce the final latent representation
        z = self.output_projection(h)

        return z


class VigilEncoder(nn.Module):
    """
    Main semantic base encoder used in VIGIL.

    The input features are expected to come from the Word2Vec
    featurization stage of PIDSMaker.

    This version does not use graph message passing yet. It is intended
    to serve as the non-GNN baseline before comparing it with a
    GraphSAGE/GAT-based encoder.
    """

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        num_residual_blocks: int = 2,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        # Shared projection applied to both source and destination nodes.
        # This adapts the original Word2Vec features to the VIGIL latent space.
        self.semantic_encoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Source node branch
        self.src_encoder = RoleEncoder(
            hidden_dim=hid_dim,
            output_dim=out_dim,
            dropout=dropout,
            num_residual_blocks=num_residual_blocks,
        )

        # Destination node branch
        self.dst_encoder = RoleEncoder(
            hidden_dim=hid_dim,
            output_dim=out_dim,
            dropout=dropout,
            num_residual_blocks=num_residual_blocks,
        )

    def encode_nodes(self, features):
        """Return the shared, role-independent embedding for each node.

        Incident signatures pool these vectors; source/destination role heads
        remain available to the detector's edge-prediction path.
        """
        if features.ndim != 2 or features.shape[1] != self.in_dim:
            raise ValueError("Node features must have shape [nodes, in_dim]")
        return self.semantic_encoder(features)

    def forward(
        self,
        x_src,
        x_dst,
        edge_index=None,
        **kwargs,
    ):
        # First map the Word2Vec features to a shared hidden space
        h_src = self.encode_nodes(x_src)
        h_dst = self.encode_nodes(x_dst)

        # Then learn source-specific and destination-specific embeddings
        z_src = self.src_encoder(h_src)
        z_dst = self.dst_encoder(h_dst)

        return z_src, z_dst


class VigilPIDSEncoder(VigilEncoder):
    """Adapt event-aligned role embeddings; keep encode_nodes() for memory."""

    def forward(self, x_src, x_dst, **kwargs):
        src, dst = super().forward(x_src, x_dst, **kwargs)
        return {"h": (src, dst), "h_src": src, "h_dst": dst}
