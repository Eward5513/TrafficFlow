"""Spatial-first block; legacy temporal-first forward remains untouched."""
from MSTT.model.transformer_block import MSTTBlock


class SFTFBlock(MSTTBlock):
    execution_order = ("S", "FFN_S", "T", "FFN_T")

    def forward(self, hidden):
        hidden = hidden + self.dropout(self.spatial(self.spatial_norm(hidden)))
        hidden = hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))
        hidden = hidden + self.dropout(self.temporal(self.temporal_norm(hidden)))
        hidden = hidden + self.dropout(self.temporal_ffn(self.temporal_ffn_norm(hidden)))
        return hidden
