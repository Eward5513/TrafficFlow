"""Seq2seq GCN-LSTM reimplementation for Xuancheng26 12-to-1 reconstruction.

Not an official implementation of Abdelraouf et al., IEEE T-ITS 2023.
"""

from reimplementation.s2s_gcn_lstm.model.s2s_gcn_lstm import MODEL_LABEL, S2SGCNLSTM

__all__ = ["MODEL_LABEL", "S2SGCNLSTM"]
