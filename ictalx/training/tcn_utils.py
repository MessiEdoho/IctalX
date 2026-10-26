"""Verbatim copies from DL_WITH_SSL_GA/tcn_utils.py at commit e463396.

Source file SHA-256 at that commit: 67bb220683f85fc2a7b6a853eebd6d80c4705d7d2ebdf584c13c61d3dfa42422
Do not edit. Each block below is preceded by its source line numbers.
"""
import time
import numpy as np
import torch
import torch.nn as nn



# tcn_utils.py lines 470-574
def filter_extreme_segments(pairs, threshold=1000.0, logger=None):
    """Remove segments whose max absolute amplitude exceeds *threshold*.

    This is a hard filter, not a clipping operation.  Segments with
    catastrophic values (raw unscaled EEG from preprocessing failures)
    are removed entirely because their waveform morphology is not
    recoverable by clipping.

    The function also removes segments that contain NaN or Inf values,
    regardless of the amplitude threshold.

    Parameters
    ----------
    pairs : list of (str or Path, int)
        File-label pairs (filepath, label).
    threshold : float, default 1000.0
        Maximum allowed ``max(|x|)`` per segment.  Segments exceeding
        this value are removed.  1000.0 is appropriate for z-scored EEG
        (median / MAD normalisation).
    logger : logging.Logger, optional
        If provided, logs progress every 50 000 segments, a summary
        of removed segments, and per-subject breakdown.

    Returns
    -------
    list of (str or Path, int)
        Pairs with extreme segments removed.

    RESEARCH REPORTING NOTE
    -----------------------
    Report the threshold value, number of segments removed, and
    affected subjects.  Example:

      "Six segments from five subjects were removed prior to training
       because max(|x|) exceeded 1000 (worst: 4.36e+17 in m340),
       indicating incomplete z-score normalisation from interrupted
       preprocessing jobs."
    """
    t0 = time.time()
    clean = []
    removed = []
    worst_val = 0.0
    worst_file = ""

    for i, (fp, label) in enumerate(pairs):
        x = np.load(fp)

        # Check for NaN / Inf first
        if not np.isfinite(x).all():
            mx = float(np.nanmax(np.abs(x))) if np.any(np.isfinite(x)) else float("inf")
            removed.append((fp, label, mx))
            if mx > worst_val:
                worst_val = mx
                worst_file = fp
        else:
            mx = float(np.abs(x).max())
            if mx > threshold:
                removed.append((fp, label, mx))
                if mx > worst_val:
                    worst_val = mx
                    worst_file = fp
            else:
                clean.append((fp, label))

        if logger and (i + 1) % 50000 == 0:
            logger.info("  filter_extreme_segments: scanned %d/%d | removed so far: %d",
                        i + 1, len(pairs), len(removed))

    elapsed = time.time() - t0

    if logger:
        logger.info("=" * 60)
        logger.info("filter_extreme_segments summary:")
        logger.info("  Threshold          : %.1f", threshold)
        logger.info("  Total scanned      : %d", len(pairs))
        logger.info("  Removed            : %d (%.2f%%)",
                    len(removed), 100 * len(removed) / max(len(pairs), 1))
        logger.info("  Retained           : %d", len(clean))
        logger.info("  Scan time          : %.1f s", elapsed)

        if removed:
            logger.info("  Worst value        : %.4e in %s", worst_val, worst_file)

            subject_counts = {}
            for fp, label, mx in removed:
                subj = Path(fp).stem.split("_", 1)[0]
                if subj not in subject_counts:
                    subject_counts[subj] = {"count": 0, "worst": 0.0}
                subject_counts[subj]["count"] += 1
                subject_counts[subj]["worst"] = max(subject_counts[subj]["worst"], mx)

            logger.info("  Affected subjects  : %d", len(subject_counts))
            for subj in sorted(subject_counts):
                info = subject_counts[subj]
                logger.info("    %s: %d segments removed (worst=%.2e)",
                            subj, info["count"], info["worst"])

            catastrophic = sum(1 for _, _, mx in removed if mx > 1e6)
            mild = len(removed) - catastrophic
            logger.info("  Catastrophic (>1e6): %d", catastrophic)
            logger.info("  Mild (%.0f-1e6)     : %d", threshold, mild)

        logger.info("=" * 60)

    return clean


# tcn_utils.py lines 746-821
class CausalConvBlock(nn.Module):
    """Two-convolution causal residual block following Bai et al. (2018).

    Architecture per block (both convolutions share the same dilation):
        Conv1d(in_ch -> out_ch) -> LayerNorm -> GELU -> Dropout1d ->
        Conv1d(out_ch -> out_ch) -> LayerNorm -> GELU -> Dropout1d ->
                                                          + Residual

    The two-convolution design provides two non-linear transformations
    per dilation level before advancing to the next scale. Causal
    padding ensures no information leaks from future time steps.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    kernel_size : int
        Convolution kernel size (must be odd).
    dilation : int
        Dilation factor for this block (same for both convolutions).
    dropout : float
        Spatial dropout rate (drops entire channels).

    Example
    -------
    >>> block = CausalConvBlock(1, 64, kernel_size=7, dilation=4, dropout=0.2)
    >>> out = block(torch.randn(8, 1, 2500))  # (8, 64, 2500)
    """

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation    # total causal padding (left side only)

        # -- Sub-layer 1: expands channels from in_ch to out_ch ----------------
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               dilation=dilation, padding=self.pad)
        self.norm1 = nn.LayerNorm(out_ch)          # normalise across channel dim per time step
        self.act1 = nn.GELU()                      # smooth activation for bio-signal features
        self.drop1 = nn.Dropout1d(dropout)         # spatial dropout: drops entire channels

        # -- Sub-layer 2: operates at full width (out_ch -> out_ch) ------------
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               dilation=dilation, padding=self.pad)
        self.norm2 = nn.LayerNorm(out_ch)          # second normalisation layer
        self.act2 = nn.GELU()                      # second activation
        self.drop2 = nn.Dropout1d(dropout)         # second spatial dropout

        # 1x1 conv for residual projection when channel counts differ
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        """Forward pass. x shape: (batch, channels, time)."""
        seq_len = x.size(2)
        res = self.residual(x)                     # project input for skip connection

        # -- Sub-layer 1 -------------------------------------------------------
        out = self.conv1(x)                        # dilated conv with causal padding
        out = out[:, :, :seq_len]                  # trim right side to enforce causality
        out = out.transpose(1, 2)                  # (B, T, C) -- LayerNorm expects channels last
        out = self.norm1(out)                      # normalise across channel dimension
        out = out.transpose(1, 2)                  # (B, C, T) -- back to conv format
        out = self.act1(out)                       # GELU activation
        out = self.drop1(out)                      # spatial dropout

        # -- Sub-layer 2 -------------------------------------------------------
        out = self.conv2(out)                      # second dilated conv at same dilation
        out = out[:, :, :seq_len]                  # trim right side to enforce causality
        out = out.transpose(1, 2)                  # (B, T, C)
        out = self.norm2(out)                      # normalise across channel dimension
        out = out.transpose(1, 2)                  # (B, C, T)
        out = self.act2(out)                       # GELU activation
        out = self.drop2(out)                      # spatial dropout

        return out + res                           # residual connection


# tcn_utils.py lines 2146-2294
class MultiScaleTCN(nn.Module):
    """Multi-Scale Temporal Convolutional Network for binary EEG seizure detection.

    Three parallel branches of CausalConvBlocks, each using a distinct
    dilation schedule, process the input simultaneously. Branch outputs
    are fused and globally average-pooled before a linear classification head.

    Architecture
    ------------
    Input: (batch, 1, SEGMENT_LEN)
        |
        +-- Branch 1 dilations [1,  2,   4]    -- fine scale         (spike morphology)
        +-- Branch 2 dilations [8,  16,  32]   -- intermediate scale (rhythmic bursts)
        +-- Branch 3 dilations [32, 64, 128]   -- coarse scale       (seizure evolution)
        |  Each branch: CausalConvBlock x len(dilations)
        |  Each block: same two-conv structure as in TCN
        |  Output per branch: (batch, num_filters, T)
        |
        v
    Fusion -> (batch, num_filters, T)
      "concat": cat on channel dim -> 1x1 Conv1d projection
      "average": element-wise mean of branch outputs
        |
        v
    Global average pool -> (batch, num_filters)
        |
        v
    Linear head -> scalar logit (batch,)

    Parameters
    ----------
    num_filters : int
        Output channels per branch and in the fused representation.
    kernel_size : int
        Kernel width for all CausalConvBlocks. Must be odd.
    dropout : float
        Spatial dropout rate inside each CausalConvBlock.
    branch1_dilations : list of int
        Dilation schedule for Branch 1. Default [1, 2, 4].
    branch2_dilations : list of int
        Dilation schedule for Branch 2. Default [8, 16, 32].
    branch3_dilations : list of int
        Dilation schedule for Branch 3. Default [32, 64, 128].
    fusion : str
        Branch fusion strategy: "concat" (default) or "average".
    return_embedding : bool
        If True, return the globally pooled feature vector
        (batch, num_filters) instead of the scalar logit. Default: False.

    Receptive field per branch
    --------------------------
    RF_branch = 1 + 2 * sum((kernel_size - 1) * d for d in dilations)
    Factor of 2 accounts for the two convolutions per CausalConvBlock.
    """

    def __init__(self,
                 num_filters,
                 kernel_size,
                 dropout,
                 branch1_dilations=None,
                 branch2_dilations=None,
                 branch3_dilations=None,
                 fusion="concat",
                 return_embedding=False):
        super().__init__()

        # Mutable default arguments handled here to avoid shared-list pitfall
        if branch1_dilations is None:
            branch1_dilations = [1, 2, 4]
        if branch2_dilations is None:
            branch2_dilations = [8, 16, 32]
        if branch3_dilations is None:
            branch3_dilations = [32, 64, 128]

        if fusion not in ("concat", "average"):
            raise ValueError(
                "fusion must be 'concat' or 'average', got '%s'" % fusion)

        self.fusion           = fusion
        self.return_embedding = return_embedding
        self.num_filters      = num_filters

        def build_branch(dilations):
            """Build one sequential branch of CausalConvBlocks.

            Uses the EXACT parameter names of CausalConvBlock.__init__:
                in_ch, out_ch, kernel_size, dilation, dropout
            The two-conv-per-block structure is inherited -- not reimplemented.
            """
            blocks = []
            in_ch = 1                                  # single-channel EEG input
            for d in dilations:
                blocks.append(
                    CausalConvBlock(
                        in_ch=in_ch,
                        out_ch=num_filters,
                        kernel_size=kernel_size,
                        dilation=d,
                        dropout=dropout))
                in_ch = num_filters                    # subsequent blocks use num_filters input
            return nn.Sequential(*blocks)

        self.branch1 = build_branch(branch1_dilations)
        self.branch2 = build_branch(branch2_dilations)
        self.branch3 = build_branch(branch3_dilations)

        # Fusion projection for concat mode
        # Projects 3*num_filters -> num_filters so pooling and head always
        # see a fixed channel dimension
        if self.fusion == "concat":
            self.fusion_conv = nn.Conv1d(
                3 * num_filters, num_filters, kernel_size=1, bias=False)
        else:
            self.fusion_conv = None                    # register None for consistent state_dict

        self.classifier = nn.Linear(num_filters, 1)    # binary classification head

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, 1, SEGMENT_LEN)

        Returns
        -------
        torch.Tensor
            return_embedding=False: shape (batch,) logit
            return_embedding=True : shape (batch, num_filters)
        """
        out1 = self.branch1(x)                         # (batch, num_filters, T)
        out2 = self.branch2(x)                         # (batch, num_filters, T)
        out3 = self.branch3(x)                         # (batch, num_filters, T)

        if self.fusion == "concat":
            # Concat then project to preserve branch-specific features
            fused = torch.cat([out1, out2, out3], dim=1)  # (batch, 3*num_filters, T)
            fused = self.fusion_conv(fused)            # (batch, num_filters, T)
        else:
            # Equal-weight average of branch outputs
            fused = (out1 + out2 + out3) / 3.0         # (batch, num_filters, T)

        # Global average pooling collapses T to a scalar per filter
        pooled = fused.mean(dim=-1)                    # (batch, num_filters)

        if self.return_embedding:
            return pooled

        return self.classifier(pooled).squeeze(-1)     # (batch,) logit
