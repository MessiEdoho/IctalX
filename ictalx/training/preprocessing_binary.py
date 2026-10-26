"""Verbatim copies from DL_WITH_SSL_GA/preprocessing_binary.py at commit e463396.

Source file SHA-256 at that commit: e41f9727d2e168c86bf5d5462dce81cd77ac3a942f43fb13f27358ab48570b1a
Do not edit. Each block below is preceded by its source line numbers.
"""
import numpy as np
from scipy.signal import iirnotch, filtfilt
from scipy.stats import median_abs_deviation

# preprocessing_binary.py lines 22-22
fs      = 500    # sampling rate in Hz -- must match the EDF file header
# preprocessing_binary.py lines 23-23
win_len = 2500   # window length in samples: 5 s * 500 Hz = 2500 samples per segment
# preprocessing_binary.py lines 24-24
step    = 1250   # step size in samples: 2.5 s * 500 Hz = 1250 samples (50% overlap)
# preprocessing_binary.py lines 27-27
segments_per_chunk = 40  # number of segments grouped into one RAM-loaded block (K)
# preprocessing_binary.py lines 30-30
notch_freq = 50.0  # powerline interference frequency to suppress (Hz)
# preprocessing_binary.py lines 31-31
notch_Q    = 30     # Q-factor of the notch filter: higher = narrower, more targeted notch


# preprocessing_binary.py lines 115-146
def compute_segment_grid(n_samples, win_len, step):
    """
    Pre-compute every valid segment start index for the entire recording.

    A start index i is valid when i + win_len <= n_samples, ensuring that
    every segment contains exactly win_len samples with no truncation.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the recording (from the EDF header).
    win_len : int
        Segment length in samples (e.g. 2500 for 5 s at 500 Hz).
    step : int
        Step size between consecutive segment starts (e.g. 1250 for 50% overlap).

    Returns
    -------
    starts : np.ndarray, dtype int64
        1-D array of every valid segment start index.
    """

    last_valid = n_samples - win_len  # largest sample index where a full window still fits

    starts = np.arange(             # generate evenly spaced start indices
        0,                          # first segment begins at sample 0
        last_valid + 1,             # +1 because arange excludes the upper bound
        step,                       # spacing equals the configured step size
        dtype=np.int64              # integer indices for array slicing
    )

    return starts  # return the complete grid of segment start indices


# preprocessing_binary.py lines 151-176
def apply_notch_filter(signal, fs, freq, Q):
    """
    Remove powerline interference with a zero-phase IIR notch filter.

    Parameters
    ----------
    signal : np.ndarray, shape (n,)
        1-D raw EEG signal (one chunk block).
    fs : float
        Sampling rate in Hz.
    freq : float
        Centre frequency of the notch in Hz (e.g. 50.0).
    Q : float
        Quality factor -- higher values produce a narrower notch.

    Returns
    -------
    filtered : np.ndarray, shape (n,)
        Signal with the specified frequency suppressed.
    """

    w0 = freq / (fs / 2.0)            # normalise notch frequency to Nyquist (must be in (0, 1))
    b, a = iirnotch(w0, Q)            # design the second-order IIR notch filter coefficients
    filtered = filtfilt(b, a, signal)  # apply forward-backward (zero-phase) filtering

    return filtered  # return the notch-filtered signal


# preprocessing_binary.py lines 179-197
def remove_dc(signal):
    """
    Remove the DC offset by subtracting the signal mean.

    Parameters
    ----------
    signal : np.ndarray, shape (n,)
        1-D EEG signal (already notch-filtered).

    Returns
    -------
    centred : np.ndarray, shape (n,)
        Signal with zero mean.
    """

    dc = np.mean(signal)   # compute the DC component (mean amplitude of the chunk)
    centred = signal - dc  # subtract it from every sample

    return centred  # return the DC-corrected signal


# preprocessing_binary.py lines 200-227
def robust_zscore(signal):
    """
    Normalise using the robust z-score: z = (x - median) / (1.4826 * MAD).

    The 1.4826 factor makes the denominator equal to the standard deviation
    for Gaussian data. If MAD = 0 (constant signal), the input is returned
    unchanged to avoid division by zero.

    Parameters
    ----------
    signal : np.ndarray, shape (n,)
        1-D EEG signal (already notch-filtered and DC-corrected).

    Returns
    -------
    normed : np.ndarray, shape (n,)
        Robustly normalised signal.
    """

    med = np.median(signal)                       # robust location estimate (centre of the distribution)
    mad = median_abs_deviation(signal, scale=1)   # MAD without the 1.4826 scaling factor

    if mad == 0.0:               # constant signal -- cannot normalise
        return signal            # return unchanged to avoid division by zero

    normed = (signal - med) / (1.4826 * mad)  # apply the robust z-score formula

    return normed  # return the normalised signal


# preprocessing_binary.py lines 230-257
def preprocess_chunk(chunk, fs, notch_freq, notch_Q):
    """
    Apply the full three-step preprocessing pipeline to one chunk block.

    Steps: (1) notch filter  (2) DC removal  (3) robust z-score.

    Parameters
    ----------
    chunk : np.ndarray, shape (n,)
        Raw 1-D signal for one chunk loaded from the EDF file.
    fs : float
        Sampling rate in Hz.
    notch_freq : float
        Notch filter centre frequency in Hz.
    notch_Q : float
        Notch filter Q-factor.

    Returns
    -------
    processed : np.ndarray, shape (n,)
        Fully preprocessed chunk ready for segment extraction.
    """

    step1 = apply_notch_filter(chunk, fs, notch_freq, notch_Q)  # step 1: remove 50 Hz powerline
    step2 = remove_dc(step1)                                     # step 2: subtract DC offset
    processed = robust_zscore(step2)                              # step 3: robust z-score normalisation

    return processed  # return the fully preprocessed chunk
