import numpy as np

from miracle_ad.data import chunk_waveform


def test_paper_chunking_geometry():
    sample_rate = 100
    waveform = np.arange(25 * sample_rate, dtype=np.float32)
    chunks, ranges = chunk_waveform(
        waveform,
        sample_rate=sample_rate,
        chunk_duration=10.0,
        overlap_factor=0.5,
    )
    assert chunks.shape == (4, 1000)
    np.testing.assert_allclose(ranges, [[0, 10], [5, 15], [10, 20], [15, 25]])


def test_short_recording_is_zero_padded():
    waveform = np.ones(250, dtype=np.float32)
    chunks, ranges = chunk_waveform(
        waveform, sample_rate=100, chunk_duration=10.0, overlap_factor=0.5
    )
    assert chunks.shape == (1, 1000)
    np.testing.assert_allclose(chunks[0, :250], 1.0)
    np.testing.assert_allclose(chunks[0, 250:], 0.0)
    np.testing.assert_allclose(ranges, [[0.0, 2.5]])
