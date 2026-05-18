"""I/O layer tests."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest

from volumetric_qc import open_volume
from volumetric_qc.io.chunked import block_mean_downsample, iter_z_chunks, stratified_z_indices


def test_open_volume_from_numpy_3d():
    arr = np.random.rand(10, 32, 32).astype(np.float32)
    vol = open_volume(arr)
    assert vol.shape == (1, 10, 32, 32)
    assert vol.fmt == "array"


def test_open_volume_from_numpy_4d():
    arr = np.random.rand(2, 10, 32, 32).astype(np.float32)
    vol = open_volume(arr)
    assert vol.shape == (2, 10, 32, 32)
    assert vol.nchannels == 2
    assert vol.nz == 10


def test_open_volume_from_dask():
    arr = da.from_array(np.random.rand(3, 8, 16, 16).astype(np.float32), chunks=(1, 4, 16, 16))
    vol = open_volume(arr)
    assert vol.shape == (3, 8, 16, 16)


def test_z_slice_materializes_numpy():
    arr = np.random.rand(2, 4, 8, 8).astype(np.float32)
    vol = open_volume(arr)
    sl = vol.z_slice(1, 2)
    assert isinstance(sl, np.ndarray)
    assert sl.shape == (8, 8)


def test_channel_index_bounds():
    arr = np.random.rand(2, 4, 8, 8).astype(np.float32)
    vol = open_volume(arr)
    with pytest.raises(IndexError):
        vol.channel(5)


def test_block_mean_downsample_2d():
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    ds = block_mean_downsample(arr, 2)
    assert ds.shape == (2, 2)
    # Top-left 2x2 block mean = (0+1+4+5)/4 = 2.5
    assert ds[0, 0] == pytest.approx(2.5)


def test_block_mean_downsample_3d():
    arr = np.ones((4, 8, 8), dtype=np.float32)
    ds = block_mean_downsample(arr, 4)
    assert ds.shape == (4, 2, 2)
    assert np.allclose(ds, 1.0)


def test_block_mean_downsample_no_op():
    arr = np.ones((4, 4), dtype=np.float32)
    assert block_mean_downsample(arr, 1) is arr


def test_stratified_z_indices_deterministic():
    a = stratified_z_indices(100, 10, seed=42)
    b = stratified_z_indices(100, 10, seed=42)
    assert a == b


def test_stratified_z_indices_all_when_more_requested_than_available():
    out = stratified_z_indices(5, 50, seed=0)
    assert out == [0, 1, 2, 3, 4]


def test_iter_z_chunks_covers_strided_slices():
    arr = da.from_array(np.arange(80).reshape(20, 2, 2).astype(np.float32), chunks=(5, 2, 2))
    all_z = []
    for z0, block in iter_z_chunks(arr, z_stride=2, chunk_z=3):
        for i in range(block.shape[0]):
            all_z.append(z0 + i * 2)
    # z_stride=2 -> we expect 0,2,4,...,18
    assert all_z == list(range(0, 20, 2))
