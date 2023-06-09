from __future__ import annotations

import math
from itertools import compress
from typing import TYPE_CHECKING, NamedTuple

import dask
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph, MaterializedLayer

from distributed.exceptions import Reschedule
from distributed.shuffle._shuffle import (
    ShuffleId,
    ShuffleType,
    _get_worker_extension,
    barrier_key,
    shuffle_barrier,
)

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import TypeAlias

    import dask.array as da


ChunkedAxis: TypeAlias = tuple[float, ...]  # chunks must either be an int or NaN
ChunkedAxes: TypeAlias = tuple[ChunkedAxis, ...]
NIndex: TypeAlias = tuple[int, ...]
NSlice: TypeAlias = tuple[slice, ...]


def rechunk_transfer(
    input: np.ndarray,
    id: ShuffleId,
    input_chunk: NIndex,
    new: ChunkedAxes,
    old: ChunkedAxes,
) -> int:
    try:
        return _get_worker_extension().add_partition(
            input,
            input_partition=input_chunk,
            shuffle_id=id,
            type=ShuffleType.ARRAY_RECHUNK,
            new=new,
            old=old,
        )
    except Exception as e:
        raise RuntimeError(f"rechunk_transfer failed during shuffle {id}") from e


def rechunk_unpack(
    id: ShuffleId, output_chunk: NIndex, barrier_run_id: int
) -> np.ndarray:
    try:
        return _get_worker_extension().get_output_partition(
            id, barrier_run_id, output_chunk
        )
    except Reschedule as e:
        raise e
    except Exception as e:
        raise RuntimeError(f"rechunk_unpack failed during shuffle {id}") from e


def rechunk_p2p(x: da.Array, chunks: ChunkedAxes) -> da.Array:
    import numpy as np

    import dask.array as da

    if x.size == 0:
        # Special case for empty array, as the algorithm below does not behave correctly
        return da.empty(x.shape, chunks=chunks, dtype=x.dtype)

    old_chunks = x.chunks
    new_chunks = chunks

    def is_unknown(dim: ChunkedAxis) -> bool:
        return any(math.isnan(chunk) for chunk in dim)

    old_is_unknown = [is_unknown(dim) for dim in old_chunks]
    new_is_unknown = [is_unknown(dim) for dim in new_chunks]

    if old_is_unknown != new_is_unknown or any(
        new != old for new, old in compress(zip(old_chunks, new_chunks), old_is_unknown)
    ):
        raise ValueError(
            "Chunks must be unchanging along dimensions with missing values.\n\n"
            "A possible solution:\n  x.compute_chunk_sizes()"
        )

    old_known = [dim for dim, unknown in zip(old_chunks, old_is_unknown) if not unknown]
    new_known = [dim for dim, unknown in zip(new_chunks, new_is_unknown) if not unknown]

    old_sizes = [sum(o) for o in old_known]
    new_sizes = [sum(n) for n in new_known]

    if old_sizes != new_sizes:
        raise ValueError(
            f"Cannot change dimensions from {old_sizes!r} to {new_sizes!r}"
        )

    dsk: dict = {}
    token = tokenize(x, chunks)
    _barrier_key = barrier_key(ShuffleId(token))
    name = f"rechunk-transfer-{token}"
    transfer_keys = []
    for index in np.ndindex(tuple(len(dim) for dim in x.chunks)):
        transfer_keys.append((name,) + index)
        dsk[(name,) + index] = (
            rechunk_transfer,
            (x.name,) + index,
            token,
            index,
            chunks,
            x.chunks,
        )

    dsk[_barrier_key] = (shuffle_barrier, token, transfer_keys)

    name = f"rechunk-p2p-{token}"

    for index in np.ndindex(tuple(len(dim) for dim in chunks)):
        dsk[(name,) + index] = (rechunk_unpack, token, index, _barrier_key)

    with dask.annotate(shuffle=lambda key: key[1:]):
        layer = MaterializedLayer(dsk)
        graph = HighLevelGraph.from_collections(name, layer, dependencies=[x])

        return da.Array(graph, name, chunks, meta=x)


class ShardID(NamedTuple):
    """Unique identifier of an individual shard within an array rechunk

    When rechunking a 1d-array with two chunks into a 1d-array with a single chunk
    >>> old = ((2, 2),)  # doctest: +SKIP
    >>> new = ((4),)  # doctest: +SKIP
    >>> rechunk_slicing(old, new)  # doctest: +SKIP
    {
        # The first chunk of the old array belongs to the first
        # chunk of the new array at the first sub-index
        (0,): [(ShardID((0,), (0,)), (slice(0, 2, None),))],

        # The second chunk of the old array belongs to the first
        # chunk of the new array at the second sub-index
        (1,): [(ShardID((0,), (1,)), (slice(0, 2, None),))],
    }
    """

    #: Index of the new output chunk to which the shard belongs
    chunk_index: NIndex
    #: Index of the shard within the multi-dimensional array of shards that will be
    # concatenated into the new chunk
    shard_index: NIndex


SplitChunk: TypeAlias = list[tuple[int, int, slice]]
SplitAxis: TypeAlias = list[SplitChunk]
SplitAxes: TypeAlias = list[SplitAxis]


def split_axes(old: ChunkedAxes, new: ChunkedAxes) -> SplitAxes:
    from dask.array.rechunk import old_to_new

    _old_to_new = old_to_new(old, new)

    axes = []
    for axis_id, new_axis in enumerate(_old_to_new):
        old_axis: SplitAxis = [[] for _ in old[axis_id]]
        for new_chunk_id, new_chunk in enumerate(new_axis):
            for new_subchunk_id, (old_chunk_id, slice) in enumerate(new_chunk):
                old_axis[old_chunk_id].append((new_chunk_id, new_subchunk_id, slice))
        for old_chunk in old_axis:
            if len(old_chunk) == 1:
                continue
            old_chunk.sort(key=lambda subchunk: subchunk[2].start)
        axes.append(old_axis)
    return axes
