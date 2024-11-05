# Copyright (c) 2021-2023 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2023 TileDB, Inc.
#
# Licensed under the MIT License.

"""Implementation of a SOMA Experiment.
"""
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import scipy.sparse as sp
import somacore
from anndata import AnnData
from somacore import DenseNDArray, query
from somacore.query import _fast_csr
from somacore.query.query import (
    AxisColumnNames,
    _Axis,
    _AxisQueryResult,
    _Experimentish,
)

from ._sparse_nd_array import SparseNDArray
from .options import SOMATileDBContext, TileDBConfig

_Exp = TypeVar("_Exp", bound="_Experimentish")
"""TypeVar for the concrete type of an experiment-like object."""


def chunk_size_list(dim_size: int, chunk_size: int) -> Tuple[int, ...]:
    """Go from single integer to list representation of a chunking scheme.

    Some rules about how this behaves:

        n_full_chunks, remainder := divmod(dim_size, chunk_size)
        (*((chunk_size,) * n_full_chunks), remainder) := chunk_size_list(dim_size, chunk_size)
        map(len, batched(range(dim_size), chunk_size))) := chunk_size_list(dim_size, chunk_size)


    Examples
    --------
    >>> chunk_size_list(25, 10)
    [10, 10, 5]
    >>> chunk_size_list(9, 3)
    [3, 3, 3]
    """
    n_full_chunks, remainder = divmod(dim_size, chunk_size)
    chunk_size_list = [chunk_size] * n_full_chunks
    if remainder:
        chunk_size_list += [remainder]
    return tuple(chunk_size_list)


def sparse_chunk(
    block_info: Dict[None, Any],
    uri: str,
    var_joinids: pa.IntegerArray,
    tiledb_config: TileDBConfig,
) -> sp.csr_matrix:
    shape = block_info[None]["chunk-shape"]
    array_location = block_info[None]["array-location"]
    (obs_start, obs_end), _ = array_location
    obs_slice = slice(obs_start, obs_end - 1)
    soma_ctx = SOMATileDBContext(tiledb_config=tiledb_config)
    with SparseNDArray.open(uri, context=soma_ctx) as arr:
        tbl = arr.read((obs_slice, var_joinids)).tables().concat()
    soma_dim_0, soma_dim_1, count = [col.to_numpy() for col in tbl.columns]
    soma_dim_0 = soma_dim_0 - obs_start
    var_joinid_idx_map = {
        var_joinid: idx
        for idx, var_joinid in enumerate(var_joinids.to_numpy().tolist())
    }
    vars = [var_joinid_idx_map[int(var_joinid)] for var_joinid in soma_dim_1]
    return sp.csr_matrix((count, (soma_dim_0, vars)), shape=shape)


def slice_chunk(
    X_chunk: sp.csr_matrix,
    obs_joinids: npt.NDArray[np.int64],
    block_info: Dict[Optional[int], Any],
) -> sp.csr_matrix:
    array_location = block_info[0]["array-location"]
    (obs_start, _), _ = array_location
    return X_chunk[np.array(obs_joinids[0, 0]) - obs_start, :]


if TYPE_CHECKING:
    import dask.array as da


def load_daskarray(
    layer: Union[SparseNDArray, DenseNDArray],
    chunk_size: int,
    obs_joinids: Optional[pa.IntegerArray] = None,
    var_joinids: Optional[pa.IntegerArray] = None,
) -> "da.Array":
    """Load a TileDB-SOMA X layer as a Dask array, using ``tiledb`` or ``tiledbsoma``."""
    import dask.array as da

    with layer:
        _, _, data_dtype = layer.schema.types
        dtype = data_dtype.to_pandas_dtype()
        nobs, nvars = layer.shape

    chunk_sizes = chunk_size_list(nobs, chunk_size)
    var_chunk_size = len(var_joinids) if var_joinids else nvars
    if isinstance(layer, somacore.SparseNDArray):
        X = da.map_blocks(
            sparse_chunk,
            chunks=(chunk_sizes, (var_chunk_size,)),
            meta=sp.csr_matrix((0, 0), dtype=dtype),
            uri=layer.uri,
            var_joinids=var_joinids,
            tiledb_config=layer.context.tiledb_config,
        )
        if obs_joinids:
            obs_chunk_joinids: List[List[int]] = []
            chunk_idx = 0
            chunk_joinids: List[int] = []
            chunk_end = chunk_sizes[chunk_idx]
            for joinid in obs_joinids:
                joinid = joinid.as_py()
                if joinid >= chunk_end:
                    obs_chunk_joinids.append(chunk_joinids)
                    chunk_idx += 1
                    chunk_end += chunk_sizes[chunk_idx]
                    chunk_joinids = [joinid]
                else:
                    chunk_joinids.append(joinid)
            if chunk_joinids:
                obs_chunk_joinids.append(chunk_joinids)

            obs_chunk_sizes: Tuple[int, ...] = tuple(
                len(chunk_joinids) for chunk_joinids in obs_chunk_joinids
            )
            num_obs_chunks = len(obs_chunk_sizes)
            arr = np.empty((num_obs_chunks, 1), dtype=object)
            for idx, chunk_joinids in enumerate(obs_chunk_joinids):
                arr[idx, 0] = chunk_joinids
            obs_joinid_arr = da.from_array(arr, chunks=((1,) * num_obs_chunks, (1,)))
            X = da.map_blocks(
                slice_chunk,
                X,
                obs_joinid_arr,
                chunks=(obs_chunk_sizes, (var_chunk_size,)),
                meta=sp.csr_matrix((0, 0), dtype=dtype),
            )
    else:
        raise ValueError(f"Can't dask-load DenseNDArray: {layer.uri}")
    return X


class ExperimentAxisQuery(query.ExperimentAxisQuery[_Exp]):

    def to_anndata(
        self,
        X_name: str,
        *,
        column_names: Optional[AxisColumnNames] = None,
        X_layers: Sequence[str] = (),
        obsm_layers: Sequence[str] = (),
        obsp_layers: Sequence[str] = (),
        varm_layers: Sequence[str] = (),
        varp_layers: Sequence[str] = (),
        drop_levels: bool = False,
        dask_chunk_size: Optional[int] = None,
    ) -> AnnData:
        ad = self._read(
            X_name,
            column_names=column_names or AxisColumnNames(obs=None, var=None),
            X_layers=X_layers,
            obsm_layers=obsm_layers,
            obsp_layers=obsp_layers,
            varm_layers=varm_layers,
            varp_layers=varp_layers,
            dask_chunk_size=dask_chunk_size,
        ).to_anndata()

        # Drop unused categories on axis dataframes if requested
        if drop_levels:
            for name in ad.obs:
                if ad.obs[name].dtype.name == "category":
                    ad.obs[name] = ad.obs[name].cat.remove_unused_categories()
            for name in ad.var:
                if ad.var[name].dtype.name == "category":
                    ad.var[name] = ad.var[name].cat.remove_unused_categories()

        return ad

    def _read(
        self,
        X_name: str,
        *,
        column_names: AxisColumnNames,
        X_layers: Sequence[str],
        obsm_layers: Sequence[str] = (),
        obsp_layers: Sequence[str] = (),
        varm_layers: Sequence[str] = (),
        varp_layers: Sequence[str] = (),
        dask_chunk_size: Optional[int] = None,
    ) -> "_AxisQueryResult":
        """Reads the entire query result in memory.

        This is a low-level routine intended to be used by loaders for other
        in-core formats, such as AnnData, which can be created from the
        resulting objects.

        Args:
            X_name: The X layer to read and return in the ``X`` slot.
            column_names: The columns in the ``var`` and ``obs`` dataframes
                to read.
            X_layers: Additional X layers to read and return
                in the ``layers`` slot.
            obsm_layers:
                Additional obsm layers to read and return in the obsm slot.
            obsp_layers:
                Additional obsp layers to read and return in the obsp slot.
            varm_layers:
                Additional varm layers to read and return in the varm slot.
            varp_layers:
                Additional varp layers to read and return in the varp slot.
        """
        x_collection = self._ms.X
        all_x_names = [X_name] + list(X_layers)
        all_x_arrays: Dict[str, SparseNDArray] = {}
        for _xname in all_x_names:
            if not isinstance(_xname, str) or not _xname:
                raise ValueError("X layer names must be specified as a string.")
            if _xname not in x_collection:
                raise ValueError("Unknown X layer name")
            x_array = x_collection[_xname]
            if not isinstance(x_array, SparseNDArray):
                raise NotImplementedError("Dense array unsupported")
            all_x_arrays[_xname] = x_array

        def _read_axis_mappings(
            fn: Callable[[_Axis, str], npt.NDArray[Any]],
            axis: _Axis,
            keys: Sequence[str],
        ) -> Dict[str, npt.NDArray[Any]]:
            return {key: fn(axis, key) for key in keys}

        obsm_ft = self._threadpool.submit(
            _read_axis_mappings, self._axism_inner_ndarray, _Axis.OBS, obsm_layers
        )
        obsp_ft = self._threadpool.submit(
            _read_axis_mappings, self._axisp_inner_ndarray, _Axis.OBS, obsp_layers
        )
        varm_ft = self._threadpool.submit(
            _read_axis_mappings, self._axism_inner_ndarray, _Axis.VAR, varm_layers
        )
        varp_ft = self._threadpool.submit(
            _read_axis_mappings, self._axisp_inner_ndarray, _Axis.VAR, varp_layers
        )

        obs_table, var_table = self._read_both_axes(column_names)

        obs_joinids = self.obs_joinids()
        var_joinids = self.var_joinids()

        x_matrices = {
            _xname: (
                _fast_csr.read_csr(
                    layer,
                    obs_joinids,
                    var_joinids,
                    index_factory=self._index_factory,
                ).to_scipy()
                if not dask_chunk_size
                else load_daskarray(
                    layer=layer,
                    chunk_size=dask_chunk_size,
                    obs_joinids=obs_joinids,
                    var_joinids=var_joinids,
                )
            )
            for _xname, layer in all_x_arrays.items()
        }

        x = x_matrices.pop(X_name)

        obs = obs_table.to_pandas()
        obs.index = obs.index.astype(str)

        var = var_table.to_pandas()
        var.index = var.index.astype(str)

        return _AxisQueryResult(
            obs=obs,
            var=var,
            X=x,
            obsm=obsm_ft.result(),
            obsp=obsp_ft.result(),
            varm=varm_ft.result(),
            varp=varp_ft.result(),
            X_layers=x_matrices,
        )
