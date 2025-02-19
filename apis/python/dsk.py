#!/usr/bin/env python

import json
import re
from concurrent.futures import ThreadPoolExecutor
from os import makedirs, remove, cpu_count
from os.path import join, exists, dirname, splitext
from sys import stdout

import dask
from click import option, group, argument
import pyarrow as pa
from humanfriendly import parse_size
from humanize import naturalsize
from pyarrow import feather
import numpy as np
from scipy.sparse import csr_matrix, vstack
from somacore import AxisQuery
from tiledbsoma import Experiment, SparseNDArray, SOMATileDBContext
from tiledbsoma._fastercsx import CompressedMatrix
from tiledbsoma._query import load_daskarray
from tiledbsoma._indexer import IntIndexer

from utz import err, Time
from utz.mem import Tracker


@group
def cli():
    pass


CHUNK_SIZE_RGX = re.compile(r'(?P<k>\d+)k')


def parse_chunk_size(ctx, param, value):
    if m := CHUNK_SIZE_RGX.fullmatch(value):
        return int(m['k']) * 1000
    else:
        return int(value)


CENSUS = "s3://cellxgene-census-public-us-west-2/cell-census/2024-07-01/soma/census_data/homo_sapiens"
mem_budget_opt = option('-b', '--mem-total-budget', callback=lambda ctx, param, val: parse_size(val) if val else None)
chunk_size_opt = option('-c', '--chunk-size', callback=parse_chunk_size)
no_keep_memray_bin_opt = option('-K', '--no-keep-memray-bin', is_flag=True)
memray_bin_opt = option('-m', '--memray-bin-path')
method_opt = option('-M', '--method', count=True, help='0x: "naive" reindexing+CSR-construction, 1x: .blockwise().scipy, 2x: IntIndexer/fastercsx')
no_native_traces_opt = option('-N', '--no-native-traces', is_flag=True)
out_dir_opt = option('-o', '--out-dir')
tissue_opt = option('-t', '--tissue', default='nose')
tdb_workers_opt = option('-T', '--tdb-workers', type=int, default=1)

# ```bash
# dsk.py joinids -t nose nose  # Write joinids to `nose/{obs,var}.feather`
# dsk.py csr -c 10000 nose     # Load `nose/{obs,var}.feather`, fetch first 10k obs rows (and all vars), build a CSR chunk, print shape+nnz
# ```


@cli.command
@tissue_opt
@argument('joinids_dir')
def joinids(tissue, joinids_dir):
    makedirs(joinids_dir, exist_ok=True)
    obs_path = join(joinids_dir, "obs.feather")
    var_path = join(joinids_dir, "var.feather")
    with Experiment.open(CENSUS, 'r') as exp:
        query = exp.axis_query(
            measurement_name="RNA",
            obs_query=AxisQuery(value_filter=f'is_primary_data == True and tissue_general == "{tissue}"'),
        )
        obs_joinids = query.obs_joinids()
        obs_tbl = pa.Table.from_arrays([obs_joinids], names=['obs_joinids'])
        feather.write_feather(obs_tbl, obs_path)
        var_joinids = query.var_joinids()
        var_tbl = pa.Table.from_arrays([var_joinids], names=['var_joinids'])
        feather.write_feather(var_tbl, var_path)


@cli.command
@mem_budget_opt
@chunk_size_opt
@no_keep_memray_bin_opt
@memray_bin_opt
@method_opt
@no_native_traces_opt
@out_dir_opt
@tdb_workers_opt
@argument('joinids_dir')
def csr(
    mem_total_budget: str | None,
    chunk_size: int | None,
    no_keep_memray_bin: bool,
    memray_bin_path: str | None,
    method: int,
    no_native_traces: bool,
    out_dir: str | None,
    tdb_workers: int,
    joinids_dir: str,
):
    if memray_bin_path is None:
        if chunk_size % 1000 == 0:
            name = f'{chunk_size // 1000}k'
        else:
            name = f'{chunk_size}'
        if tdb_workers != 1:
            name += f'_{tdb_workers if tdb_workers else cpu_count()}'
        memray_bin_path = join(out_dir or joinids_dir, f'{name}.bin')
        err(f"memray logging to {memray_bin_path}")

    native_traces = not no_native_traces
    if exists(memray_bin_path):
        err(f"Removing existing {memray_bin_path}")
        remove(memray_bin_path)

    makedirs(dirname(memray_bin_path), exist_ok=True)
    time = Time()
    with (mem := Tracker(memray_bin_path, native_traces=native_traces, keep=not no_keep_memray_bin)):
        time("start")
        obs_joinids_path = join(joinids_dir, "obs.feather")
        var_joinids_path = join(joinids_dir, "var.feather")
        obs_tbl = feather.read_table(obs_joinids_path)
        all_obs_joinids = obs_tbl['obs_joinids']
        obs_joinids = all_obs_joinids[:chunk_size].to_numpy().tolist()
        var_tbl = feather.read_table(var_joinids_path)
        var_joinids = var_tbl['var_joinids'].to_numpy().tolist()
        obs_joinid_idx_map = {obs_joinid: idx for idx, obs_joinid in enumerate(sorted(obs_joinids))}
        var_joinid_idx_map = {var_joinid: idx for idx, var_joinid in enumerate(sorted(var_joinids))}
        shape = (len(obs_joinids), len(var_joinids))
        soma_ctx = SOMATileDBContext(
            tiledb_config={
                "vfs.s3.no_sign_request": "true",
                "vfs.s3.region": "us-west-2",
                "sm.io_concurrency_level": tdb_workers,
                "sm.compute_concurrency_level": tdb_workers,
                **({
                       "sm.mem.total_budget": mem_total_budget,
                       # "sm.memory_budget": mem_total_budget,
                       # "sm.memory_budget_var": mem_total_budget,
                   } if mem_total_budget else {}),
            },
            threadpool=ThreadPoolExecutor(max_workers=tdb_workers)
        )
        uri = f"{CENSUS}/ms/RNA/X/raw"
        time("open")
        with SparseNDArray.open(uri, context=soma_ctx) as arr:
            time("read")
            if method == 0:
                tbl = arr.read((obs_joinids, var_joinids)).tables().concat()
                soma_dim_0, soma_dim_1, data = [col.to_numpy() for col in tbl.columns]
                time("maps")
                obs = [obs_joinid_idx_map[int(obs_joinid)] for obs_joinid in soma_dim_0]
                var = [var_joinid_idx_map[int(var_joinid)] for var_joinid in soma_dim_1]
                time("csr")
                csr = csr_matrix((data, (obs, var)), shape=shape)
                time()
            elif method == 1:
                scipy = arr.read((obs_joinids, var_joinids)).blockwise(0).scipy()
                time("csrs")
                csrs, idxs = zip(*list(iter(scipy)))
                time("close")
                time("csr")
                csr = vstack(csrs)
                time()
                if len(csrs) > 1:
                    for i, c in enumerate(csrs):
                        err(f"CSR block {i}: {repr(c)}")
            elif method == 2:
                tbl = arr.read((obs_joinids, var_joinids)).tables().concat()
                time("indexers")
                obs_indexer = IntIndexer(obs_joinids, context=soma_ctx)  # TODO: i64 only
                var_indexer = IntIndexer(var_joinids, context=soma_ctx)
                time("indexing")
                new_dim0 = obs_indexer.get_indexer(tbl['soma_dim_0'])
                new_dim1 = var_indexer.get_indexer(tbl['soma_dim_1'])
                time("data")
                data = tbl['soma_data'].to_numpy()
                time("tbl")
                new_tbl = pa.Table.from_pydict({
                    'soma_dim_0': new_dim0,
                    'soma_dim_1': new_dim1,
                    'soma_data': data,
                })
                time("csr")
                cm = CompressedMatrix.from_soma(
                    new_tbl,
                    shape=shape,
                    format='csr',
                    make_sorted=True,
                    context=soma_ctx,
                )
                time("csr")
                csr = cm.to_scipy()
                time()
            else:
                raise ValueError(f"Unrecognized -M/--method count: {method}")

        nnz = csr.nnz
        err(f"CSR: {csr.shape}, {csr.nnz}")

    peak = mem.peak_mem
    err(f"Peak memory use: {peak} ({naturalsize(peak, binary=True, format="%.3g")}); {peak / nnz:.3g} bytes/nz")
    total_measured_time = sum(time.times.values())
    err(f"Total measured time: {total_measured_time:.4g}")
    out_json_path = f'{splitext(memray_bin_path)[0]}.json'
    stats = dict(
        shape=shape,
        nnz=nnz,
        peak_mem=peak,
        times=time.times,
        total_time=total_measured_time,
        bytes_per_nz=peak / nnz,
    )
    json.dump(stats, stdout, indent=2)
    print()
    with open(out_json_path, 'w') as f:
        json.dump(stats, f, indent=2)


@cli.command
@chunk_size_opt
@tissue_opt
def block(
    chunk_size,
    tissue,
):
    with (
        Experiment.open(CENSUS, 'r') as exp,
        dask.config.set(scheduler="single-threaded"),
    ):
        query = exp.axis_query(
            measurement_name="RNA",
            obs_query=AxisQuery(value_filter=f'is_primary_data == True and tissue_general == "{tissue}"'),
        )
        obs_joinids = query.obs_joinids()
        X = exp.ms["RNA"].X["raw"]
        x = load_daskarray(
            layer=X,
            chunk_size=chunk_size,
            obs_joinids=obs_joinids,
        )
        # Compute just the first chunk
        block = x.blocks[0, 0].compute()
        print(block.shape)
        print(block.nnz)


if __name__ == "__main__":
    cli()
