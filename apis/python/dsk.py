#!/usr/bin/env python
#
# ```bash
# for k in 5 10 20 40 80; do
#   n=nose-${k}k
#   time mrno $n.bin dsk.py block -c ${k}000
#   mfo $n.{html,bin}
#   msjo $n.{json,bin}
#   echo -n "Peak memory: "
#   jr .metadata.peak_memory ${n}.json | thr
# done
# ```
#
# ```bash
# for k in 5 10 20 40 80; do
#   n=nose-${k}k
#   time memray run --native -o $n.bin dsk.py block -c ${k}000
#   memray flamegraph -o $n.{html,bin}
#   memray stats --json -o $n.{json,bin}
#   echo -n "Peak memory: "
#   jq -r .metadata.peak_memory ${n}.json | thr
# done
# ```
#
# ```python
# nnz = {
#      5: 14670255,
#     10: 24588046,
#     20: 46447704,
#     30: 68203751,
#     40: 92832399,
#     60: 137194577,
#     80: 179431081,
#    100: 210581850,
# }
# peaks = { n: json.load(open(f"nose-{n}k.json", 'r'))['metadata']['peak_memory'] for n, nz in nnz.items() }
# { n: peaks[n] / nz for n, nz in nnz.items() }
# ```
import json
import re
from os import makedirs, remove
from os.path import join, exists, splitext

import dask
from click import option, group, argument
import pyarrow as pa
from humanize import naturalsize
from pyarrow import feather
from scipy.sparse import csr_matrix
from somacore import AxisQuery
from tiledbsoma import Experiment, SparseNDArray, SOMATileDBContext
from tiledbsoma._query import load_daskarray

import memray
from utz import err, run


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
chunk_size_opt = option('-c', '--chunk-size', callback=parse_chunk_size)
overwrite_bin_opt = option('-f', '--overwrite-bin', is_flag=True)
no_overwrite_memray_stats_opt = option('-F', '--no-overwrite-memray-stats', is_flag=True)
memray_bin_opt = option('-m', '--memray-bin-path')
no_native_traces_opt = option('-N', '--no-native-traces', is_flag=True)
tissue_opt = option('-t', '--tissue', default='nose')

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
@chunk_size_opt
@overwrite_bin_opt
@no_overwrite_memray_stats_opt
@memray_bin_opt
@no_native_traces_opt
@argument('joinids_dir')
def csr(
    chunk_size,
    overwrite_bin,
    no_overwrite_memray_stats,
    memray_bin_path,
    no_native_traces,
    joinids_dir,
):
    if memray_bin_path is None:
        if chunk_size % 1000 == 0:
            name = f'{chunk_size // 1000}k'
        else:
            name = f'{chunk_size}'
        memray_bin_path = join(joinids_dir, f'{name}.bin')
        err(f"memray logging to {memray_bin_path}")

    regen_bin = True
    native_traces = not no_native_traces
    if exists(memray_bin_path):
        if overwrite_bin:
            err(f"Removing existing {memray_bin_path}")
            remove(memray_bin_path)
        else:
            regen_bin = False
            err(f"{memray_bin_path} exists, skipping `memray run` (use `-f` to overwrite)")

    if regen_bin:
        with memray.Tracker(memray_bin_path, native_traces=native_traces):
            obs_joinids_path = join(joinids_dir, "obs.feather")
            var_joinids_path = join(joinids_dir, "var.feather")
            obs_tbl = feather.read_table(obs_joinids_path)
            all_obs_joinids = obs_tbl['obs_joinids']
            obs_joinids = all_obs_joinids[:chunk_size].to_numpy().tolist()
            var_tbl = feather.read_table(var_joinids_path)
            var_joinids = var_tbl['var_joinids'].to_numpy().tolist()
            shape = (len(obs_joinids), len(var_joinids))
            tiledb_config = {
                "vfs.s3.no_sign_request": "true",
                "vfs.s3.region": "us-west-2",
            }
            soma_ctx = SOMATileDBContext(tiledb_config=tiledb_config)
            uri = f"{CENSUS}/ms/RNA/X/raw"
            with SparseNDArray.open(uri, context=soma_ctx) as arr:
                tbl = arr.read((obs_joinids, var_joinids)).tables().concat()
            soma_dim_0, soma_dim_1, data = [col.to_numpy() for col in tbl.columns]
            obs_joinid_idx_map = {obs_joinid: idx for idx, obs_joinid in enumerate(obs_joinids)}
            obs = [obs_joinid_idx_map[int(obs_joinid)] for obs_joinid in soma_dim_0]
            var_joinid_idx_map = {var_joinid: idx for idx, var_joinid in enumerate(var_joinids)}
            var = [var_joinid_idx_map[int(var_joinid)] for var_joinid in soma_dim_1]
            csr = csr_matrix((data, (obs, var)), shape=shape)
            err(f"CSR: {csr.shape}, {csr.nnz}")

    name, _ = splitext(memray_bin_path)
    memray_json_path = f"{name}.stats.json"
    run(
        'memray', 'stats', '--json',
        *([] if no_overwrite_memray_stats else ['-f']),
        '-o', memray_json_path,
        memray_bin_path)
    with open(memray_json_path, 'r') as f:
        stats = json.load(f)
    peak = stats['metadata']['peak_memory']
    err(f"Peak memory use: {peak} ({naturalsize(peak)})")
    out_json_path = f'{name}.json'
    with open(out_json_path, 'w') as f:
        json.dump(
            dict(
                shape=csr.shape,
                nnz=csr.nnz,
                peak_mem=peak,
            ),
            f,
            indent=2,
        )


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
