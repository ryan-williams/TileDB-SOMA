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
from os import makedirs
from os.path import join

import dask
from click import option, group, argument
import pyarrow as pa
from pyarrow import feather
from scipy.sparse import csr_matrix
from somacore import AxisQuery
from tiledbsoma import Experiment, SparseNDArray, SOMATileDBContext
from tiledbsoma._query import load_daskarray


@group
def cli():
    pass


CENSUS = "s3://cellxgene-census-public-us-west-2/cell-census/2024-07-01/soma/census_data/homo_sapiens"
chunk_size_opt = option('-c', '--chunk-size', type=int)
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
@argument('joinids_dir')
def csr(
    chunk_size,
    joinids_dir,
):
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
    print(f"{csr.shape}, {csr.nnz}")


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
