#!/usr/bin/env python
#
# ```bash
# for k in 5 10 20 40 80; do
#   n=nose-${k}k
#   time mrno $n.bin dask_block.py -c ${k}000
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
#   time memray run --native -o $n.bin dask_block.py -c ${k}000
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


import dask
from click import command, option
from somacore import AxisQuery
from tiledbsoma import Experiment
from tiledbsoma._query import load_daskarray

@command
@option('-c', '--chunk-size', type=int)
@option('-t', '--tissue', default='nose')
def main(
    chunk_size,
    tissue,
):
    uri = "s3://cellxgene-census-public-us-west-2/cell-census/2024-07-01/soma/census_data/homo_sapiens"
    with (
        Experiment.open(uri, 'r') as exp,
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
    main()
