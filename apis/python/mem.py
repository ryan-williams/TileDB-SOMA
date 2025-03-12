#!/usr/bin/env python
import json
from functools import wraps
from os import makedirs
from os.path import join
from sys import stdout

from click import option, command
from humanfriendly import parse_size
from somacore import AxisQuery, AxisColumnNames
from utz import Time, iec
from utz.mem import Tracker

from tiledbsoma import Experiment, SOMATileDBContext, DataFrame
from tiledbsoma._query import AxisName

CENSUS_S3 = "s3://cellxgene-census-public-us-west-2/cell-census"
DEFAULT_CENSUS_VERSION = "2024-07-01"
OBS = AxisName.OBS
VAR = AxisName.VAR
DEFAULT_CONFIG = {
    "vfs.s3.no_sign_request": "true",
    "vfs.s3.region": "us-west-2",
}


def sz(size: int) -> str:
    return f"{size} ({iec(size)})"


def summary_stats(stats):
    obj = {}
    md = stats["metadata"]
    obj["peak_mem"] = md["peak_memory"]

    # Copy some memray stats, include IEC string reprs (e.g. "2.1 GiB")
    top_allocs = [
        {**alloc, "iec": iec(alloc["size"])}
        for alloc in stats["top_allocations_by_size"]
    ]
    obj.update(
        {
            "top_allocs": top_allocs,
            "tracked_allocs": stats["total_num_allocations"],
            "tracked_bytes": stats["total_bytes_allocated"],
            "total_allocs": md["total_allocations"],
            "total_frames": md["total_frames"],
        }
    )
    return obj


def census_version_opt(fn):
    @option("-C", "--census-version", "census_version", default=DEFAULT_CENSUS_VERSION, help=f"Default: {DEFAULT_CENSUS_VERSION}")
    @wraps(fn)
    def _fn(*args, census_version: str, **kwargs):
        exp_uri = f"{CENSUS_S3}/{census_version}/soma/census_data/homo_sapiens"
        return fn(*args, exp_uri=exp_uri, **kwargs)

    return _fn


@command("mem", no_args_is_help=True)
@option("-b", "--mem-total-budget", help='TileDB "sm.mem.total_budget" config value (default: "10GiB")')
@option('-B', '--init-buffer-bytes', help='"soma.init_buffer_bytes" config value')
@option("-c", '--columns', callback=lambda ctx, param, val: None if val is None else val.split(','), help="DataFrame columns to fetch")
@census_version_opt
@option("-K", "--no-keep-memray-bin", is_flag=True, help="Rm memray profile before exiting")
@option("-o", "--out-dir", help="Write Memray profile, stats, and flamegraph to this directory (default: `out/<tissue>`)")
@option("-P", "--no-trace-python-allocators", is_flag=True, help="Don't pass `trace_python_allocators=True` to Memray")
@option("-t", "--tissue", help="Query Census cells with this `tissue_general` value")
@option("-V", '--fetch-vars', is_flag=True, help="Profile fetching `vars` DataFrame (default: `obs`)")
def main(
    mem_total_budget: str | None,
    init_buffer_bytes: str | None,
    columns: list[str] | None,
    exp_uri: str,
    no_keep_memray_bin: bool,
    out_dir: str | None,
    no_trace_python_allocators: bool,
    tissue: str,
    fetch_vars: bool,
):
    """Output feather files containing the obs and var joinids responsive to a CELLxGENE Census query."""
    time = Time(log=True)
    if not out_dir:
        out_dir = f"out/{tissue}"
    makedirs(out_dir, exist_ok=True)
    context = SOMATileDBContext(
        tiledb_config={
            **({"sm.mem.total_budget": parse_size(mem_total_budget)} if mem_total_budget else {}),
            **({"soma.init_buffer_bytes": parse_size(init_buffer_bytes)} if init_buffer_bytes else {}),
            **DEFAULT_CONFIG,
        },
    )
    name = "var" if fetch_vars else "obs"
    if init_buffer_bytes:
        name = f"{name}_{init_buffer_bytes}"
    memray_bin_path = join(out_dir, f"{name}.memray")
    with (mem := Tracker(
        memray_bin_path,
        keep=not no_keep_memray_bin,
        trace_python_allocators=not no_trace_python_allocators,
        log=True,
    )):
        time("open")
        with Experiment.open(exp_uri, "r", context=context) as exp:
            time("query")
            obs_query = AxisQuery(value_filter=f'is_primary_data == True and tissue_general == "{tissue}"')
            var_query = AxisQuery()
            query = exp.axis_query(
                measurement_name="RNA",
                obs_query=obs_query,
                var_query=var_query,
            )

            def run(
                name: str,
                axis: AxisName,
                sdf: DataFrame,
                axis_query: AxisQuery,
                axis_column_names: AxisColumnNames,
            ):
                time(name)
                sdf = query._read_axis_dataframe(
                    axis=axis,
                    axis_df=sdf,
                    axis_query=axis_query,
                    axis_column_names=axis_column_names,
                )
                time(f"{name}-df")
                df = sdf.to_pandas()
                time()
                print(f"{name} {df.shape}: {','.join(df.columns)}")

            if fetch_vars:
                run("var", VAR, query._var_df, var_query, AxisColumnNames(obs=None, var=columns))
            else:
                run("obs", OBS, query._obs_df, obs_query, AxisColumnNames(obs=columns, var=None))

    stats = summary_stats(mem.stats)
    json.dump(
        stats,
        stdout,
        indent=2,
    )
    print()
    print(f"Peak memory use: {sz(mem.peak_mem)}")


if __name__ == "__main__":
    main()
