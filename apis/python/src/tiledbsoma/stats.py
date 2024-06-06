"""
TileDB stats collection utilities.

```python
from tiledbsoma.stats import profile, stats

with profile("open-exp"):
    exp = Experiment.open(uri)

with profile("query"):
    q = exp.query(
        "mtdna",
        obs_query=tiledbsoma.AxisQuery(value_filter="tissue == 'lung'"),
        var_query=tiledbsoma.AxisQuery(coords=(slice(50, 100),)),
    )

with profile("obs-to-pandas"):
    query_obs = q.obs().concat().to_pandas()

# DataFrames of timers and counters, for each profiled block
timers_df, counters_df = stats.dfs
```
"""

import json
from abc import ABC
from contextlib import redirect_stdout, contextmanager
from dataclasses import dataclass, asdict
from os.path import join
from re import fullmatch
from tempfile import TemporaryDirectory
from typing import Union

import pandas as pd

from tiledbsoma import (
    tiledbsoma_stats_enable,
    tiledbsoma_stats_disable,
    tiledbsoma_stats_reset,
    tiledbsoma_stats_dump,
)


@dataclass
class Timer:
    avg: float
    sum: float
    num: float

    @staticmethod
    def from_dict(o: dict[str, int], scale=1e4) -> 'Timer':
        """Parse a ``Timer`` from a ``dict`` (as returned from tiledb{,soma} stats).

        Input ``dict`` should contain keys ``avg`` and ``sum``, and ``sum`` should be an integer multiple of ``avg``
        (with tolerance equal to the reciprocal of the provided ``scale`` param), from which ``num`` is inferred.
        """
        keys = o.keys()
        if set(keys) != {'avg', 'sum'}:
            raise ValueError(f'Unexpected timer kinds: {list(keys)} (expected ["avg","sum"])')
        avg = o['avg']
        tot = o['sum']
        scaled = int(round(tot / avg * scale))
        if scaled % 1 != 0:
            raise ValueError(f"sum {tot} / avg {avg} == {tot / avg}, error ≥ {1/scale}")
        num = int(round(scaled / scale))
        return Timer(avg=avg, sum=tot, num=num)


Timers = dict[str, Timer]
Counters = dict[str, int]


@dataclass
class StatsDataFrames:
    timers_df: pd.DataFrame
    counters_df: pd.DataFrame

    def __iter__(self):
        yield self.timers_df
        yield self.counters_df


@dataclass
class StatsElem:
    timers: Timers
    counters: Counters

    @staticmethod
    def from_dict(stats: list[dict]) -> 'StatsElem':
        if not isinstance(stats, list):
            raise ValueError(f"Expected list: {stats}")
        if len(stats) != 1:
            raise ValueError(f"Expected list with one elem: {stats}")
        stats = stats[0]
        if set(stats.keys()) != {'timers', 'counters'}:
            raise ValueError(f'Unrecognized stats object: {stats}')
        timers = stats['timers']
        timers_dict = {}
        for k, v in timers.items():
            key, kind = k.rsplit('.', 1)
            if kind not in ['avg', 'sum']:
                raise ValueError(f'Unexpected timer key: {k}')
            if key not in timers_dict:
                timers_dict[key] = {}
            timers_dict[key][kind] = v

        timers: Timers = {
            key: Timer.from_dict(timer_dict)
            for key, timer_dict in timers_dict.items()
        }
        return StatsElem(timers=timers, counters=stats['counters'])

    @property
    def timers_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            dict(key=key, **asdict(timer))
            for key, timer in self.timers.items()
        ])

    @property
    def counters_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            dict(key=key, num=num)
            for key, num in self.counters.items()
        ])

    @property
    def dfs(self) -> StatsDataFrames:
        return StatsDataFrames(self.timers_df, self.counters_df)


StatsElems = list[StatsElem]
StatsVal = Union[StatsElem, StatsElems]


class Stats:
    def __init__(self):
        self.stats: dict[str, StatsVal] = {}

    def enable(self) -> None:
        tiledbsoma_stats_enable()

    def disable(self) -> None:
        tiledbsoma_stats_disable()

    def reset(self) -> None:
        tiledbsoma_stats_reset()

    def dump(self) -> None:
        tiledbsoma_stats_dump()

    def header_regexs(self) -> list[str]:
        return [r'libtiledb=\d+\.\d+\.\d+']

    def get(self, reset: bool = False) -> list[dict]:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = join(tmp_dir, 'stats.txt')
            with open(tmp_path, 'w') as f:
                with redirect_stdout(f):
                    self.dump()
            with open(tmp_path, 'r') as f:
                lines = [ line.rstrip('\n') for line in f.readlines() ]
                header_regexs = self.header_regexs()
                for idx, header_regex in enumerate(header_regexs):
                    line = lines[idx]
                    if not fullmatch(header_regex, line):
                        raise RuntimeError(f"Expected line {idx+1} to match {header_regex}: {line}")
                stats = json.loads('\n'.join(lines[len(header_regexs):]))
            if reset:
                self.reset()
        return stats

    @contextmanager
    def collect(self, name: str, append: bool = False):
        stats = self.stats
        if name in stats:
            if not append:
                raise ValueError(f'Name {name} already exists in stats obj')
        else:
            if append:
                stats[name] = []
        self.reset()
        self.enable()
        try:
            yield
        finally:
            self.disable()
            cur_stats = self.get()
            if cur_stats:
                stats_elem = StatsElem.from_dict(cur_stats)
                if append:
                    stats[name].append(stats_elem)
                else:
                    stats[name] = stats_elem

    @property
    def dfs(self) -> StatsDataFrames:
        timer_dfs = []
        counter_dfs = []
        for name, stats_val in self.stats.items():
            stats_elems = stats_val if isinstance(stats_val, list) else [ stats_val ]
            for stats_elem in stats_elems:
                timers_df, counters_df = stats_elem.dfs
                timer_dfs.append(timers_df.assign(name=name))
                counter_dfs.append(counters_df.assign(name=name))

        timers_df = pd.concat(timer_dfs).reset_index(drop=True)
        counters_df = pd.concat(counter_dfs).reset_index(drop=True)
        return StatsDataFrames(timers_df=timers_df, counters_df=counters_df)


stats = Stats()
profile = stats.collect
