"""Read verified September hyperslabs from local or HTTP range-addressable CMIP files."""
from collections import OrderedDict
from contextlib import contextmanager
from functools import lru_cache
import io
import itertools
from pathlib import Path
import time

import cftime
import h5py
import numpy as np
import pandas as pd
import os
import requests

from .common import MODEL, TABLES, ROOT, atomic


@lru_cache(maxsize=1)
def session():
    return requests.Session()


class RangeFile(io.RawIOBase):
    """Bound HTTP reads, checking byte ranges. Nearby HDF chunks share a request."""
    def __init__(self, url, size):
        self.url, self.size = url, size
        self.pos = self.bytes_read = self.requests = 0
        self.blocks = OrderedDict()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = (0, self.pos, self.size)[whence] + offset
        return self.pos

    def fetch(self, start, stop):
        if start >= stop:
            return b""
        for key, data in reversed(self.blocks.items()):
            if key <= start and stop <= key + len(data):
                return data[start-key:stop-key]
        for retry in range(4):
            try:
                with session().get(self.url, headers={"Range":f"bytes={start}-{stop-1}", "Accept-Encoding":"identity"},
                                   timeout=(12,70), stream=True) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers["Content-Range"].split("/")[0] != f"bytes {start}-{stop-1}":
                        raise OSError(f"Expected HTTP byte range from {self.url}")
                    data = response.content
                    if len(data)!=stop-start:
                        raise OSError("Short byte range")
                break
            except (requests.RequestException, OSError):
                if retry==3:
                    raise
                time.sleep(2**retry)
        self.requests += 1
        self.bytes_read += len(data)
        self.blocks[start] = data
        while sum(map(len,self.blocks.values())) > 96*1024**2 and len(self.blocks)>1:
            self.blocks.popitem(last=False)
        return data

    def readinto(self, buffer):
        n = min(len(buffer),self.size-self.pos)
        if n <= 0:
            return 0
        pos = self.pos
        for base,data in reversed(self.blocks.items()):
            if base <= pos and pos+n <= base+len(data):
                buffer[:n] = memoryview(data)[pos-base:pos-base+n]
                self.pos += n
                return n
        data = self.fetch(pos, min(self.size,pos+max(n,65536)))
        buffer[:n] = memoryview(data)[:n]
        self.pos += n
        return n

    def preload(self, ds, start, stop):
        if ds.chunks is None:
            return
        axes = [range(start//ds.chunks[0]*ds.chunks[0],stop,ds.chunks[0])]
        axes.extend(range(0,n,c) for n,c in zip(ds.shape[1:],ds.chunks[1:]))
        spans = []
        for origin in itertools.product(*axes):
            info = ds.id.get_chunk_info_by_coord(origin)
            if info.byte_offset is not None:
                spans.append((info.byte_offset,info.byte_offset+info.size))
        merged = []
        for lo,hi in sorted(spans):
            if merged and lo-merged[-1][1]<131072 and hi-merged[-1][0]<32*1024**2:
                merged[-1][1] = hi
            else:
                merged.append([lo,hi])
        for lo,hi in merged:
            self.fetch(lo,hi)


def attr(obj, key, default=None):
    val = obj.attrs.get(key,default)
    return val.decode() if isinstance(val,bytes) else val


def array(ds, index=slice(None)):
    raw = ds[index]
    result = np.asarray(raw, dtype=np.float64)
    for key in ("_FillValue","missing_value"):
        missing = ds.attrs.get(key)
        if missing is not None:
            result[result==np.ravel(missing)[0]] = np.nan
    return result * float(np.asarray(ds.attrs.get("scale_factor",1)).item()) + float(np.asarray(ds.attrs.get("add_offset",0)).item())


@lru_cache(maxsize=2)
def catalog(path, timestamp):
    frame = pd.read_parquet(path)
    return {k:list(group.itertuples(index=False)) for k,group in frame.groupby(["member","experiment","variable"])}


def resolve(root, member, exp, year):
    path = root/"inventory.parquet"
    index = catalog(path,path.stat().st_mtime_ns)
    selected = {}
    for var,table in TABLES.items():
        start,end = (f"{year}0901",f"{year}0930") if table == "day" else (f"{year}09",f"{year}09")
        rows = [r for r in index[member,exp,var] if r.first <= start and r.last >= end]
        if not rows:
            raise ValueError(f"Missing {member} {exp} {var} {year}")
        selected[var] = max(rows,key=lambda r:r.version)
    return selected


def extract_month(h, remote, row, year):
    var = row.variable
    for key,expected in [("source_id",MODEL),("experiment_id",row.experiment),("variant_label",row.member),("grid_label","gn")]:
        if attr(h,key)!=expected:
            raise ValueError(f"{key} mismatch for {row.esgf}")
    if row.experiment != "historical" and attr(h,"parent_variant_label")!=row.member:
        raise ValueError("Scenario lineage mismatch")
    ds = h[var]
    time_ds = h["time"]
    units, calendar = attr(time_ds,"units"),attr(time_ds,"calendar","standard")
    begin = cftime.datetime(year,9,1,calendar=calendar)
    finish = cftime.datetime(year,10,1,calendar=calendar)
    a,b = cftime.date2num([begin,finish],units,calendar=calendar)
    # HDF dimension scales can be split across hundreds of chunks interleaved
    # with physical data. Cache the checked, complete coordinate per immutable file.
    clock_path = ROOT / "runs/.coordinates" / (row.checksum + ".npy")
    if clock_path.is_file():
        t = np.load(clock_path, allow_pickle=False)
    else:
        t = time_ds[:]
        if len(np.unique(t)) != len(t) or not np.isfinite(t).all():
            raise ValueError("Duplicate or missing CMIP time coordinate")
        with atomic(clock_path) as tmp:
            np.save(tmp,t)
    indices = np.flatnonzero((t>=a)&(t<b))
    n = 30 if var in ("ua","va") else 1
    if len(indices)!=n or (n>1 and np.any(np.diff(indices)!=1)):
        raise ValueError(f"Incomplete September: {row.member} {year} {var}")
    left,right = int(indices[0]),int(indices[-1]+1)
    centers = (t[indices]-a)/(b-a)*30
    bounds = h[attr(time_ds,"bounds")][left:right]
    expected = np.column_stack([np.arange(n),np.arange(1,n+1)])*30/n
    if not np.allclose((bounds-a)/(b-a)*30,expected,atol=1e-5) or not np.allclose(centers,expected.mean(axis=1)):
        raise ValueError(f"Calendar or interval mismatch: {var} {year}")
    if "time: mean" not in attr(ds,"cell_methods",""):
        raise ValueError(f"Expected time means: {var}")
    if remote:
        remote.preload(ds,left,right)
    values = array(ds,slice(left,right))
    coords = {k:array(h[k]) for k in ("plev","lev","lat","lon","latitude","longitude") if k in h}
    coords["units"] = {k:attr(h[k],"units") for k in coords}
    if var=="tas":
        coords["lat_bnds"]=h["lat_bnds"][:]
    return dict(values=values if n>1 else values[0],coords=coords,units=attr(ds,"units"),
                standard_name=attr(ds,"standard_name"),centers=centers,calendar=calendar,positive=attr(h["lev"],"positive") if "lev" in h else None)


def read_month(row, year):
    errors = []
    for url in dict.fromkeys([row.local,row.aws,row.esgf]):
        if not url:
            continue
        remote = None
        try:
            if url.startswith("https://"):
                remote = RangeFile(url,row.bytes)
            with h5py.File(remote if remote is not None else url,"r") as h:
                result = extract_month(h,remote,row,year)
            info = {"url":url,"sha256" if row.checksum_type.upper()=="SHA256" else "checksum":row.checksum,
                    "official_file_bytes":row.bytes,"version":row.version,"grid":row.grid,
                    "http_bytes":remote.bytes_read if remote else 0, "http_requests":remote.requests if remote else 0,
                    "local_hyperslab_bytes":0 if remote else result["values"].nbytes//2,
                    "replica_errors":errors}
            return result,info
        except (OSError, KeyError, ValueError, requests.RequestException) as exc:
            errors.append(f"{url}: {exc}")
        finally:
            if remote is not None:
                remote.close()
    raise OSError("No readable replica: " + " | ".join(errors))
