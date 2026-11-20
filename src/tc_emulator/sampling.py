"""Independent background/weather streams and one event contract for all driving modes."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .common import COMPONENTS, atomic, fingerprint_rng, sha
from .physical import thermodynamics
from .tc_adapter import DAY

UNITS={"sst":"K","mld":"m","gamma":"K per 100 m","psl":"Pa","ta":"K","hus":"kg kg-1",
       "pi":"m s-1","vpot":"m s-1","chi":"1","rh_mid":"1","chi_fast":"1"}


def fair_years(path):
    """CSV or JSON records. 'time' is a fractional-year boundary, 'year' a mean."""
    path=Path(path)
    table=pd.read_csv(path,dtype={"scenario":"string","draw_id":"string"}) if path.suffix==".csv" else pd.DataFrame(json.loads(path.read_text()))
    keys=["scenario","draw_id","forced_temperature","unit","baseline"]
    if not set(keys).issubset(table.columns) or table.empty or table[keys].isna().any().any():
        raise ValueError("FaIRGP needs identified scenario, draw, temperature, unit and baseline for every row")
    if not np.isfinite(table.forced_temperature.to_numpy(float)).all():
        raise ValueError("Missing or infinite FaIRGP temperature")
    input_sha=sha(path)
    records=[]
    for (scenario,draw),rows in table.groupby(["scenario","draw_id"],sort=False):
        if rows.unit.nunique()!=1 or rows.baseline.nunique()!=1:
            raise ValueError("Units and baseline must be constant within a path")
        unit=rows.unit.iloc[0]
        if unit not in ("K","degC","°C"):
            raise ValueError(f"Unsupported FaIRGP temperature unit {unit}")
        clocks=[col for col in ("time","year") if col in rows and rows[col].notna().any()]
        if len(clocks)!=1 or rows[clocks[0]].isna().any():
            raise ValueError("Each FaIRGP path needs one consistent annual or time-boundary clock")
        if clocks[0]=="time":
            rows=rows.sort_values("time")
            t=rows.time.to_numpy(float)
            v=rows.forced_temperature.to_numpy(float)
            if not np.isfinite(t).all() or np.any(np.diff(t)<=0) or np.any(np.diff(t)>1+1e-8):
                raise ValueError("Temperature boundaries must bracket every year")
            years=np.arange(int(np.ceil(t[0])),int(np.floor(t[-1])))
            gs=[]
            for y in years:
                nodes=np.r_[y,t[(t>y)&(t<y+1)],y+1]
                gs.append(np.trapezoid(np.interp(nodes,t,v),nodes))
            gs=np.array(gs)
        else:
            rows=rows.sort_values("year")
            labels=rows.year.to_numpy(float)
            if not np.isfinite(labels).all() or np.any(labels!=np.floor(labels)):
                raise ValueError("Annual labels must be integer years")
            years=labels.astype(int)
            if np.any(np.diff(years)<=0):
                raise ValueError("Duplicate annual label")
            gs=rows.forced_temperature.to_numpy(float)
        if not len(years) or not np.isfinite(gs).all():
            raise ValueError("Missing FaIRGP annual temperature")
        baseline=str(rows.baseline.iloc[0]).replace("–","-")
        offset=0.
        if baseline!="1850-1900":
            ref=(years>=1850)&(years<=1900)
            if ref.sum()!=51:
                raise ValueError("Rebaselining needs all 51 historical annual means for this path")
            offset=float(gs[ref].mean())
            gs=gs-offset
        if "temperature_definition" in rows and rows.temperature_definition.nunique(dropna=False)!=1:
            raise ValueError("Temperature definition changes within a FaIRGP path")
        definition=str(rows.temperature_definition.iloc[0]) if "temperature_definition" in rows else "unspecified"
        records.extend(dict(scenario=str(scenario),draw_id=str(draw),year=int(y),g=float(g),
                            temperature_definition=definition,baseline="1850-1900",input_baseline=baseline,input_unit=unit,
                            rebaseline_offset=offset,input_clock=clocks[0],forcing_file_sha256=input_sha) for y,g in zip(years,gs))
    return pd.DataFrame(records)


def physical_background(model,z,device="cpu"):
    fields=model.rep.decode(z[None],device)
    fields={key:a[0] for key,a in fields.items()}
    qc={}
    # Keep the valid-value denominator for physical adjustment rates.
    for key,low,high in [("mld",.01,np.inf),("gamma",0.,np.inf),("hus",0.,.10)]:
        a=fields[key]
        qc[key+"_finite"]=int(np.isfinite(a).sum())
        qc[key+"_clipped"]=int(np.sum((a<low)|(a>high)))
        fields[key]=np.clip(a,low,high)
    if not np.all((fields["psl"][np.isfinite(fields["psl"])]>70000)&(fields["psl"][np.isfinite(fields["psl"])]<115000)):
        raise ValueError("Sampled pressure outside the physical input range")
    fields.update(thermodynamics(fields["sst"].astype(float),fields["psl"].astype(float),
                                 model.rep.plev,fields["ta"].astype(float),fields["hus"].astype(float)))
    return fields,qc


def event(model,g,background_id,weather_id,days=15,seed=123,device="cpu",observed=None,extra_days=0):
    if not np.isfinite(g):
        raise ValueError("Annual warming must be finite")
    if days<1 or extra_days<0:
        raise ValueError("Positive integration window required")
    rb=fingerprint_rng(seed,"background",background_id)
    rw=fingerprint_rng(seed,"weather",background_id,weather_id)
    if observed is None:
        noise=rb.normal(size=model.bg.beta.shape[1])
        z=model.bg.draw(np.array([g]),noise[None],device)[0]
        fields,qc=physical_background(model,z,device)
    else:
        for name in ("plev","lat","lon"):
            if not np.array_equal(observed[name].values,getattr(model.rep,name)):
                raise ValueError(f"Observed {name} coordinate does not match the fitted representation")
        z=model.rep.encode(observed)
        fields={k:observed[k].values for k in UNITS}
        qc={}
    mu=model.mean.predict(model.bg,np.array([g]),z[None])[0]
    time=np.arange(-1,days+extra_days+2)*DAY
    noise=rw.normal(size=(len(time),len(model.lim.a)))
    w=model.lim.simulate(noise,device)+mu
    wind=model.rep.blocks["wind"].inverse(w,device)
    ds=xr.Dataset(coords={"lat":("lat",model.rep.lat,{"units":"degrees_north"}),
                          "lon":("lon",model.rep.lon,{"units":"degrees_east"}),
                          "plev":("plev",model.rep.plev,{"units":"Pa","positive":"down"}),
                          "t_seconds":("t_seconds",time,{"units":"s"}),"component":COMPONENTS})
    for k,a in fields.items():
        dims=("plev","lat","lon") if a.ndim==3 else ("lat","lon")
        ds[k]=(dims,a.astype("float32"),{"units":UNITS[k]})
    ds["wind"]=(("t_seconds","component","lat","lon"),wind,{"units":"m s-1"})
    ds["background_pc"]=("background_mode",z)
    ds["conditional_wind_pc"]=("wind_mode",mu)
    ds["wind_pc"]=(("t_seconds","wind_mode"),w)
    lo,hi=model.meta["g_range"]
    ds.attrs.update(model_id=model.meta["id"],g_degC=float(g),baseline="1850-1900",background_id=str(background_id),
                    weather_id=str(weather_id),duration_days=int(days),seed=int(seed),background_source="observed" if observed is not None else "emulated",
                    g_extrapolated=int(not lo<=g<=hi),physical_qc=json.dumps(qc),statistical_month=9)
    if observed is not None:
        for key in ("member","experiment","year"):
            ds.attrs["observed_"+key]=observed.attrs[key]
    return ds


def save_event(ds,path):
    with atomic(path) as tmp:
        ds.to_netcdf(tmp)
