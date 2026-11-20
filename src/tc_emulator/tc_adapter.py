"""Replay observed or sampled environments with the supplied coupled BAM/FAST solver."""
from functools import lru_cache
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
import xarray as xr

from .common import COMPONENTS, ROOT, atomic, fingerprint_rng
from .physical import nearest_valid, upstream

DAY=86400.


class WindGridBoundary(ValueError):
    """A finite trial position is outside the daily wind field's spatial support."""


class EventWind:
    def __init__(self, ds, origin_s=0.):
        if list(ds.component.values)!=COMPONENTS:
            raise ValueError("Unexpected daily wind channel order")
        self.x=ds.lon.values
        self.y=ds.lat.values
        self.t=ds.t_seconds.values.astype(float)
        self.wind=ds.wind.transpose("t_seconds","lat","lon","component").values
        self.origin_s=origin_s
        if self.wind.shape!=(len(self.t),len(self.y),len(self.x),4) or not np.isfinite(self.wind).all():
            raise ValueError("Invalid daily wind grid")
        if any(np.any(np.diff(axis)<=0) for axis in (self.t,self.y,self.x)):
            raise ValueError("Event axes must increase")

    @staticmethod
    def bracket(axis,v,*,spatial=False):
        if not np.isfinite(v) or v<axis[0]-1e-7 or v>axis[-1]+1e-7:
            error = WindGridBoundary if spatial and np.isfinite(v) else ValueError
            raise error(f"Event coordinate {v} outside [{axis[0]}, {axis[-1]}]")
        i=int(np.clip(np.searchsorted(axis,v,side="right")-1,0,len(axis)-2))
        return i,(v-axis[i])/(axis[i+1]-axis[i])

    def __call__(self,lon,lat,seconds):
        xx=lon+360*np.rint(((self.x[0]+self.x[-1])/2-lon)/360)
        i,a=self.bracket(self.x,xx,spatial=True)
        j,b=self.bracket(self.y,lat,spatial=True)
        k,c=self.bracket(self.t,seconds+self.origin_s)
        cube=self.wind[k:k+2,j:j+2,i:i+2]
        by_lon=(1-a)*cube[:,:,0]+a*cube[:,:,1]
        by_lat=(1-b)*by_lon[:,0]+b*by_lon[:,1]
        return (1-c)*by_lat[0]+c*by_lat[1]


@lru_cache(maxsize=1)
def geography():
    upstream()
    from intensity import geo
    from util.basins import TC_Basin
    basin=TC_Basin("NA")
    mask_file=ROOT/"vendor/tropical_cyclone_risk/land/NA.nc"
    with xr.open_dataset(mask_file) as mask:
        region=RectBivariateSpline(mask.lon,mask.lat,mask.basin.values.T,kx=1,ky=1)
    return basin,region,geo.read_bathy(basin),geo.read_land(basin),geo.read_drag(basin)


def model(ds,days,offset,wind_provider=None):
    upstream()
    from intensity.coupled_fast import Coupled_FAST
    import namelist
    basin,region,bathy,land,drag=geography()
    w=EventWind(ds,offset) if wind_provider is None else wind_provider
    if w.t[0] > offset-DAY or w.t[-1] < offset+(days+1)*DAY:
        raise ValueError("Wind time padding is insufficient")
    # Use the upstream constructor including all physics and its external-wind branch.
    fast=Coupled_FAST(None,basin,np.datetime64("2000-09-05T12"),3600,days*DAY,wind_provider=w)
    fast.f_bath,fast.f_land,fast.f_Cd=bathy,land,drag
    fast.max_step_seconds=10800
    fast.h_bl=namelist.atm_bl_depth["NA"]
    lon,lat=ds.lon.values,ds.lat.values
    def field(name):
        return nearest_valid(ds[name].values.astype(float))
    # Coastal interpolation extends valid ocean values to the stencil. The solver's
    # high-resolution land, bathymetry and drag still apply at every evaluation.
    fast.init_fields(lon,lat,field("chi_fast"),field("vpot"),field("mld"),field("gamma"))
    fast.f_rh=RectBivariateSpline(lon,lat,field("rh_mid").T,kx=1,ky=1)
    return fast,region


def panel(seed,n):
    rng=fingerprint_rng(seed,"seed_panel")
    return pd.DataFrame({"seed_id":np.arange(n),"lon":rng.uniform(260.5,359.5,n),
                         "lat":np.rad2deg(np.arcsin(rng.uniform(np.sin(np.deg2rad(3)),np.sin(np.deg2rad(45)),n))),
                         "filter_draw":rng.uniform(size=n),"v_init":5+rng.normal(size=n),
                         "offset_s":np.zeros(n)})


def replay_event(ds,days,member_month=None,*,origin_day=5):
    start=origin_day-1  # The origin is the centre of its UTC day.
    if days<1 or start<1 or start+days+2>ds.sizes["day"]:
        raise ValueError("Observed September cannot cover the requested event and its two boundary nodes")
    if int(ds.day.values[start])!=origin_day or np.any(np.diff(ds.day)!=1):
        raise ValueError("Incomplete observed event")
    out=ds.drop_vars(["day","time"]).isel(day=slice(start-1,start+days+2)).rename_dims({"day":"t_seconds"})
    out=out.assign_coords(t_seconds=("t_seconds",np.arange(-1,days+2,dtype=float)*DAY,{"units":"s"}))
    ident=member_month or f"{ds.attrs['member']}-{ds.attrs['experiment']}-{ds.attrs['year']}"
    out.attrs.update(background_id=ident,weather_id=ident+"-replay",duration_days=days,origin_day=origin_day)
    return out


def along_track_wind(fast,lon,lat,seconds,intensity):
    weights=fast._calc_steering_coefs(intensity)
    motion,wind=fast._step_bam_track(lon,lat,seconds,weights)
    steering=[wind[indices]@weights for indices in (fast.u_Mean_idxs,fast.v_Mean_idxs)]
    return np.r_[wind,steering,motion,fast._calc_S(wind),weights]


def integrate_track(fast, lon, lat, intensity, moisture, seed_id, event_id):
    # RK trial stages can cross the wind halo before solve_ivp checks for a basin exit.
    # Retry the identical initial condition with a smaller step, using the same wind field.
    initial_step = fast.max_step_seconds
    try:
        for subdivision in (1, 2, 4, 8):
            fast.max_step_seconds = initial_step / subdivision
            try:
                result = fast.gen_track(lon, lat, intensity, moisture)
            except WindGridBoundary:
                if subdivision == 8:
                    raise
            else:
                if subdivision > 1:
                    warnings.warn(
                        f"Resolved FAST wind-grid boundary for {event_id}, seed {seed_id}: "
                        f"step cap {initial_step:g} -> {fast.max_step_seconds:g} seconds",
                        RuntimeWarning, stacklevel=2,
                    )
                return result
    finally:
        fast.max_step_seconds = initial_step


def tracks(event,seeds,days,*,wind_provider=None):
    upstream()
    import namelist
    from wind.tc_wind import axi_to_max_wind
    times=np.arange(0,days*DAY+1,3600)
    n,nstep=len(seeds),len(times)
    names=["storm_lon","storm_lat","v_axisymmetric","moisture","vmax",*COMPONENTS,
           "steering_u","steering_v","motion_u","motion_v","deep_shear","weight_250","weight_850"]
    units=["degrees_east","degrees_north","m s-1","1","m s-1"]+["m s-1"]*9+["1","1"]
    trace=np.full((n,nstep,len(names)),np.nan,dtype="float32")
    fast,region=model(event,days,float(seeds.offset_s.min()),wind_provider)
    event_id = f"{event.attrs.get('background_id','unknown')}/{event.attrs.get('weather_id','unknown')}"
    outcomes=[]
    for idx,s in enumerate(seeds.itertuples(index=False)):
        rec={"seed_id":int(s.seed_id),"lon":float(s.lon),"lat":float(s.lat),"offset_s":float(s.offset_s),
             "v_init":float(s.v_init),"filter_draw":float(s.filter_draw),"wet_seed":False,"latitude_pass":False,
             "pi_pass":False,"vent_pass":False,"integrated":False,"tc":False,"landfall":False,
             "termination":"screened","last_time_s":0.,"lmi":np.nan,"landfall_vmax":np.nan,
             "landfall_lon":np.nan,"landfall_lat":np.nan,"first_landfall_s":np.nan}
        outcomes.append(rec)
        fast.wind_provider.origin_s=float(s.offset_s)
        if fast.wind_provider.t[-1]<s.offset_s+(days+1)*DAY:
            raise ValueError("Insufficient padding for final seed")
        lon,lat=float(s.lon),float(s.lat)
        rec["wet_seed"]=bool(region.ev(lon,lat).item()>.5 and not fast._get_over_land(lon,lat))
        if not rec["wet_seed"]:
            rec["reason"]="outside_ocean_genesis_mask"
            continue
        prob=np.clip((abs(lat)-namelist.lat_vort_fac)/12,0,1)**namelist.lat_vort_power["NA"]
        rec["latitude_pass"]=bool(s.filter_draw<prob)
        if not rec["latitude_pass"]:
            rec["reason"]="latitude"
            continue
        vp=fast._get_current_vpot(lon,lat)
        rec["pi_pass"]=bool(vp>35)
        if not rec["pi_pass"]:
            rec["reason"]="low_pi"
            continue
        m=float(namelist.f_mInit(fast.f_rh.ev(lon,lat).item()))
        if wind_provider is not None:
            wind_provider.begin_seed(s.seed_id)
        result=integrate_track(fast,lon,lat,float(s.v_init),m,s.seed_id,event_id)
        if result is None:
            rec["reason"]="ventilation"
            continue
        if not result.success:
            raise RuntimeError(result.message)
        rec["vent_pass"]=rec["integrated"]=True
        t=result.t
        last=float(result.t_events[0][0]) if len(result.t_events[0]) else float(t[-1])
        rec["last_time_s"]=last
        if len(result.y_events[0]):
            end=result.y_events[0][0]
            rec["termination"]="basin_exit" if not fast.basin.in_basin(*end[:2],1) else "equator" if abs(end[1])<=2 else "dissipation"
        else:
            rec["termination"]="duration"
        xx,yy,v,moist=result.y
        env=np.stack([along_track_wind(fast,x,y,ts,strength) for x,y,ts,strength in zip(xx,yy,t,v)])
        vmax=np.asarray(axi_to_max_wind(xx,yy,3600,v,env[:,:4])).reshape(-1) if len(v)>1 else v.copy()
        trace[idx,:len(t)]=np.column_stack([xx,yy,v,moist,vmax,env])
        rec["lmi"]=float(vmax.max())
        rec["tc"]=bool(np.interp(2*DAY,t,v)>=namelist.seed_v_2d_threshold_ms and
                       v.max()>=namelist.seed_v_threshold_ms and vmax.max()>=namelist.seed_vmax_threshold_ms)
        land=fast.f_land.ev(xx,yy)>.5
        crossings=np.flatnonzero(land[1:]&~land[:-1])+1
        crossings=crossings[vmax[crossings]>=namelist.seed_vmax_threshold_ms]
        rec["landfall"]=bool(rec["tc"] and crossings.size)
        if rec["landfall"]:
            rec["landfall_vmax"]=float(vmax[crossings].max())
            first=int(crossings[0])
            rec.update(landfall_lon=float(xx[first]),landfall_lat=float(yy[first]),first_landfall_s=float(t[first]))
        rec["reason"]="accepted_tc" if rec["tc"] else "intensity_duration_threshold"
    ds=xr.Dataset({v:(("seed_id","elapsed_s"),trace[:,:,i],{"units":units[i]}) for i,v in enumerate(names)},
                  coords={"seed_id":seeds.seed_id.values,"elapsed_s":("elapsed_s",times,{"units":"s"})},
                  attrs={key:event.attrs[key] for key in ("background_id","weather_id")})
    ds.attrs.update(steering_definition="FAST intensity-dependent vertical weighting at the storm location",
                    motion_definition="FAST environmental steering plus latitude-dependent beta drift")
    frame=pd.DataFrame(outcomes)
    for k in ("background_id","weather_id"):
        frame[k]=event.attrs[k]
    return ds,frame


def write_tracks(event,seeds,days,path,*,wind_provider=None):
    ds,outcomes=tracks(event,seeds,days,wind_provider=wind_provider)
    path=Path(path)
    with atomic(path) as tmp:
        ds.to_netcdf(tmp,encoding={v:{"zlib":True,"complevel":2} for v in ds.data_vars})
    with atomic(path.with_suffix(".parquet")) as tmp:
        outcomes.to_parquet(tmp,index=False)
    return ds,outcomes
