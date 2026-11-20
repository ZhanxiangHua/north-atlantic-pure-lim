"""Canonical MPI grid, sub-MLD cooling gradients and FAST thermodynamic fields."""
from functools import lru_cache
from pathlib import Path
import sys
import warnings

import numpy as np
from scipy.spatial import cKDTree, Delaunay
from scipy.ndimage import distance_transform_edt

from .common import ROOT


def upstream():
    path = ROOT/"vendor/tropical_cyclone_risk"
    if not (path/".installed").exists():
        raise RuntimeError("Run scripts/prepare_fast.py before using FAST")
    if str(path) not in sys.path:
        sys.path.insert(0,str(path))


def pressure(values, units):
    scale = {"Pa":1,"hPa":100,"millibars":100,"mbar":100}.get(units)
    if scale is None:
        raise ValueError(f"Unrecognized pressure unit {units}")
    return np.asarray(values)*scale


def temperature(values, units):
    if units in ("K","kelvin"):
        return values
    if units in ("degC","degrees_C","degree_Celsius","degrees_Celsius"):
        return values+273.15
    raise ValueError(f"Unrecognized temperature unit {units}")


def grid_indices(lat,lon,domain,halo):
    west,east,south,north = domain
    x = (lon-west+halo)%360+west-halo
    ii = np.flatnonzero(x<=east+halo)
    ii = ii[np.argsort(x[ii])]
    jj = np.flatnonzero((lat>=south-halo)&(lat<=north+halo))
    jj = jj[np.argsort(lat[jj])]
    return jj,ii,lat[jj],x[ii]


def cut(a,j,i):
    return np.take(np.take(a,j,axis=-2),i,axis=-1)


def gradient(depth, temp, mld, height):
    """OLS slope on both interpolated endpoints and all interior native centres."""
    d = np.asarray(depth)
    if np.any(np.diff(d)<=0) or temp.shape[0]!=len(d):
        raise ValueError("Depth must increase in metres")
    finite = np.isfinite(temp)
    bottom = np.max(np.where(finite,d[:,None,None],-np.inf),axis=0)
    # Linear interpolation only with two valid enclosing depths.
    def at(z):
        i=np.clip(np.searchsorted(d,np.nan_to_num(z),side="right")-1,0,len(d)-2)
        j,k=np.indices(i.shape)
        w=(z-d[i])/(d[i+1]-d[i])
        val=(1-w)*temp[i,j,k]+w*temp[i+1,j,k]
        return np.where(z==d[i],temp[i,j,k],np.where(z==d[i+1],temp[i+1,j,k],val))
    lower=mld+height
    start,end=at(mld),at(lower)
    sums=[np.zeros_like(mld) for _ in range(5)]
    def add(z,t,valid):
        z=np.where(valid,z,0.)
        t=np.where(valid,t,0.)
        for total,term in zip(sums,(valid,z,t,z*z,z*t)):
            total+=term
    add(0.,start,np.isfinite(start))
    add(height,end,np.isfinite(end))
    for i,z in enumerate(d):
        add(z-mld,temp[i],(z>mld)&(z<lower)&finite[i])
    n,sx,sy,sxx,sxy=sums
    valid=np.isfinite(start)&np.isfinite(end)&(mld>=d[0])&(lower<=bottom)&(n>=2)
    slope=np.full_like(mld,np.nan)
    np.divide(-100*(sxy-sx*sy/np.maximum(n,1)),sxx-sx*sx/np.maximum(n,1),where=valid,out=slope)
    return slope,{"shallow_or_truncated":int(np.sum(np.isfinite(mld)&(bottom<lower))),
                  "missing":int(np.sum(np.isfinite(mld)&~np.isfinite(slope))),
                  "inversion":int(np.sum(slope<0)),"near_neutral":int(np.sum(np.abs(slope)<0.01))}


def ocean_geometry(lat,lon,y,x):
    """Local Delaunay interpolation with wet-vertex renormalization."""
    xx=(lon-250)%360+250
    # wrap the western fringe back toward the Atlantic domain
    xx=np.where(xx>500,xx-360,xx)
    near=(lat>=y.min()-6)&(lat<=y.max()+6)&(xx>=x.min()-6)&(xx<=x.max()+6)
    indices=np.flatnonzero(near)
    points=np.column_stack([xx.ravel()[indices],lat.ravel()[indices]])
    tri=Delaunay(points)
    tx,ty=np.meshgrid(x,y)
    targets=np.column_stack([tx.ravel(),ty.ravel()])
    simplex=tri.find_simplex(targets)
    if np.any(simplex<0):
        raise ValueError("Incomplete ocean halo")
    trans=tri.transform[simplex]
    bary=np.einsum("nij,nj->ni",trans[:,:2],targets-trans[:,2])
    weights=np.column_stack([bary,1-bary.sum(axis=1)]).clip(0,1)
    vertices=indices[tri.simplices[simplex]]
    return vertices,weights,tx.shape


def remap(a,geom):
    vertices,weights,shape=geom
    values=a.ravel()[vertices]
    w=weights*np.isfinite(values)
    total=w.sum(axis=1)
    return np.divide((np.nan_to_num(values)*w).sum(axis=1),total,out=np.full(len(total),np.nan),where=total>1e-7).reshape(shape)


def nearest_valid(a):
    valid=np.isfinite(a)
    if not valid.any():
        raise ValueError("No valid points on grid")
    idx=distance_transform_edt(~valid,return_distances=False,return_indices=True)
    return a[tuple(idx)]


def thermodynamics(sst,psl,plev,ta,q,units="Pa", legacy_q=False):
    upstream()
    from thermo import thermo
    import namelist
    p=pressure(plev,units)
    if not np.all(np.diff(p)<0) or not np.any(p==60000):
        raise ValueError("FAST requires descending pressure and a 600 hPa level")
    mid=int(np.flatnonzero(p==60000)[0])
    wet=np.isfinite(sst)&np.isfinite(psl)&np.all(np.isfinite(ta)&np.isfinite(q),axis=0)
    if not wet.any():
        raise ValueError("No complete profiles for PI")
    # Evaluate only valid columns. Keep a singleton horizontal axis for FAST.
    t=ta[:,wet,None]
    humidity=q[:,wet,None]
    mix=humidity if legacy_q else humidity/(1-humidity)
    ts=sst[wet,None]
    ps=psl[wet,None]
    with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
        pi=thermo.CAPE_PI_vectorized(ts,ps,p,t,mix)
        raw=thermo.sat_deficit(ts,ps,t[mid],p[mid],mix[mid])
        rh=thermo.conv_q_to_rh(t[mid],humidity[mid],p[mid])
        chi=raw.clip(0,10)
        adjusted=(np.exp(np.log(chi+1.e-3)+namelist.log_chi_fac)+namelist.chi_fac).clip(1e-5,5)
    result={}
    for key,sub in dict(pi=pi,vpot=pi*namelist.PI_reduc*np.sqrt(namelist.Ck/namelist.Cd),
                        chi=chi,chi_fast=adjusted,rh_mid=rh).items():
        field=np.full(sst.shape,np.nan)
        field[wet]=sub.ravel()
        result[key]=field
    return result
