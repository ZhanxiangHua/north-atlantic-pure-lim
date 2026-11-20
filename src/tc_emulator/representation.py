"""Area-weighted EOF transforms fitted only on training members."""
from dataclasses import dataclass,replace
import json

import numpy as np
import scipy.linalg as la
import torch
import xarray as xr

from .common import atomic, fingerprint_rng

BLOCKS={"sst":("sst",), "ocean":("mld","gamma"), "psl":("psl",), "profiles":("ta","hus")}


def stack_fields(ds,fields,log_mld=True):
    pieces=[]
    for name in fields:
        x=np.asarray(ds[name],dtype=np.float32)
        if name=="mld" and log_mld:
            x=np.log(np.maximum(x,0.01))
        if name=="hus":
            x=np.log(np.maximum(x,1e-10))
        pieces.append(x.reshape(-1,*x.shape[-2:]))
    return np.concatenate(pieces,axis=0)


def training_stats(x,area,vertical):
    # Accumulate with a bounded sample block to avoid a full float64 copy.
    count=np.zeros(x.shape[1:],np.int64)
    total=np.zeros(x.shape[1:])
    squared=np.zeros(x.shape[1:])
    for i in range(0,len(x),64):
        a=np.array(x[i:i+64],dtype=float)
        finite=np.isfinite(a)
        count+=finite.sum(axis=0)
        np.nan_to_num(a,copy=False,posinf=0.,neginf=0.)
        total+=a.sum(axis=0)
        squared+=np.square(a).sum(axis=0)
    valid=count>=max(1,int(np.ceil(.95*len(x))))
    mean=total/np.maximum(count,1)
    variance=np.maximum(0,squared/np.maximum(count,1)-mean**2)
    metric=area[None]*valid
    channel_scale=np.sqrt(np.sum(variance*metric,axis=(1,2))/np.maximum(metric.sum(axis=(1,2)),1e-14))
    channel_scale=np.maximum(channel_scale,1e-8)
    area_fraction=area[None]/np.maximum(metric.sum(axis=(1,2))[:,None,None],1e-14)
    weights=np.sqrt(area_fraction*vertical[:,None,None])/channel_scale[:,None,None]
    return mean,weights,valid


@dataclass
class EOF:
    mean: np.ndarray
    weights: np.ndarray
    valid: np.ndarray
    basis: np.ndarray
    variance: np.ndarray
    total_variance: float

    def transform(self,x,k=None):
        x=np.asarray(x).reshape(len(x),-1)[:,self.valid.ravel()]
        centered=np.where(np.isfinite(x),x-self.mean[self.valid],0.)
        return (centered*self.weights[self.valid])@self.basis[:,:k]

    def inverse(self,pc,device=None):
        pc=np.asarray(pc)
        e=self.basis[:,:pc.shape[-1]]
        if device is None:
            flat=pc@e.T
        else:
            flat=(torch.as_tensor(pc,dtype=torch.float64,device=device)@
                  torch.as_tensor(e.T,dtype=torch.float64,device=device)).cpu().numpy()
        flat=flat/self.weights[self.valid]+self.mean[self.valid]
        out=np.full(pc.shape[:-1]+(self.valid.size,),np.nan,dtype=np.float32)
        out[...,self.valid.ravel()]=flat
        return out.reshape(pc.shape[:-1]+self.mean.shape)


def eof(x,area,vertical,k,seed):
    mean,weights,valid=training_stats(x,area,vertical)
    n=len(x)
    f=int(valid.sum())
    k=min(k,n-1,f)
    if k<1:
        raise ValueError("At least two samples and one valid feature required")
    def chunks():
        for i in range(0,n,128):
            a=x[i:i+128].reshape(-1,valid.size)[:,valid.ravel()]
            yield np.where(np.isfinite(a),a-mean[valid],0.)*weights[valid]
    if n<=1024 and n<f:
        # Exact sample-space eigensystem for monthly maps.
        data=np.concatenate(list(chunks()))
        vals,vec=la.eigh(data@data.T,subset_by_index=[n-k,n-1])
        vals=vals[::-1]
        good=vals>max(vals[0],1.)*1.e-12
        basis=(data.T@vec[:,::-1][:,good])/np.sqrt(vals[good])
        total=float(np.square(data).sum()/(n-1))
        vals=vals[good]/(n-1)
    else:
        # Block subspace iteration on X^T X. The largest dense matrix is F x (k+24).
        rng=fingerprint_rng(seed,"EOF",k,n)
        q=rng.normal(size=(f,min(f,k+24)))
        for _ in range(3):
            product=np.zeros_like(q)
            for a in chunks():
                product+=a.T@(a@q)
            q=la.qr(product,mode="economic")[0]
        gram=np.zeros((q.shape[1],q.shape[1]))
        total=0.
        for a in chunks():
            aq=a@q
            gram+=aq.T@aq
            total+=np.einsum("ij,ij->",a,a)
        vals,rot=la.eigh(gram,subset_by_index=[gram.shape[0]-k,gram.shape[0]-1])
        vals=vals[::-1]/(n-1)
        basis=q@rot[:,::-1]
        total/=n-1
    return EOF(mean,weights,valid,basis,vals,total)


def vertical_weights(fields,levels):
    layer=np.abs(np.gradient(levels))
    layer/=layer.sum()
    return np.concatenate([layer if name in ("ta","hus") else np.ones(1) for name in fields])/len(fields)


def select_background_rank(block,values,members,candidates):
    """One-standard-error selection on member-level held-out reconstruction loss."""
    flat=values.reshape(len(values),-1)[:,block.valid.ravel()]
    x=np.where(np.isfinite(flat),flat-block.mean[block.valid],0.)*block.weights[block.valid]
    scores=x@block.basis
    total=np.einsum("ij,ij->i",x,x)
    ordered=sorted({min(k,scores.shape[1]) for k in candidates if k>0})
    if not ordered:
        raise ValueError("At least one positive rank is required")
    people=np.unique(members)
    rows=[]
    retained=np.cumsum(scores**2,axis=1)
    for k in ordered:
        residual=np.maximum(total-retained[:,k-1],0.)
        by_member=np.array([residual[members==p].sum()/max(total[members==p].sum(),1e-20) for p in people])
        sem=float(by_member.std(ddof=1)/np.sqrt(len(people))) if len(people)>1 else 0.
        rows.append({"rank":k,"unexplained_fraction":float(by_member.mean()),"member_standard_error":sem})
    best=min(rows,key=lambda row:row["unexplained_fraction"])
    cutoff=best["unexplained_fraction"]+best["member_standard_error"]
    k=next(row["rank"] for row in rows if row["unexplained_fraction"]<=cutoff)
    return replace(block,basis=block.basis[:,:k],variance=block.variance[:k]),{
        "selected_rank":k,"heldout_members":len(people),"one_standard_error_cutoff":cutoff,"candidates":rows}


class Representation:
    def __init__(self,blocks,lat,lon,plev,log_mld=True):
        self.blocks,self.lat,self.lon,self.plev=blocks,lat,lon,plev
        self.log_mld=log_mld

    def encode(self,ds):
        return np.concatenate([self.blocks[b].transform(stack_fields(ds,names,self.log_mld)[None])[0]
                               for b,names in BLOCKS.items()])

    def decode(self,z,device=None):
        fields={}
        start=0
        for b,names in BLOCKS.items():
            basis=self.blocks[b]
            k=basis.basis.shape[1]
            data=basis.inverse(z[...,start:start+k],device)
            start+=k
            pos=0
            for name in names:
                if name in ("ta","hus"):
                    value=data[...,pos:pos+len(self.plev),:,:]
                    pos+=len(self.plev)
                else:
                    value=data[...,pos,:,:]
                    pos+=1
                if name=="hus" or (name=="mld" and self.log_mld):
                    value=np.exp(value)
                fields[name]=value
        return fields

    def save(self,path):
        arrays=dict(lat=self.lat,lon=self.lon,plev=self.plev,log_mld=self.log_mld)
        for key,e in self.blocks.items():
            arrays.update({key+"_"+name:v for name,v in vars(e).items()})
        with atomic(path) as tmp:
            np.savez(tmp,**arrays)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as a:
            blocks={k:EOF(**{v:a[k+"_"+v] for v in EOF.__dataclass_fields__}) for k in [*BLOCKS,"wind"]}
            return cls(blocks,a["lat"],a["lon"],a["plev"],bool(a["log_mld"]))


def reconstruction_error(samples,rep,z,w,k,path):
    """Measure physical EOF truncation error independently of fitted dynamics."""
    stats={}
    for start in range(0,len(samples),4):
        part=samples[start:start+4]
        decoded=rep.decode(z[start:start+len(part)])
        pc=w[start:start+len(part),:,:k].reshape(-1,k)
        decoded["wind"]=rep.blocks["wind"].inverse(pc)
        for name,reconstructed in decoded.items():
            observed=np.stack([ds[name].values for ds in part]).reshape(reconstructed.shape)
            valid=np.isfinite(observed)&np.isfinite(reconstructed)
            actual=np.where(valid,observed,0.).astype(np.float64)
            projected=np.where(valid,reconstructed,0.).astype(np.float64)
            if name not in stats:
                stats[name]=np.zeros((6,*actual.shape[1:]),dtype="float64")
            totals=stats[name]
            for target,values in zip(totals,(valid,actual,projected,actual**2,projected**2,(actual-projected)**2)):
                target+=values.sum(axis=0)
    maps=xr.Dataset(coords={c:samples[0][c] for c in ("lat","lon","plev","component")})
    summary={}
    for name,(count,truth,recon,truth2,recon2,error2) in stats.items():
        count=np.where(count>0,count,np.nan)
        var=np.maximum(truth2/count-(truth/count)**2,0)
        pred=np.maximum(recon2/count-(recon/count)**2,0)
        mse=error2/count
        dims=samples[0][name].dims
        dims=dims[1:] if name=="wind" else dims
        unit=samples[0][name].attrs["units"]
        for label,value in (("reference_sd",var**.5),("reconstruction_sd",pred**.5),("truncation_rmse",mse**.5)):
            maps[f"{name}_{label}"]=(dims,value.astype("float32"),{"units":unit})
        area=np.broadcast_to(np.cos(np.deg2rad(rep.lat))[:,None],mse.shape)
        summary[name]=float(np.sqrt(np.nansum(mse*area)/np.nansum(var*area)))
    maps.attrs.update(member_months=len(samples),wind_rank=k,definition="Physical unit EOF reconstruction; identical input PCs before any stochastic dynamics")
    with atomic(path) as tmp:
        maps.to_netcdf(tmp)
    return summary
