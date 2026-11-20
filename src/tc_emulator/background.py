"""Conditional joint monthly Gaussian and conditional circulation regression."""
from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from .common import atomic, digest
from .wind_lim import regress


def polynomial(g):
    g=np.atleast_1d(g)
    return np.stack([np.ones_like(g),g,g**2],axis=-1)


@dataclass
class Background:
    center: float
    scale: float
    beta: np.ndarray
    linear_beta: np.ndarray
    lower0: np.ndarray
    lower1: np.ndarray
    pooled: np.ndarray

    def standard_g(self,g):
        return (np.asarray(g)-self.center)/self.scale

    def mean(self,g):
        return polynomial(self.standard_g(g))@self.beta

    def factor(self,g):
        return self.lower0+self.standard_g(g)[...,None,None]*self.lower1

    def draw(self,g,noise,device=None):
        mu=self.mean(g)
        f=self.factor(g)
        if device is None:
            return mu+np.einsum("...ij,...j->...i",f,noise)
        tensors=[torch.as_tensor(v,device=device,dtype=torch.float64) for v in (mu,f,noise)]
        return (tensors[0]+torch.einsum("...ij,...j->...i",*tensors[1:])).cpu().numpy()


def fit_background(g,z,years,checkpoint,steps=350,device="cpu"):
    center=float(g.mean())
    scale=float(g.std())
    if scale<1.e-6 or len(np.unique(g))<3:
        raise ValueError("Warming range does not support quadratic mean")
    gn=(g-center)/scale
    design=polynomial(gn)
    beta=np.linalg.lstsq(design,z,rcond=None)[0]
    linear=np.linalg.lstsq(design[:,:2],z,rcond=None)[0]
    unique=np.unique(years)
    groups=[z[years==y] for y in unique]
    if min(map(len,groups))<2:
        raise ValueError("Yearly covariance requires multiple training members")
    weights=np.array([len(group)-1 for group in groups],dtype=float)
    weights/=weights.sum()
    covs=np.stack([np.cov(group,rowvar=False) for group in groups])
    pooled=np.average(covs,weights=weights,axis=0)
    normalization=np.sqrt(np.maximum(pooled.diagonal(),1.e-12))
    target=covs/normalization[None,:,None]/normalization[None,None,:]
    pooled_std=np.average(target,axis=0,weights=weights)
    cov0=.95*pooled_std+.05*np.eye(z.shape[-1])
    # Use this same nonsingular, shrunken covariance for the constant baseline,
    # for the fitted factor's starting point and for its likelihood comparison.
    pooled=cov0*normalization[:,None]*normalization[None,:]
    def tensor(x):
        return torch.as_tensor(x,device=device,dtype=torch.float64)
    L0=torch.nn.Parameter(tensor(np.linalg.cholesky(cov0)))
    L1=torch.nn.Parameter(torch.zeros_like(L0))
    opt=torch.optim.Adam([L0,L1],lr=.009)
    gtarget=tensor(np.array([gn[years==y][0] for y in unique])[:,None,None])
    targets=tensor(target)
    weight=tensor(weights)
    target_hash=hashlib.sha256(np.ascontiguousarray(covs).view(np.uint8)).hexdigest()
    fingerprint=digest([unique.tolist(),g.tolist(),beta.tolist(),target_hash,steps])
    initial=0
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location=device,weights_only=True)
        if state["fingerprint"]==fingerprint:
            with torch.no_grad():
                L0.copy_(state["L0"])
                L1.copy_(state["L1"])
            opt.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["rng"].cpu())
            initial=state["step"]
    for step in range(initial,steps):
        factor=torch.tril(L0)+gtarget*torch.tril(L1)
        errors=(factor@factor.mT-targets).square().mean(dim=(1,2))
        loss=errors@weight+.005*L1.square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step+1)%50==0 or step+1==steps:
            with atomic(checkpoint) as tmp:
                torch.save({"fingerprint":fingerprint,"step":step+1,"L0":L0.detach(),"L1":L1.detach(),
                            "optimizer":opt.state_dict(),"rng":torch.get_rng_state()},tmp)
    return Background(center,scale,beta,linear,np.tril(L0.detach().cpu().numpy())*normalization[:,None],
                      np.tril(L1.detach().cpu().numpy())*normalization[:,None],pooled)


@dataclass
class Circulation:
    coefficients: np.ndarray
    scale: np.ndarray

    def predict(self,bg,g,z):
        design=np.column_stack([np.ones(len(z)),bg.standard_g(g),(z-bg.mean(g))/self.scale])
        return design@self.coefficients


def fit_mean(bg,g,z,wbar,ridge):
    anomaly=z-bg.mean(g)
    scale=np.maximum(anomaly.std(axis=0),1e-8)
    design=np.column_stack([np.ones(len(z)),bg.standard_g(g),anomaly/scale])
    return Circulation(regress(design,wbar,ridge),scale)


def gaussian_scores(bg,g,z):
    """Score exactly the covariance used for sampling. No likelihood-only jitter."""
    delta=z-bg.mean(g)
    scores=[]
    for f,v in zip(bg.factor(g),delta):
        logdiag=np.abs(f.diagonal())
        if np.any(logdiag<1e-10):
            return np.inf
        residual=np.linalg.solve(f,v)
        scores.append(.5*(residual@residual+2*np.log(logdiag).sum()+len(v)*np.log(2*np.pi)))
    return float(np.mean(scores))


def select_covariance(bg,g,z):
    from dataclasses import replace
    baseline=np.linalg.cholesky(bg.pooled)
    # The convex combination of lower triangular factors preserves the affine
    # factor form. Select its strength with held-member Gaussian log scores.
    candidates=[]
    for s in (1.,.75,.5,.25,0.):
        model=replace(bg,lower0=s*bg.lower0+(1-s)*baseline,lower1=s*bg.lower1)
        loss=gaussian_scores(model,g,z)
        candidates.append((loss,s,model))
    loss,s,chosen=min(candidates,key=lambda row:row[0])
    if not np.isfinite(loss):
        raise ValueError("No nonsingular background factor on validation members")
    return chosen,{"selected_fraction":s,"held_member_mean_nll":[{"fraction":x,"nll":v if np.isfinite(v) else None} for v,x,_ in candidates]}
