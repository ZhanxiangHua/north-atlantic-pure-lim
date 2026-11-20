"""Discrete Gaussian propagation, stationary initialization, and deterministic backends."""
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
import torch


def factor(cov):
    vals,v=np.linalg.eigh((cov+cov.T)/2)
    tol=1e-10*max(1.,abs(vals).max())
    if vals.min() < -tol:
        raise ValueError(f"Indefinite covariance, minimum eigenvalue {vals.min()}")
    return v*np.sqrt(vals.clip(0))


def paired(x,day):
    """x [month,day,PC]. Adjacency is evaluated inside each member/month."""
    if x.ndim!=3 or len(day)!=x.shape[1]:
        raise ValueError("Expect grouped monthly records")
    valid=np.diff(day)==1
    return x[:,:-1][:,valid].reshape(-1,x.shape[2]), x[:,1:][:,valid].reshape(-1,x.shape[2])


@dataclass
class Dynamics:
    a: np.ndarray
    q: np.ndarray
    sigma: np.ndarray
    order: int
    observed_dim: int

    @cached_property
    def initial_factor(self):
        return factor(self.sigma)

    @cached_property
    def innovation_factor(self):
        return factor(self.q)

    def simulate(self,noise,device=None):
        """noise[...,day,state], with a single stationary initial draw."""
        d=self.observed_dim
        if device is None:
            a,x0,innovation=self.a,self.initial_factor,self.innovation_factor
            current=noise[...,0,:]@x0.T
            output=[current[...,:d]]
            for i in range(1,noise.shape[-2]):
                current=current@a.T+noise[...,i,:]@innovation.T
                output.append(current[...,:d])
            return np.stack(output,axis=-2)
        dtype=torch.float64
        a,initial,innovation,eps=[torch.as_tensor(v,device=device,dtype=dtype) for v in
                                  (self.a,self.initial_factor,self.innovation_factor,noise)]
        current=eps[...,0,:]@initial.T
        output=[current[...,:d]]
        for i in range(1,eps.shape[-2]):
            current=current@a.T+eps[...,i,:]@innovation.T
            output.append(current[...,:d])
        return torch.stack(output,dim=-2).cpu().numpy()

    @property
    def rho(self):
        return float(abs(np.linalg.eigvals(self.a)).max())

    @property
    def residual(self):
        return float(np.linalg.norm(self.sigma-self.a@self.sigma@self.a.T-self.q)/np.linalg.norm(self.sigma))

    def window_cov(self,days):
        s=self.sigma
        c=s.copy()
        out=days*s
        for h in range(1,days):
            c=self.a@c
            out=out+(days-h)*(c+c.T)
        return out[:self.observed_dim,:self.observed_dim]/days**2


def fit_dynamics(x,ridge,order=1):
    n,d,k=x.shape
    if order==1:
        previous,next_day=paired(x,np.arange(d))
    elif order==2:
        previous=np.concatenate([x[:,1:-1],x[:,:-2]],axis=-1).reshape(-1,2*k)
        next_day=x[:,2:].reshape(-1,k)
    else:
        raise ValueError("Supported orders are 1 and 2")
    c0=previous.T@previous/len(previous)
    c1=next_day.T@previous/len(previous)
    penalty=ridge*np.trace(c0)/len(c0)
    a=np.linalg.solve(c0+penalty*np.eye(len(c0)),c1.T).T
    eps=next_day-previous@a.T
    q=eps.T@eps/len(previous)
    if order==2:
        a=np.vstack([a,np.column_stack([np.eye(k),np.zeros((k,k))])])
        q=np.pad(q,[(0,k),(0,k)])
    if np.max(abs(np.linalg.eigvals(a)))>=1:
        raise ValueError("Unstable propagator")
    sigma=solve_discrete_lyapunov(a,q)
    out=Dynamics(a,q,(sigma+sigma.T)/2,order,k)
    factor(out.sigma)
    if out.residual>=1e-8:
        raise ValueError("Lyapunov residual exceeds tolerance")
    return out


def regress(design,target,ridge):
    gram=design.T@design
    penalty=np.eye(gram.shape[0])*ridge*len(design)
    penalty[0,0]=0
    return np.linalg.lstsq(gram+penalty,design.T@target,rcond=1e-12)[0]


def pair_statistics(x,order):
    """Unnormalized residual sufficient statistics. Each row comes from one September."""
    n,days,k=x.shape
    if order==1:
        prev,nxt=paired(x,np.arange(days))
    elif order==2:
        prev=np.concatenate([x[:,1:-1],x[:,:-2]],axis=2).reshape(-1,2*k)
        nxt=x[:,2:].reshape(-1,k)
    else:
        raise ValueError("Unsupported LIM order")
    return dict(xx=prev.T@prev,yx=nxt.T@prev,yy=nxt.T@nxt,count=len(nxt))


def from_statistics(stats,k,ridge,order):
    full=stats["yy"].shape[0]
    keep=np.arange(k) if order==1 else np.r_[np.arange(k),np.arange(k)+full]
    n=stats["count"]
    c0=stats["xx"][np.ix_(keep,keep)]/n
    c1=stats["yx"][:k,keep]/n
    cplus=stats["yy"][:k,:k]/n
    a=np.linalg.solve(c0+ridge*np.trace(c0)/len(c0)*np.eye(len(c0)),c1.T).T
    q=cplus-a@c1.T-c1@a.T+a@c0@a.T
    q=(q+q.T)/2
    if order==2:
        a=np.vstack([a,np.column_stack([np.eye(k),np.zeros((k,k))])])
        q=np.pad(q,[(0,k),(0,k)])
    if np.max(abs(np.linalg.eigvals(a)))>=1:
        raise ValueError("Unstable propagator")
    sigma=solve_discrete_lyapunov(a,q)
    out=Dynamics(a,q,(sigma+sigma.T)/2,order,k)
    factor(q)
    factor(out.sigma)
    if out.residual>=1e-8:
        raise ValueError("Lyapunov residual exceeds tolerance")
    return out


def matrix_diagnostics(dyn,stats,ridge):
    k=dyn.observed_dim
    full=stats["yy"].shape[0]
    ix=np.arange(k) if dyn.order==1 else np.r_[np.arange(k),np.arange(k)+full]
    count=int(stats["count"])
    xx=stats["xx"][np.ix_(ix,ix)]/count
    yx=stats["yx"][:k,ix]/count
    a=dyn.a[:k]
    q=stats["yy"][:k,:k]/count-a@yx.T-yx@a.T+a@xx@a.T
    penalty=float(ridge*np.trace(xx)/len(xx))
    return {"lag_pairs":count,"ridge_absolute":penalty,"ridge_trace_fraction":ridge,
            "regularized_predictor_condition_number":float(np.linalg.cond(xx+penalty*np.eye(len(xx)))),
            "residual_covariance_relative_asymmetry_before_symmetrizing":float(np.linalg.norm(q-q.T)/max(np.linalg.norm(q),1e-30)),
            "residual_covariance_relative_asymmetry_after_symmetrizing":float(np.linalg.norm(dyn.q-dyn.q.T)/max(np.linalg.norm(dyn.q),1e-30)),
            "minimum_residual_covariance_eigenvalue":float(np.linalg.eigvalsh(dyn.q)[0]),
            "minimum_stationary_covariance_eigenvalue":float(np.linalg.eigvalsh(dyn.sigma)[0]),
            "spectral_radius":dyn.rho,"relative_lyapunov_residual":dyn.residual}
