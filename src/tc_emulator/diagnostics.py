"""Weather diagnostics on the physical wind grid with equal member-month weighting."""
import numpy as np

from .wind_lim import paired


def region_area(lat,lon):
    mask=(lat[:,None]>=0)&(lat[:,None]<=60)&(lon[None,:]>=260)&(lon[None,:]<=360)
    area=np.cos(np.deg2rad(lat))[:,None]*mask
    return area/area.sum()


def steering(u):
    return np.stack([.2*u[...,0,:,:]+.8*u[...,2,:,:],.2*u[...,1,:,:]+.8*u[...,3,:,:]],axis=-3)


def window(a,d):
    c=np.concatenate([np.zeros_like(a[:,:1]),np.cumsum(a,axis=1,dtype=np.float64)],axis=1)
    return (c[:,d:]-c[:,:-d])/d


def sample_moments(wind,area):
    """Per-month sufficient statistics for physical, multi-day steering variance."""
    mean={d:[] for d in (1,5,10,15)}
    second={d:[] for d in mean}
    for i in range(0,len(wind),8):
        s=steering(wind[i:i+8])
        for d in mean:
            a=window(s,d)
            mean[d].append(a.mean(axis=1).astype("float32"))
            second[d].append(np.einsum("ntcyx,ntcyx,yx->n",a,a,area,optimize=True)/(a.shape[1]*2))
    return {d:(np.concatenate(mean[d]),np.concatenate(second[d])) for d in mean}


def observed_variance(moments,index,area):
    return {d:float(sec[index].mean()-(np.square(mu[index].mean(axis=0))*area).sum()/2)
            for d,(mu,sec) in moments.items()}


def steering_metric(rep,k):
    eof=rep.blocks["wind"]
    zero=eof.inverse(np.zeros((1,k)))
    maps=steering(eof.inverse(np.eye(k))-zero)
    area=region_area(rep.lat,rep.lon)
    b=(maps*np.sqrt(area/2)).reshape(k,-1)
    return b@b.T


def predict_variance(dyn,mu,metric):
    between=np.einsum("ni,ij,nj->",mu-mu.mean(axis=0),metric,mu-mu.mean(axis=0),optimize=True)/len(mu)
    return {d:float(np.sum(metric*dyn.window_cov(d))+between) for d in (1,5,10,15)}


def bootstrap_reference(moments,ids,area,seed,held_members=None):
    """Disjoint halves of training members, each retaining its whole sequence."""
    rng=np.random.default_rng(seed)
    people=np.unique(ids)
    ratios={d:[] for d in moments}
    if len(people)<4:
        raise ValueError("Need independent member halves for reference bands")
    # Integrate the Gram matrix over space once; cluster resamples index only members.
    grams={}
    for d,(mu,sec) in moments.items():
        grouped=np.stack([mu[ids==p].mean(axis=0) for p in people])
        flat=(grouped*np.sqrt(area/2)).reshape(len(people),-1)
        grams[d]=flat@flat.T,np.array([sec[ids==p].mean() for p in people])
    for _ in range(400):
        group=rng.permutation(len(people))
        size=min(held_members or len(people)//2,len(people)//2)
        l,r=group[:size],group[size:]
        for d,(gram,sec) in grams.items():
            a=sec[l].mean()-gram[np.ix_(l,l)].mean()
            b=sec[r].mean()-gram[np.ix_(r,r)].mean()
            ratios[d].extend([a/b,b/a])
    return {d:np.quantile(v,[.025,.975]).tolist() for d,v in ratios.items()}


def weak_runs(weak,area):
    lengths=np.zeros(weak.shape[1:],np.int32)
    hist=np.zeros(len(weak)+1)
    for flag in weak:
        ends=(~flag)&(lengths>0)
        np.add.at(hist,lengths[ends],area[ends])
        lengths=(lengths+1)*flag
    np.add.at(hist,lengths[lengths>0],area[lengths>0])
    return hist


def weighted_quantile(a,w,q):
    v=np.asarray(a).ravel()
    weights=np.broadcast_to(w,a.shape).ravel()
    good=np.isfinite(v)&(weights>0)
    order=np.argsort(v[good])
    val,weight=v[good][order],weights[good][order]
    return np.interp(q,(np.cumsum(weight)-.5*weight)/weight.sum(),val)


def climate_wind_stats(wind,area,weak_cut,reference=None):
    """Pool homologous grid cells. Lagged products stay within each 30-day month."""
    n,t,c,y,x=wind.shape
    avg=wind.mean(axis=(0,1),dtype=np.float64) if reference is None else reference
    lag=np.zeros((16,c))
    sum_wnd=np.zeros((c,y,x))
    cross=np.zeros((c,c))
    hist=np.zeros(t+1)
    shear_values=[]
    for month in wind:
        anom=month-avg
        sum_wnd+=month.mean(axis=0)
        for h in range(16):
            left=anom[:t-h]
            right=anom[h:]
            lag[h]+=np.einsum("dcyx,dcyx,yx->c",left,right,area,optimize=True)/(t-h)
        cross+=np.einsum("dcyx,dfyx,yx->cf",anom,anom,area,optimize=True)/t
        shear=np.hypot(month[:,0]-month[:,2],month[:,1]-month[:,3])
        shear_values.append(shear.astype("float32"))
        hist+=weak_runs(shear<weak_cut,area)
    return {"mean_ms":np.einsum("cyx,yx->c",sum_wnd/n,area).tolist(),
            "covariance_m2s2":(cross/n).tolist(),"lag0_to_15":(lag/lag[0]).tolist(),
            "shear_q05_q50_q95_ms":weighted_quantile(np.stack(shear_values),area,[.05,.5,.95]).tolist(),
            "weak_run_length_probability":(hist/max(hist.sum(),1e-20)).tolist(),
            "weak_runs_ge5_per_month":float(hist[5:].sum()/n)}


def steering_spread_maps(wind,threshold):
    """Gridpoint variance over member/window starts, and 5-day weak-shear spells."""
    sums,squares={},{}
    counts={d:0 for d in (5,10,15)}
    shape=wind.shape[-2:]
    long_runs=np.zeros(shape)
    for i in range(0,len(wind),8):
        chunk=wind[i:i+8]
        s=steering(chunk)
        for d in counts:
            avg=window(s,d)
            sums[d]=sums.get(d,0.)+avg.sum(axis=(0,1))
            squares[d]=squares.get(d,0.)+np.square(avg).sum(axis=(0,1))
            counts[d]+=avg.shape[0]*avg.shape[1]
        weak=np.hypot(chunk[:,:,0]-chunk[:,:,2],chunk[:,:,1]-chunk[:,:,3])<threshold
        length=np.zeros((len(chunk),*shape),int)
        for day in weak.transpose(1,0,2,3):
            length=(length+1)*day
            long_runs+=(length==5).sum(axis=0)
    spread={d:np.maximum(squares[d]/counts[d]-(sums[d]/counts[d])**2,0).mean(axis=0) for d in counts}
    return spread,long_runs/len(wind)


def modal_diagnostics(model,gg,samples):
    from .background import gaussian_scores
    z=np.stack([model.rep.encode(ds) for ds in samples])
    f=model.bg.factor(gg)
    cov=f@f.transpose(0,2,1)
    residual=z-model.bg.mean(gg)
    scale=np.sqrt(np.diagonal(cov,axis1=1,axis2=2))
    normalized=residual/scale
    # Center within each observed year before comparing member covariance.
    years=np.array([ds.attrs["year"] for ds in samples])
    scatter=np.zeros_like(cov[0])
    total=0
    for yr in np.unique(years):
        ix=years==yr
        if ix.sum()<2:
            continue
        scatter+=(ix.sum()-1)*np.cov(z[ix],rowvar=False)
        total+=ix.sum()-1
    if not total:
        raise ValueError("At least two members per year required for modal validation")
    scatter/=total
    predicted=cov.mean(axis=0)
    def correlation(c):
        sig=np.sqrt(c.diagonal())
        return c/np.outer(sig,sig)
    return {"marginal_normalized_second_moment":float(np.mean(normalized**2)),
            "marginal_90pct_coverage":float(np.mean(np.abs(normalized)<=1.644854)),
            "correlation_RMSE":float(np.mean((correlation(predicted)-correlation(scatter))**2)**.5),
            "held_nll":gaussian_scores(model.bg,gg,z),
            "empirical_year_centered_covariance":scatter.tolist(),"mean_predicted_covariance":predicted.tolist(),
            "min_factor_singular_value":float(np.linalg.svd(f,compute_uv=False).min())}
