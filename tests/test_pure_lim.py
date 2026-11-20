import numpy as np

from tc_emulator.pure_lim import lag_trace
from tc_emulator.pure_lim_memory import MemoryCalibration, MemoryLIM, response
from tc_emulator.wind_lim import Dynamics


def test_distributed_memory_stationary_windows():
    # A scalar AR(1) has a known covariance at every lag and across the event boundary.
    base = Dynamics(np.array([[.74]]), np.array([[1-.74**2]]), np.eye(1), 1, 1)
    cal = MemoryCalibration("distributed_memory", 1., 0., 3., 1., .08, -.12, .87,
                            delay=3, last_lag=8, feedback=.8, normalization=.96)
    process = MemoryLIM(base,cal)
    assert process.dynamics.residual < 1e-12
    d = np.array([1,5,10,15,30])
    lag = lag_trace(base,np.eye(1),40)
    # Independent time-domain calculation using the joint Toeplitz covariance.
    t = np.arange(-8,30)
    joint = (.74*.87)**abs(t[:,None]-t)
    filt = np.eye(30,len(t),k=8)
    for h in range(3,9):
        filt -= .8*np.eye(30,len(t),k=8-h)/6
    scale = (cal.amplitude(2.)*cal.normalization)**2
    truth = scale*filt@joint@filt.T
    np.testing.assert_allclose(cal.window_variance(2.,lag,d)[0],[truth[:n,:n].mean() for n in d])
    np.testing.assert_allclose(response(lag,.87,3,8,.8,30),truth[0]/scale,atol=1e-15)
    draws = process.sample(np.full(20000,2.),30,np.random.default_rng(25))[...,0]
    np.testing.assert_allclose([draws[:,:n].mean(axis=1).var() for n in d],
                               cal.window_variance(2.,lag,d)[0],rtol=.035)
    np.testing.assert_allclose(draws.var(axis=0)[[0,15,29]],np.diag(truth)[[0,15,29]],rtol=.035)


def test_memory_backends_and_event_prefix():
    base = Dynamics(np.array([[.5]]),np.array([[.75]]),np.eye(1),1,1)
    p = MemoryLIM(base, MemoryCalibration("distributed_memory",1.,0.,3.,1.,0.,-.1,.8,
                                          delay=2,last_lag=5,feedback=.65))
    g = np.array([-.5,1.,4.])
    noise = np.random.default_rng(72).standard_normal((3,33,1))
    full = p.propagate(g,noise,np.random.default_rng(73))
    torch = p.propagate(g,noise,np.random.default_rng(73),"cpu")
    np.testing.assert_allclose(full,torch,atol=1e-12)
    np.testing.assert_array_equal(full[:,:18],p.propagate(g,noise[:,:18],np.random.default_rng(73)))
