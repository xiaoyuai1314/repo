import sys, os

sys.path.append('/home/xiaoyu/HelloWorld/venv/lib/python3.12/site-packages/Renormalizer_20240823')
sys.path.append('/home/xiaoyu/HelloWorld/tempo/主程序')
from tnpi_2 import InfluenceFunctionalCoeff
from EH_tempo2 import firstorder_IMTNPI
from renormalizer.utils import EvolveConfig, CompressConfig, CompressCriteria, EvolveMethod
from renormalizer.utils.constant import *
import numpy as np
import scipy
from functools import partial
import os
import logging

lamb = 35 * cm2au
w_c = 1/(50*fs2au)##频率的单位HZ（s^-1）,同样波数(cm-1)也能用作为单位。

def Jw(w):
    omega = np.absolute(w)
    res = 2*lamb*omega*w_c/(omega**2+w_c**2)
    if w<0:
        res*=-1
    return res

rho0 = np.zeros((7,7))
rho0[0,0] = 1
nsteps = 5

dt_fs = 4
dt = dt_fs * fs2au
beta = 1/ (77 * K2au)
h0 = np.array(
     [[200, -87.7, 5.5, -5.9, 6.7, -13.7, -9.9],
     [-87.7, 320, 30.8, 8.2, 0.7, 11.8, 4.3],
     [5.5, 30.8, 0, -53.5, -2.2, -9.6, 6.0],
     [-5.9, 8.2, -53.5, 110, -70.7, -17.0, -63.3],
     [6.7, 0.7, -2.2, -70.7, 270, 81.1, -1.3],
     [-13.7, 11.8, -9.6, -17.0, 81.1, 420, 39.7],
     [-9.9, 4.3, 6.0, -63.3, -1.3, 39.7, 230]]) * cm2au

assert np.allclose(h0, h0.T)

### eta
# \sum_{k=0}^{N-1} sum_{k'=0}^{k}
marg_multi = 50
wrange = [-w_c*marg_multi, w_c*marg_multi*(1+1e-10)]#np.array([-600,600+1e-10])*cm2au
coeff = InfluenceFunctionalCoeff(beta, Jw, dt, wrange, nsteps-1, order=1, two_side=False)
eta = np.ones((nsteps, nsteps), dtype=np.complex128)*np.nan
tmp = np.ones(nsteps, dtype=np.complex128)*np.nan
for dk in range(nsteps):
    tmp[dk] = coeff.kernel(dk,0)[0]

np.save("eta", tmp)
for k in range(nsteps):
    for kp in range(k+1):
        eta[k,kp] = tmp[k-kp]

fname=f"imag_evolution_{nsteps-1}_gpu"
evolve_time=[0]
rho=[rho0]

evolve_config = EvolveConfig(EvolveMethod.tdvp_ps,
        adaptive=False,
        guess_dt=1e-2/1j,
        adaptive_rtol=1e-5,
        normalize="mps_and_coeff",
        ivp_solver="RK45",
        ivp_rtol=1e-8,
        ivp_atol=1e-11,
        )
compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=64)

job=firstorder_IMTNPI(rho0,eta,nsteps,h0,dt,
                      compress_config = compress_config,
                      evolve_config = evolve_config,
                      test_model ="holstein",
                      insteps=2,
                      mpo_thresh=1e-7,
                      save_sparse=True)

rho = job.process()