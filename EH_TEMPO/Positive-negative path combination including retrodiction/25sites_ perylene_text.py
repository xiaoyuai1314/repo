import sys, os

sys.path.append('/home/xiaoyu/HelloWorld/venv/lib/python3.12/site-packages/Renormalizer_20240823')
sys.path.append('/home/xiaoyu/HelloWorld/tempo/主程序')
from tnpi_2 import InfluenceFunctionalCoeff
from renormalizer.utils import EvolveConfig, CompressConfig, CompressCriteria, EvolveMethod
from renormalizer.utils.constant import *
from EHsys_tempo import firstorder_IMTNPI
import numpy as np
import scipy
from functools import partial
import os
import logging
import time

logger = logging.getLogger("renormalizer")
S = [0.015, 0.028, 0.449, 3.800, 1.169, 0.337, 0.317, 0.601,
     0.033, 0.045, 0.023, 0.033, 0.029, 0.040, 0.197, 0.215,
     0.027, 0.013, 0.021, 0.019, 0.037, 0.010, 0.033, 0.010,
     0.208, 0.042, 0.083, 0.039]

w = [7, 10, 12, 14, 26, 26, 32, 46,
     50, 55, 63, 79, 104, 183, 206, 211,
     272, 416, 483, 540, 552, 643, 751, 1325,
     1371, 1469, 1570, 1628]

def compute_Jw(S_list, w_list):

    S = np.asarray(S_list)
    w = np.asarray(w_list)
    
    assert len(S) == len(w)
    Jw_val = np.pi * S * w ** 2
    return Jw_val

w_au = np.array(w) * cm2au  
Jw_au = compute_Jw(S, w_au)

# 拼接正负频率（TEMPO 要求）
w_combined = np.concatenate([w_au, -w_au])
Jw_combined = np.concatenate([Jw_au, -Jw_au])

nsteps = 5
dt_fs = 1
dt = dt_fs * fs2au
beta = 1/(300 * K2au)
rho0 = np.zeros((25,25))
rho0[0,0] = 1
#print("nsteps",nsteps)

def H_s(sites,J):
    h0 = np.zeros((sites,sites))
    for i in range(sites-1):
        h0[i,i+1] = J
        h0[i+1,i] = J
    return h0
h0 = H_s(25, -500) * cm2au
assert np.allclose(h0, h0.T)

wrange = w_combined
coeff = InfluenceFunctionalCoeff(beta, Jw_combined, dt, wrange, nsteps-1, order=1, two_side=False,discrete = True)
eta = np.ones((nsteps, nsteps), dtype=np.complex128)*np.nan
tmp = np.ones(nsteps, dtype=np.complex128)*np.nan
for dk in range(nsteps):
    tmp[dk] = coeff.kernel(dk,0)[0]

np.save("eta", tmp)
for k in range(nsteps):
    for kp in range(k+1):
        eta[k,kp] = tmp[k-kp]

logger.info(f'{eta[-1,:]}')

fname=f"{nsteps}_gpu_bond_200"
evolve_time=[0]
rho=[rho0]
start_time = time.time()
evolve_config = EvolveConfig(EvolveMethod.tdvp_ps,
        adaptive=True,
        guess_dt=1e-2/1j,
        adaptive_rtol=1e-5,
        normalize="mps_and_coeff",
        ivp_solver="RK45",
        ivp_rtol=1e-8,
        ivp_atol=1e-11,
        )
#compress_config = CompressConfig(CompressCriteria.threshold, threshold=1e-6)
compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=50)

job=firstorder_IMTNPI(rho0,eta,nsteps,h0,dt,
                      compress_config = compress_config,
                      evolve_config = evolve_config,
                      test_model ="holstein",
                      insteps=10,
                      mpo_thresh=1e-7,
                      save_sparse=True)
rho = job.process()
dump_dict = dict()
dump_dict["rho"] = rho
np.savez(f"{fname}", **dump_dict)
end_time = time.time()
envolution_time = end_time - start_time
print(f"运行时间: {envolution_time:.6f} 秒")
