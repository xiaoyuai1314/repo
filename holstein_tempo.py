import sys, os

sys.path.append('/home/xiaoyu/HelloWorld/venv/lib/python3.12/site-packages/Renormalizer_20240823')
sys.path.append('/home/xiaoyu/HelloWorld/tempo/主程序')

from mps_compress import mps_compress
import numpy as np
import scipy
import sympy as sp
from tnpi_2 import InfluenceFunctionalCoeff
from f_version_tempo  import tempo
import time
from renormalizer.utils.constant import *

lamb = 35 * cm2au
w_c = 1/(50*fs2au)

def Jw(w):
    omega = np.absolute(w)
    res = 2*lamb*omega*w_c/(omega**2+w_c**2)
    if w<0:
        res*=-1
    return res

rho0 = np.zeros((7,7))
rho0[0,0] = 1
nsteps = 10
dt_fs = 4
dt = dt_fs * fs2au
beta = 1/ (77 * K2au)
dtype = np.complex256

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

nmems = None  # 记忆长度（设置为None则不截断）
M = 30

start_time = time.time()
system = tempo(h0, dt, rho0, eta, nsteps, M, problem='holstein', dump_res = False, nmems=nmems, precision=np.complex128)
rho = system.contract_mps_to_density_matrix()
end_time = time.time()
# 计算运行时间（秒）
execution_time = end_time - start_time
print(f"运行时间: {execution_time:.6f} 秒")
