import sys
sys.path.append('/home/xiaoyu/HelloWorld/venv/lib/python3.12/site-packages/Renormalizer_20240823')
sys.path.append('/home/xiaoyu/HelloWorld/tempo/主程序')
from renormalizer.mps.backend import OE_BACKEND, xp, np
from renormalizer.mps.matrix import asxp, asnumpy
from renormalizer.mps.lib import compressed_sum
from renormalizer.mps import Mps, Mpo
from renormalizer.model import Model, Op, basis as Ba
from renormalizer.utils import TdMpsJob, OptimizeConfig, EvolveConfig, CompressConfig, CompressCriteria, EvolveMethod
from renormalizer.utils import log, Quantity
from scipy.ndimage import gaussian_filter1d
import scipy.linalg
import itertools
import scipy.integrate
import logging
import opt_einsum as oe
from functools import reduce
import mpmath 


logger = logging.getLogger("renormalizer")

def shrink_mps(mps):
    """
    shrink mps by 2 sites
    """
    mps_list = mps._mp
    del mps_list[-2:]
    basis = mps.model.basis
    del basis[-2:]

    return Mps.from_mp(Model(basis,[]), mps_list)

def expectation_ob(mps1, mpo, mps2):
    tmp = np.tensordot(np.eye(mps1[0].shape[0]), np.eye(mpo[0].shape[0]), axes=0)
    res = np.tensordot(tmp, np.eye(mps2[0].shape[0]), axes=0).transpose(0,2,4,1,3,5)
    
    for i in range(mps1.site_num):
        assert res.shape[-3] == mps1[i].shape[0]
        assert res.shape[-2] == mpo[i].shape[0]
        assert res.shape[-1] == mps2[i].shape[0]

        res = oe.contract("xyzabc, ade, bdfg, cfh -> xyzegh", res, mps1[i], mpo[i],
                mps2[i], backend=OE_BACKEND)
    return res

def contract_single_mps(mps, mode):
    res = xp.eye(1)
    for ims, ms in enumerate(mps):
        if mode[ims] == "all":
            tmp = oe.contract("ijk->ik", asxp(ms), backend=OE_BACKEND)
        elif mode[ims] is None:
            tmp = ms
        else:
            tmp = ms[:,mode[ims],:]

        res = xp.tensordot(res, asxp(tmp), axes=1)
    return asnumpy(res)


class InfluenceFunctionalCoeff:

    def __init__(self, beta, Jw, dt, wrange, N, order=2, discrete=False,
            two_side=True, epsabs=1e-10, epsrel=1e-5):
        self.beta = beta
        self.Jw = Jw
        self.dt = dt
        self.wrange = wrange 
        self.N = N
        self.order = order
        self.discrete = discrete
        self.two_side = two_side
        self.epsabs = epsabs
        self.epsrel = epsrel
        logger.info(f"eta integration threshold, epsabs:{epsabs}, epsrel:{epsrel}")

    def kernel(self, k, kp, N=None):
        if N is None:
            N = self.N
        assert k in np.arange(self.N+1)
        assert kp in np.arange(self.N+1)
        if self.order == 2:
            if 0 < kp < k < N:
                def func(w):
                    res = 2/np.pi * self.Jw(w)/w**2 \
                        * np.exp(self.beta*w/2)/np.sinh(self.beta*w/2)\
                        * np.sin(w*self.dt/2)**2 \
                        * np.exp(-1j*w*self.dt*(k-kp))
                    return res
            elif 0 < k == kp  < N:
                def func(w):
                    res = 0.5/np.pi * self.Jw(w)/w**2 \
                        * np.exp(self.beta*w/2)/np.sinh(self.beta*w/2)\
                        * (1-np.exp(-1j*w*self.dt))
                    return res
            elif k == N and kp == 0:
                def func(w):
                    res = 2/np.pi * self.Jw(w)/w**2 \
                        * np.exp(self.beta*w/2)/np.sinh(self.beta*w/2)\
                        * np.sin(w*self.dt/4)**2 \
                        * np.exp(-1j*w*self.dt*(N-1/2))
                    return res
            elif k == kp == 0 or k == kp == N:
                def func(w):
                    res = 0.5 / np.pi * self.Jw(w)/w**2 \
                        * np.exp(self.beta*w/2)/np.sinh(self.beta*w/2)\
                        * (1-np.exp(-1j*w*self.dt/2))
                    return res
            elif kp == 0 and 0 < k < N:
                def func(w):
                    res = 2/np.pi * self.Jw(w)/w**2 \
                        * np.exp(self.beta*w/2)/np.sinh(self.beta*w/2)\
                        * np.sin(w*self.dt/2)*np.sin(w*self.dt/4) \
                        * np.exp(-1j*w*self.dt*(k-1/4))
                    return res
            elif k == N  and 0 < kp < N:
                def func(w):
                    res = 2/np.pi * self.Jw(w)/w**2 \
                        * np.exp(self.beta*w/2)/np.sinh(self.beta*w/2)\
                        * np.sin(w*self.dt/2)*np.sin(w*self.dt/4) \
                        * np.exp(-1j*w*self.dt*(N-kp-1/4))
                    return res
            else:
                assert False
        elif self.order == 1:
            if kp < k:
                if self.discrete:
                    w = self.wrange
                    res = 2/np.pi * np.sum(self.Jw/w**2 \
                        * np.exp(self.beta*w/2)/np.sinh(self.beta*w/2)\
                        * np.sin(w*self.dt/2)**2 \
                        * np.exp(-1j*w*self.dt*(k-kp)))
                    return res, 0
                if self.beta == np.inf:
                    def func(w):
                        #η_{k,k'} = (4/π) ∫_0 dω J(ω)/ω² sin²(ωΔt/2) e^{-iωΔt(k-k')}
                        w = mpmath.mpf(w)
                        res = 4/np.pi * self.Jw(w)/w**2 \
                            * mpmath.sin(w*self.dt/2)**2 \
                            * mpmath.exp(-1j*w*self.dt*(k-kp))
                        return complex(res)
                else:
                    def func(w):
                        w = mpmath.mpf(w)
                        res = 2/np.pi * self.Jw(w)/w**2 \
                            * mpmath.exp(self.beta*w/2)/mpmath.sinh(self.beta*w/2)\
                            * mpmath.sin(w*self.dt/2)**2 \
                            * mpmath.exp(-1j*w*self.dt*(k-kp))
                        return complex(res)
                
            elif k == kp:
                if self.discrete:
                    w = self.wrange
                    res = 0.5/np.pi * np.sum(self.Jw/w**2 \
                        * np.exp(self.beta*w/2)/np.sinh(self.beta*w/2)\
                        * (1-np.exp(-1j*w*self.dt)))
                    return res, 0
                if self.beta == np.inf:
                    def func(w):
                        w = mpmath.mpf(w)
                        res = 1/np.pi * self.Jw(w)/w**2 \
                            * (1-mpmath.exp(-1j*w*self.dt))
                        return complex(res)
                else:
                    def func(w):
                        w = mpmath.mpf(w)
                        res = 0.5/np.pi * self.Jw(w)/w**2 \
                            * mpmath.exp(self.beta*w/2)/mpmath.sinh(self.beta*w/2)\
                            * (1-mpmath.exp(-1j*w*self.dt))
                        return complex(res)
            else:
                assert False
        else:
            assert False
        
        if self.two_side:
            val1, error1 = scipy.integrate.quad(
                      func, self.wrange[0], self.wrange[1], complex_func=True,
                      limit=1000, epsabs=self.epsabs, epsrel=self.epsrel)
            val2, error2 = scipy.integrate.quad(
                      func, -self.wrange[1], -self.wrange[0], complex_func=True,
                      limit=1000, epsabs=self.epsabs, epsrel=self.epsrel)
            logger.debug(f"{k}, {kp}, eta: val1, error1: {val1}, {error1}")
            logger.debug(f"{k}, {kp}, eta: val2, error2: {val2}, {error2}")
            return val1+val2, error1+error2
        else:
            val, error = mpmath.quadsubdiv(func, [self.wrange[0], self.wrange[1]], error=True)
            logger.debug(f"{k}, {kp}, eta: val, mpmath error: {val}, {error}")
            
            val, error = scipy.integrate.quad(
                      func, self.wrange[0], self.wrange[1], complex_func=True,
                      limit=1000, epsabs=self.epsabs, epsrel=self.epsrel)
            logger.debug(f"{k}, {kp}, eta: val, scipy error: {val}, {error}")
            return val, error
        


class TimeCorrelationFunction:
    r"""
    Bath time correlation function

    C(t) = (1/pi) ∫₀^∞ dω J(ω) [ coth(βω/2) cos(ωt) - i sin(ωt) ]

    Atomic units: ħ = 1
    """

    def __init__(
        self,
        beta,
        Jw,
        dt,
        w_range,
        N=1000,
        two_side=True,
        discrete=False,
        smooth_sigma=None,
        omega_k=None,#特定频率的列表
        
    ):
        self.beta = beta
        self.Jw = Jw
        self.dt = dt
        self.w_range = w_range
        self.Nt = N
        self.two_side = two_side
        self.discrete = discrete
        self.omega_k = omega_k
        self.smooth_sigma = smooth_sigma

        self.t_list = self._generate_time_list()
        self.re_C, self.im_C = self._compute_correlation()

    # ------------------------------------------------------------------
    def _generate_time_list(self):
        t_pos = np.linspace(0.0, self.Nt * self.dt, self.Nt)

        if self.two_side:
            t_neg = -t_pos[1:]
            t_list = np.concatenate([t_neg[::-1], t_pos])
        else:
            t_list = t_pos

        return t_list

    # ------------------------------------------------------------------
    def _compute_correlation(self):
        if self.discrete:
            return self._compute_discrete()
        else:
            return self._compute_continuous()

    # ------------------------------------------------------------------
    def _compute_discrete(self):
        """
        Discrete bath modes:
        C(t) = (1/pi) Σ_k J(ω_k)[ coth(βω_k/2)cos(ω_k t) - i sin(ω_k t) ]
        """
        omega = np.asarray(self.omega_k)
        J_vals = self.Jw(omega)
        coth = 1.0 / np.tanh(self.beta * omega / 2.0)

        cos_mat = np.cos(np.outer(self.t_list, omega))
        sin_mat = np.sin(np.outer(self.t_list, omega))

        re_C = np.sum(J_vals * coth * cos_mat, axis=1) / np.pi
        im_C = -np.sum(J_vals * sin_mat, axis=1) / np.pi

        return re_C, im_C

    # ------------------------------------------------------------------
    def _compute_continuous(self):
        """
        Continuous bath using frequency discretization + trapezoidal rule
        """
        w_min, w_max = self.w_range
        omega = np.linspace(w_min, w_max, self.Nt)

        # avoid ω=0 singularity explicitly
        omega[0] = 1e-12

        J_vals = self.Jw(omega)
        coth = 1.0 / np.tanh(self.beta * omega / 2.0)

        cos_mat = np.cos(np.outer(self.t_list, omega))
        sin_mat = np.sin(np.outer(self.t_list, omega))

        re_integrand = J_vals * coth * cos_mat
        im_integrand = J_vals * sin_mat

        re_C = np.trapz(re_integrand, omega, axis=1) / np.pi
        im_C = -np.trapz(im_integrand, omega, axis=1) / np.pi

        # 对结果进行高斯平滑（消除积分噪声）
        if self.smooth_sigma is not None:
            logger.info(f"Applying Gaussian smoothing with sigma={self.smooth_sigma}")
            re_C = gaussian_filter1d(np.array(re_C), sigma=self.smooth_sigma)
            im_C = gaussian_filter1d(np.array(im_C), sigma=self.smooth_sigma)

        return re_C, im_C


class FirstOrderBSTopDown:
    """
    SBM
    GU = Ub(dt) Us(dt)
    rho_0|s0><s0|Us|s1><s1|Us|s2>....<sN-1|Us|sN>  Ub(s0)Ub(s1)....Ub(SN-1)
    """
    def __init__(self, eta, nsteps, rho0, h0, dt,
            nmem=np.inf, 
            compress_config: CompressConfig = None,
            evolve_config: EvolveConfig = None,
            fname = "",
            ):

        self.eta = eta
        self.nsteps = nsteps
        self.rho = [rho0]
        self.h0 = h0
        self.dt = dt
        self.nmem = nmem #memory length

        if compress_config is None:
            self.compress_config = CompressConfig()
        else:
            self.compress_config = compress_config
        
        if evolve_config is None:
            self.evolve_config = EvolveConfig()
        else:
            self.evolve_config = evolve_config
        
        self.sys_mps = self.system_mps()
        self.inf_mps = self.influence_mps()
        self.fname = fname

    def system_mps(self):
        """
        (rho0, branchf_0) - (branchb_0, propagator_{0,1}) - branchf_1 -
        (branchb_1, propagator{1,2}) --- (branchb_N-1, propagator{N-1,N}) -
        """

        identity = np.eye(2)
        
        branch_node = np.zeros((2,2,2), dtype=np.complex128)
        for i in range(2):
            branch_node[i,i,i] = 1
        branch = np.tensordot(identity, branch_node, axes=0)
        branchf = branch.transpose(2, 0, 3, 4, 1).reshape(4,2,4) # forward
        branchb = branch.transpose(0, 2, 3, 1, 4).reshape(4,2,4) # backward
        
        # SBM
        w, v = scipy.linalg.eigh(self.h0)
        propagator_f = v @ np.diag(np.exp(-1j*w*self.dt)) @ v.T
        propagator_b = v @ np.diag(np.exp(1j*w*self.dt)) @ v.T
        
        propagator = np.einsum("ab,cd->acbd", propagator_f, propagator_b).reshape(4,4)
        
        node0 = np.einsum("ab, bcd -> acd", self.rho[0].reshape(1,-1), branchf)
        node1 = np.einsum("abc,cd -> abd", branchb, propagator)

        mps_list = [node0, node1]
        
        for istep in range(1, self.nsteps):
            mps_list.extend([branchf, node1])
        
        basis = []
        for k in range(self.nsteps):
            basis.append(Ba.BasisHalfSpin(f"s_{k}+"))
            basis.append(Ba.BasisHalfSpin(f"s_{k}-"))
        model = Model(basis, [])
        mps = Mps.from_mp(model, mps_list)
        logger.info(f"system_mps: {mps}")
               
        return mps
    

    def influence_mps(self):
        
        ham_terms = []
        for k in range(self.nsteps):
            for kp in range(k+1):
                if k-kp > self.nmem:
                    continue
                op1 = Op("Z Z", [f"s_{k}+", f"s_{kp}+"], self.eta[k, kp])
                op2 = Op("Z Z", [f"s_{k}+", f"s_{kp}-"], -self.eta[k, kp].conj())
                op3 = Op("Z Z", [f"s_{k}-", f"s_{kp}+"], -self.eta[k, kp])
                op4 = Op("Z Z", [f"s_{k}-", f"s_{kp}-"], self.eta[k, kp].conj())
                ham_terms.extend([op1, op2, op3, op4])
        
        basis = []
        for k in range(self.nsteps):
            basis.append(Ba.BasisHalfSpin(f"s_{k}+"))
            basis.append(Ba.BasisHalfSpin(f"s_{k}-"))
        
        model = Model(basis, ham_terms)
        hmpo = Mpo(model)
        mps = Mps.ground_state(model, True, normalize=True)
        
        mps.evolve_config = self.evolve_config
        mps.compress_config = self.compress_config
        if self.evolve_config.is_tdvp and self.evolve_config.method != EvolveMethod.tdvp_ps2:
            mps = mps.expand_bond_dimension(hmpo, coef=1e-10, include_ex=False)
        
        assert self.evolve_config.adaptive

        art_beta = 1
        mps = mps.evolve(hmpo, art_beta/1j, normalize=True)
        logger.info(f"influence_mps: {mps}")

        return mps

    def calc_rho(self):
        self.sys_mps.dump(f"sys_mps_{self.fname}")
        self.inf_mps.dump(f"inf_mps_{self.fname}")
        sys_mps = self.sys_mps.copy()
        inf_mps = self.inf_mps.copy()

        for istep in range(self.nsteps,0,-1):
            if istep != self.nsteps:
                sys_mps = shrink_mps(sys_mps)
                inf_mps = self.shrink_inf_mps(inf_mps)
            self.rho.insert(1, np.squeeze(sys_mps.dot_ob(inf_mps)).reshape(2,2))
        np.save(f"rho_{self.fname}", np.array(self.rho))
    
    def shrink_inf_mps(self, mps):
        mps_list = mps._mp
        tmp = mps_list[-2][:,1,:].dot(mps_list[-1][:,1,:])
        mps_list[-3] = np.einsum("abc,cd->abd", mps_list[-3], tmp)
        basis = mps.model.basis
        del mps_list[-2:]
        del basis[-2:]
    
        return Mps.from_mp(Model(basis,[]), mps_list)

class SecondOrderSBSTopDown(FirstOrderBSTopDown):

    def system_mps(self):
        """
        (rho0, propagator_{i,0 1/2} branchf_0) - (branchb_0, propagator_{0,1,
        1/2}) - (prop_{0,1 1/2}, branchf_1) - (branchb_1, propagator{1,2,1/2})
        --- (branchb_N-1, propagator{N-1,N,1/2}) -
        """

        identity = np.eye(2)
        
        branch_node = np.zeros((2,2,2), dtype=np.complex128)
        for i in range(2):
            branch_node[i,i,i] = 1
        branch = np.tensordot(identity, branch_node, axes=0)
        branchf = branch.transpose(2, 0, 3, 4, 1).reshape(4,2,4) # forward
        branchb = branch.transpose(0, 2, 3, 1, 4).reshape(4,2,4) # backward
        
        # SBM
        w, v = scipy.linalg.eigh(self.h0)
        propagator_f = v @ np.diag(np.exp(-1j*w*self.dt/2)) @ v.T
        propagator_b = v @ np.diag(np.exp(1j*w*self.dt/2)) @ v.T
        
        propagator = np.einsum("ab,cd->acbd", propagator_f, propagator_b).reshape(4,4)
        
        branchf = np.einsum("ab, bcd -> acd", propagator, branchf) 
        branchb = np.einsum("abc, cd-> abd", branchb, propagator) 

        node0 = np.einsum("ab, bcd -> acd", self.rho[0].reshape(1,-1), branchf)

        mps_list = [node0, branchb]
        
        for istep in range(1, self.nsteps):
            mps_list.extend([branchf, branchb])
        
        basis = []
        for k in range(self.nsteps):
            basis.append(Ba.BasisHalfSpin(f"s_{k}+"))
            basis.append(Ba.BasisHalfSpin(f"s_{k}-"))
        model = Model(basis, [])
        mps = Mps.from_mp(model, mps_list)
        logger.info(f"system_mps: {mps}")
               
        return mps


class SecondOrderBSBTopDown(FirstOrderBSTopDown):
    

    def __init__(self, eta, etaN, nsteps, rho0, h0, dt,
            nmem=np.inf, 
            compress_config: CompressConfig = None,
            evolve_config: EvolveConfig = None,
            fname = "",
            ):
        
        self.etaN = etaN
        super().__init__(eta, nsteps, rho0, h0, dt,
            nmem=nmem, 
            compress_config=compress_config,
            evolve_config=evolve_config,
            fname=fname)
            
    def system_mps(self):
        """
        (rho0, branchf_0) - branchb_0 - (prop_{0,1} branchf_1) -
        branchb_1, --- (prop{N-1,N}, branchf_N) - branchb_N
        """

        identity = np.eye(2)
        
        branch_node = np.zeros((2,2,2), dtype=np.complex128)
        for i in range(2):
            branch_node[i,i,i] = 1
        branch = np.tensordot(identity, branch_node, axes=0)
        branchf = branch.transpose(2, 0, 3, 4, 1).reshape(4,2,4) # forward
        branchb = branch.transpose(0, 2, 3, 1, 4).reshape(4,2,4) # backward
        
        # SBM
        w, v = scipy.linalg.eigh(self.h0)
        propagator_f = v @ np.diag(np.exp(-1j*w*self.dt)) @ v.T
        propagator_b = v @ np.diag(np.exp(1j*w*self.dt)) @ v.T
        
        propagator = np.einsum("ab,cd->acbd", propagator_f, propagator_b).reshape(4,4)
        
        node0 = np.einsum("ab, bcd -> acd", self.rho[0].reshape(1,-1), branchf)
        branchf = np.einsum("ab,bcd -> acd", propagator, branchf)

        mps_list = [node0, branchb]
        
        for istep in range(self.nsteps):
            mps_list.extend([branchf, branchb])
        
        basis = []
        for k in range(self.nsteps+1):
            basis.append(Ba.BasisHalfSpin(f"s_{k}+"))
            basis.append(Ba.BasisHalfSpin(f"s_{k}-"))
        model = Model(basis, [])
        mps = Mps.from_mp(model, mps_list)
        logger.info(f"system_mps: {mps}")
               
        return mps
    
    def calc_rho(self):
        self.sys_mps.dump(f"sys_mps_{self.fname}")
        self.inf_mps.dump(f"inf_mps_{self.fname}")
        sys_mps = self.sys_mps.copy()
        inf_mps = self.inf_mps.copy()

        for istep in range(self.nsteps,0,-1):
            if istep != self.nsteps:
                sys_mps = shrink_mps(sys_mps)
                inf_mps = self.shrink_inf_mps(inf_mps)
            
            mpo =  self.last_step_mpo(istep)
            rho = expectation_ob(sys_mps, mpo,
                    self.enlarge_inf_mps(inf_mps.copy()))   

            self.rho.insert(1, np.squeeze(rho).reshape(2,2))
        np.save(f"rho_{self.fname}", np.array(self.rho))

    def enlarge_inf_mps(self, mps):
        
        mps_list = mps._mp
        mo = np.ones((1,2,1), dtype=np.complex128)
        mps_list.extend([mo, mo])
        
        basis = mps.model.basis
        nsite = mps.site_num // 2

        basis.extend([Ba.BasisHalfSpin(f"s_{nsite}+"),
            Ba.BasisHalfSpin(f"s_{nsite}-")])
    
        return Mps.from_mp(Model(basis,[]), mps_list)


    def last_step_mpo(self, cap_step):
        sf = np.array([1,-1], dtype=np.int32)
        sb = np.array([1,-1], dtype=np.int32)
        s_unique = np.unique(sf-sb.reshape(2,-1))
        bond_dim = len(s_unique)
        
        # construct the 0 to N-1 site
        mpo_list = []
        for istep in range(cap_step):
            if istep == 0:
                left_dim = 1
            else:
                left_dim = bond_dim
            mo_merge_f = np.zeros((left_dim, 2, 2, bond_dim), dtype=np.complex128)
            mo_merge_b = np.zeros((bond_dim, 2, 2, bond_dim), dtype=np.complex128)
            
            for i, s_value in enumerate(s_unique):
                mof = np.diag(np.exp(-s_value*self.etaN[cap_step,istep]*sf))
                mob = np.diag(np.exp(s_value*self.etaN[cap_step,istep].conj()*sb))
                if istep == 0:
                    mo_merge_f[0,:,:,i] = mof
                else:
                    mo_merge_f[i,:,:,i] = mof
                mo_merge_b[i,:,:,i] = mob
        
            mpo_list.extend([mo_merge_f,mo_merge_b])
            

        # last_site
        mo_last = np.zeros((bond_dim, 2, 2), dtype=np.complex128) 
        eta = self.etaN[cap_step,cap_step]
        for jf in range(2):
            for jb in range(2):
                coeff = np.exp(-(sf[jf]-sb[jb])*(eta*sf[jf]-eta.conj()*sb[jb]))
                mo_last[np.where(s_unique==sf[jf]-sb[jb]), jf, jb] = coeff
        
        q, r = scipy.linalg.qr(mo_last.reshape(2*bond_dim,-1), mode="economic")
        mpo_list.extend([self.add_diag_axis(q.reshape(bond_dim, 2, -1)),
            self.add_diag_axis(r.reshape(-1,2,1))])
        
        basis = []
        for k in range(cap_step+1):
            basis.append(Ba.BasisHalfSpin(f"s_{k}+"))
            basis.append(Ba.BasisHalfSpin(f"s_{k}-"))
        model = Model(basis, [])
        mpo = Mpo.from_mp(model, mpo_list)
        #logger.info(f"last_step_mpo: {mpo}")
               
        return mpo
    

    def add_diag_axis(self, mat):
        shape = list(mat.shape)
        shape.insert(2, mat.shape[1])
        new_mat = np.zeros(shape, dtype=mat.dtype)
        for i in range(mat.shape[1]):
            new_mat[:,i,i,:] = mat[:,i,:]
        return new_mat


class FirstOrderBSTopDown_v2:
    """
    SBM
    GU = Ub(dt) Us(dt)
    rho_0|s0><s0|Us|s1><s1|Us|s2>....<sN-1|Us|sN>  Ub(s0)Ub(s1)....Ub(SN-1)
    """
    def __init__(self, eta, nsteps, rho0, h0, dt,
            nmem=np.inf, 
            compress_config: CompressConfig = None,
            evolve_config: EvolveConfig = None,
            fname = "",
            ):

        self.eta = eta
        self.nsteps = nsteps
        self.rho = [rho0]
        self.h0 = h0
        self.dt = dt
        self.nmem = nmem #memory length

        if compress_config is None:
            self.compress_config = CompressConfig()
        else:
            self.compress_config = compress_config
        
        if evolve_config is None:
            self.evolve_config = EvolveConfig()
        else:
            self.evolve_config = evolve_config
        
        self.fname = fname
        # SBM
        w, v = scipy.linalg.eigh(self.h0)
        self.propagator_f = v @ np.diag(np.exp(-1j*w*self.dt)) @ v.T
        self.propagator_b = v @ np.diag(np.exp(1j*w*self.dt)) @ v.T
        
        self.mps = self.calc_mps()
        

    def calc_mps(self):
        """
        (rho0, branchf_0) - (branchb_0, propagator_{0,1}) - branchf_1 -
        (branchb_1, propagator{1,2}) --- (branchb_N-1), propagator{N-1,N} -
        """

        identity = np.eye(2)
        
        branch_node = np.zeros((2,2,2), dtype=np.complex128)
        for i in range(2):
            branch_node[i,i,i] = 1
        branch = np.tensordot(identity, branch_node, axes=0)
        branchf = branch.transpose(2, 0, 3, 4, 1).reshape(4,2,4) # forward
        branchb = branch.transpose(0, 2, 3, 1, 4).reshape(4,2,4) # backward
        
        propagator = np.einsum("ab,cd->acbd", self.propagator_f, self.propagator_b).reshape(4,4)
        
        node0 = np.einsum("ab, bcd -> acd", self.rho[0].reshape(1,-1), branchf)
        node1 = np.einsum("abc,cd -> abd", branchb, propagator)

        mps_list = [node0, node1]
        
        for istep in range(1, self.nsteps):
            if istep == self.nsteps-1:
                mps_list.extend([branchf, branchb])
            else:
                mps_list.extend([branchf, node1])
        
        q,r = scipy.linalg.qr(propagator.reshape(8,2), mode="economic")
        mps_list.extend([q.reshape(4,2,-1), r.reshape(-1,2,1)])

        basis = []
        for k in range(self.nsteps+1):
            basis.append(Ba.BasisHalfSpin(f"s_{k}+"))
            basis.append(Ba.BasisHalfSpin(f"s_{k}-"))
        
        model = Model(basis, [])
        mps = Mps.from_mp(model, mps_list)#.normalize("mps_and_coeff")
        logger.info(f"pi_mps0: {mps}")

        ham_terms = []
        for k in range(self.nsteps):
            for kp in range(k+1):
                if k-kp > self.nmem:
                    continue
                op1 = Op("Z Z", [f"s_{k}+", f"s_{kp}+"], self.eta[k, kp])
                op2 = Op("Z Z", [f"s_{k}+", f"s_{kp}-"], -self.eta[k, kp].conj())
                op3 = Op("Z Z", [f"s_{k}-", f"s_{kp}+"], -self.eta[k, kp])
                op4 = Op("Z Z", [f"s_{k}-", f"s_{kp}-"], self.eta[k, kp].conj())
                ham_terms.extend([op1, op2, op3, op4])
        
        model = Model(basis, ham_terms)
        hmpo = Mpo(model)
        
        mps.evolve_config = self.evolve_config
        mps.compress_config = self.compress_config
        if self.evolve_config.is_tdvp and self.evolve_config.method != EvolveMethod.tdvp_ps2:
            mps = mps.expand_bond_dimension(hmpo, coef=1e-10, include_ex=False)
        
        assert self.evolve_config.adaptive

        art_beta = 1
        mps = mps.evolve(hmpo, art_beta/1j, normalize=False)
        logger.info(f"pi_mps: {mps}")

        return mps

    def calc_rho(self):
        self.mps.dump(f"mps_{self.fname}")
        mps = self.mps.copy()

        for istep in range(self.nsteps,0,-1):
            if istep != self.nsteps:
                mps = self.shrink_mps(mps)
            mode = ["all" for i in range(istep)]*2 + [None]*2
            self.rho.insert(1, np.squeeze(contract_single_mps(mps,mode)).reshape(2,2))
        np.save(f"rho_{self.fname}", np.array(self.rho))
    
    def shrink_mps(self, mps):
        mps_list = mps._mp
        tmp = mps_list[-2][:,1,:].dot(mps_list[-1][:,1,:])
        mps_list[-3] = np.einsum("abc,cd->abd", mps_list[-3], tmp)
        mps_list[-3] = oe.contract("abc,b->abc", mps_list[-3],
                1/self.propagator_b[:,1])
        mps_list[-4] = oe.contract("abc,b->abc", mps_list[-4],
                1/self.propagator_f[:,1])
        
        basis = mps.model.basis
        del mps_list[-2:]
        del basis[-2:]
    
        return Mps.from_mp(Model(basis,[]), mps_list)


