from renormalizer.mps.backend import np, xp, OE_BACKEND
from renormalizer.mps.matrix import asxp, asnumpy, tensordot
from renormalizer import Mps, Mpo, Model
from renormalizer.model import Model, Op, basis as Ba
from renormalizer.utils import CompressConfig, CompressCriteria, EvolveConfig, EvolveMethod
from tnpi_2 import contract_single_mps

import logging
import scipy.linalg
import opt_einsum as oe
import os
#from pympler import asizeof

logger = logging.getLogger("renormalizer")

class firstorder_IMTNPI:
    #the initial definition of the procedure
    #first order or second order?
    
    def __init__(self,rho0,eta,nsteps,h0,dt,
                compress_config,
                evolve_config,
                test_model="sbm",
                insteps=1,
                mpo_thresh=1e-7,
                hmpo=None,
                save_sparse=True):
        
        self.eta = eta
        self.nsteps = nsteps
        self.insteps = insteps
        self.h0 = h0
        self.dt = dt
        self.test_model = test_model
        #self.nmems=nmems
        self.rho = [rho0]
        self.evolve_times= [0]
        self.ndim = self.h0.shape[0]
        #self.influence_mps=influence_mps
        self.sys_mps = None
        self.entropy = []
        self.mpo_thresh = mpo_thresh
        self.hmpo = hmpo
        self.rho_1,self.propagator,self.n_propagator,self.n_rho_1 = self.system_mps_one()
        self.save_sparse = save_sparse

        if compress_config is None:
            self.compress_config = CompressConfig()
        else:
            self.compress_config = compress_config

        if evolve_config is None:
            self.evolve_config = EvolveConfig()
        else:
            self.evolve_config = evolve_config

    def system_mps_one(self):
        ndim = self.ndim
        ndim_2 = ndim**2

        w, v = scipy.linalg.eigh(self.h0)
        propagator_f = v @ np.diag(np.exp(-1j*w*self.dt)) @ v.T
        propagator_b = v @ np.diag(np.exp(1j*w*self.dt)) @ v.T
        propagator = np.einsum("ab,cd->acbd", propagator_f, propagator_b).reshape(ndim_2, ndim_2)
        rho_reshaped = self.rho[0].reshape(1, ndim_2)
        rho_1=np.einsum('ab,bc -> ac',rho_reshaped, propagator)

        d2 = ndim ** 2  # 简化变量名
        dtype = np.complex128
        branch = np.eye(d2, dtype=dtype)[:, :, None] * np.eye(d2, dtype=dtype)[None, :, :]

        n_propagator = np.einsum("abc,cd->abd", branch, propagator, optimize=True)
        n_rho_1 = np.einsum('ab,bcd->acd', rho_1, branch, optimize=True)

        return rho_1,propagator,n_propagator,n_rho_1
    
    def system_mps(self, a: int):
        '''
        构造系统部分的mps,根据n来判断演化过程中需要构造的mps的长度
        如输入4,则输出为4个系统部分的传播子的mps
        加速优化：向量化构造对角张量 + 优化张量缩并
        '''
        ndim = self.ndim
        d2 = ndim ** 2  
        assert a >= 1
        rho_1 = self.rho_1
        propagator = self.propagator
        n_propagator = self.n_propagator
        n_rho_1 = self.n_rho_1

        if a == 1:
            sys_mps = [rho_1]
        else:
            sys_mps = [n_rho_1] + [n_propagator]*(a-2) + [propagator.reshape(d2, d2, 1)]
        
        return sys_mps
    
    def system_evolution(self):
        ham_terms = []
        ham_guess = [] # only for initial state guess 
        print("model",self.test_model)
        #print("self.nstep",self.nsteps)

        for k in range(1,self.nsteps+1):
            #print("k",k)
            for kp in range(1,k+1):
                if self.test_model == "sbm":
                    op1 = Op("Z Z", [f"s_{k}+", f"s_{kp}+"], self.eta[k, kp])
                    op2 = Op("Z Z", [f"s_{k}+", f"s_{kp}-"], -self.eta[k, kp].conj())
                    op3 = Op("Z Z", [f"s_{k}-", f"s_{kp}+"], -self.eta[k, kp])
                    op4 = Op("Z Z", [f"s_{k}-", f"s_{kp}-"], self.eta[k, kp].conj())
                    ops = [op1, op2, op3, op4]
                elif self.test_model == "holstein":
                    ops = []
                    for i in range(self.ndim):
                        op1 = Op("n n", [f"{i}_{k}+",f"{i}_{kp}+"],self.eta[k-kp, 0])
                        op2 = Op("n n", [f"{i}_{k}+",f"{i}_{kp}-"],-self.eta[k-kp, 0].conj())
                        op3 = Op("n n", [f"{i}_{k}-",f"{i}_{kp}+"],-self.eta[k-kp, 0])
                        op4 = Op("n n", [f"{i}_{k}-",f"{i}_{kp}-"],self.eta[k-kp, 0].conj())       
                        ops.extend([op1, op2, op3, op4])
                ham_terms.extend(ops)
                
                if k-kp <= 1:
                    ham_guess.extend(ops)

        basis = []
        for k in range(1,self.nsteps+1):
            if self.test_model == "sbm":
                basis.append(Ba.BasisHalfSpin(f"s_{k}+"))
                basis.append(Ba.BasisHalfSpin(f"s_{k}-"))
            elif self.test_model == "holstein":
                basis.append(Ba.BasisMultiElectron([f"{i}_{k}+" for i in range(self.ndim)], [0,]*self.ndim))
                basis.append(Ba.BasisMultiElectron([f"{i}_{k}-" for i in range(self.ndim)], [0,]*self.ndim))

        model = Model(basis, ham_terms)
        
        hmpo_guess = Mpo(model, ham_guess)
        logger.info(f"hmpo_guess: {hmpo_guess}")
        
        # compress mpo 

        if self.hmpo is None:
            hmpo = Mpo(model, diagonal=True)
            logger.info(f"hmpo: {hmpo}")
            hmpo = Mps.from_mp(model, hmpo._mp)
            #hmpo.dump(f"{self.nsteps}_hmpo", save_sparse=self.save_sparse)
        else:
            if isinstance(self.hmpo, str):
                hmpo = Mps.load(model, self.hmpo, save_sparse=self.save_sparse)
            else:
                assert isinstance(self.hmpo, Mps)
                hmpo = self.hmpo
            logger.info(f"loading hmpo: {hmpo}")
        
        if self.mpo_thresh != 0:
            compress_config = CompressConfig(CompressCriteria.threshold,
                    threshold=self.mpo_thresh)
            hmpo_old = hmpo.copy()
            hmpo.compress_config = compress_config
            hmpo.canonicalise(equal_distr=True).compress(equal_distr=True,check_normalize=False)
            logger.info(f"hmpo after compression: {hmpo}")
            logger.info(f"distance: {hmpo.distance(hmpo_old)/hmpo.mp_norm}")
            del hmpo_old
       
        hmpo = Mpo.from_mps(hmpo)
    
        if self.test_model == "sbm":
            mps = Mps.ground_state(model, True, normalize=True)
        elif self.test_model == "holstein":
            condition = {f"0_{k}+":np.ones(self.ndim) for k in range(1,self.nsteps+1)}
            condition.update({f"0_{k}-":np.ones(self.ndim) for k in range(1,self.nsteps+1)})
            mps = Mps.ground_state(model, True, normalize=True, condition=condition)
        
        mps.evolve_config = self.evolve_config
        mps.compress_config = self.compress_config

        if self.evolve_config.is_tdvp and self.evolve_config.method != EvolveMethod.tdvp_ps2:
            mps = mps.expand_bond_dimension(hmpo_guess, coef=1e-10, include_ex=False)
    
        
        art_beta = 1
        for istep in range(self.insteps):
            mps = mps.evolve(hmpo, art_beta/1j/self.insteps, normalize=False)
            logger.info(f"mps: {mps}")
        #self.entropy = mps.calc_bond_entropy()
        name_num = self.dt*self.nsteps
        #mps.dump(f"{self.nsteps}_mps")
        print("entropy",self.entropy)
        self.sys_mps = mps
        logger.info(f"new_mps: {mps}")
        return mps
    
    def inf_mpo(self, env_mps, step=None):
        env_mps = env_mps.copy()._mp
        logger.info("start to construct env_mpo")

        if step == self.nsteps:
            L = len(env_mps)
        else:
            assert step >= 0
            L = step * 2
            env_mps = self.back_env_mps(env_mps, step)

        assert L % 2 == 0, "MPS length must be multiple of 2"

        N = L // 2
        emps_list = []

        env_mps = [xp.asarray(t) for t in env_mps]

        for i in range(N):
            A0, A1 = env_mps[2*i : 2*(i+1)]
            bra = oe.contract(
                'ldm,mrn->ldrn',
                A0,
                A1,
                backend=OE_BACKEND,     
                optimize='optimal'   
            )

            Dl = bra.shape[0]
            d_in = A0.shape[1] * A1.shape[1]
            Dmid = bra.shape[3]
            bra = bra.reshape(Dl, d_in, Dmid)
            emps_list.append(bra)

        return emps_list

    def back_env_mps(self,env_mps,step):
        '''
        回推得到整个的环境的pt-mpo;
        '''
        nsteps = self.nsteps

        site_num = (nsteps-step+1)*-2
        tale_sum = env_mps[-1][:,0,:]
        for i in range(-2,site_num+1,-1):#-2,-3,...,site_num+2
            tale_sum = np.tensordot(env_mps[i][:,0,:],tale_sum,axes=(1,0))

        tmp2 = oe.contract("abc,cd->abd",env_mps[site_num+1],tale_sum)
        calc_mps = []    
        for i in range(step*2 - 1):
            calc_mps.append(env_mps[i])
        calc_mps.append(tmp2)
        
        return calc_mps
        
    
    def process(self):
        rhos = self.rho
        self.sys_mps = self.system_evolution()
        for istep in range(1,self.nsteps+1):
            rho = self.calc_rho_back1(istep)
            rho = rho / np.trace(rho)
            logger.info(f"rho at step {istep}: {rho}")
            rhos.append(rho)
        return rhos

    def calc_rho_back1(self, step):
        sys_mps = self.system_mps(step)
        env_mps = self.inf_mpo(self.sys_mps, step)
        ndim = self.ndim

        assert len(sys_mps) == len(env_mps), "MPS长度不匹配"
        N = len(sys_mps)

        sys_mps = [xp.asarray(t) for t in sys_mps]
        env_mps = [xp.asarray(t) for t in env_mps]
        e0 = xp.eye(1)

        for i, (mt1, mt2) in enumerate(zip(sys_mps, env_mps)):
            if i == N - 1:
                break
            e0 = xp.tensordot(e0, mt2, axes=(0, 0))
            e0 = xp.tensordot(e0, mt1, axes=([0, 1], [0, 1]))

        tmp1 = sys_mps[-1].reshape(sys_mps[-1].shape[:2])
        tmp2 = env_mps[-1].reshape(env_mps[-1].shape[:2])

        rho = oe.contract("ab,ac,bc->c",\
                          e0,tmp2,tmp1,backend=OE_BACKEND).reshape(ndim, ndim)

        return asnumpy(rho)
    
   
