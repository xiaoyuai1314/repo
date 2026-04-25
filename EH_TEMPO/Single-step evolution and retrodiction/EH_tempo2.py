import sys, os

sys.path.append('/home/xiaoyu/HelloWorld/venv/lib/python3.12/site-packages/Renormalizer_20240823')
sys.path.append('/home/xiaoyu/HelloWorld/tempo/主程序')
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

logger = logging.getLogger("renormalizer")

class firstorder_IMTNPI:
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
        self.rho = [rho0]
        self.evolve_times= [0]
        self.ndim = self.h0.shape[0]
        self.sys_mps = None
        self.entropy = []
        self.mpo_thresh = mpo_thresh
        self.hmpo = hmpo
        self.rho_1,self.propagator = self.system_mps_one()
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
        return rho_1,propagator

    def system_mps(self, a : int ):
        '''
        构造系统部分的mps,根据n来判断演化过程中需要构造的mps的长度
        通过svd分解将正负传播部分的系统部分的传播子分开
        '''
        assert a>=1
        propagator = self.propagator
        ndim = self.ndim
        ndim_2 = ndim**2
        ndim_3 = ndim**3
        
        branch = np.zeros((ndim_2,ndim_2,ndim_2), dtype=np.complex128)
        for i in range(ndim_2):
            branch[i,i,i]=1
        branch=branch.reshape(ndim_3,ndim_3)
        u,s,vt=np.linalg.svd(branch)
        
        branch_f=u[:,:ndim_2].reshape(ndim_2,ndim,ndim_2)
        branch_b=np.diag(s[:ndim_2]).dot(vt[:ndim_2,:])
        branch_b=branch_b.reshape(ndim_2,ndim,ndim_2)

        node0=np.einsum('ab,bc -> ac',self.rho[0].reshape(1,ndim_2),propagator)
        node0=np.einsum('ab,bcd -> acd',node0,branch_f)
        node1=np.einsum('ab,bcd -> acd',propagator,branch_f)

        branch_N =np.eye(ndim_2)
        u_N,s_N,vt_N=np.linalg.svd(branch_N.reshape(ndim_3,ndim),full_matrices=0)
        branch_fN=u_N.reshape(ndim_2,ndim,ndim)
        branch_bN=np.diag(s_N).dot(vt_N)
        branch_bN=branch_bN.reshape(ndim,ndim,1)
        nodeN=np.einsum('ab,bcd -> acd',propagator,branch_fN)
        node0_N=np.einsum('ab,bc -> ac',self.rho[0].reshape(1,ndim_2),propagator)
        node0_N=np.einsum('ab,bcd -> acd',node0_N,branch_fN)

        if a == 1:
            mps_list= [node0_N,branch_bN]    
        else:
            mps_list = [node0, branch_b]
            for istep in range(2, a + 1):
                if istep != a:
                    mps_list.extend([node1,branch_b])
                else:
                    mps_list.extend([nodeN,branch_bN])
                    
        return mps_list
    
    def system_evolution(self):
        ham_terms = []
        ham_guess = [] 
        print("model",self.test_model)
        print("ndim",self.ndim)

        for k in range(1,self.nsteps+1):
            for kp in range(1,k+1):
                if self.test_model == "sbm":
                    op1 = Op("Z Z", [f"s_{k}+", f"s_{kp}+"], self.eta[k-kp, 0])
                    op2 = Op("Z Z", [f"s_{k}+", f"s_{kp}-"], -self.eta[k-kp, 0].conj())
                    op3 = Op("Z Z", [f"s_{k}-", f"s_{kp}+"], -self.eta[k-kp, 0])
                    op4 = Op("Z Z", [f"s_{k}-", f"s_{kp}-"], self.eta[k-kp, 0].conj())
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
            hmpo = Mps.from_mp(model, hmpo._mp)#这里老师先将hmpo转换为mps格式，为的是希望将先将mps进行一个特殊的压缩方式，再从后面将mpo转换回mpo格式
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
            logger.info(f"distance: {hmpo.distance(hmpo_old)/hmpo.mp_norm}")#这部分代码是计算压缩前后的距离，但是由于压缩前的hmpo_old过于大，导致内层溢出得到nan的数值
            #hmpo.dump(f"{self.nsteps}_hmpo_c")
            del hmpo_old
       
        hmpo = Mpo.from_mps(hmpo)
        
        if self.test_model == "sbm":
            mps = Mps.ground_state(model)
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
        mps.dump(f"{self.nsteps}_mps")
        print("entropy",self.entropy)
        self.sys_mps = mps
        logger.info(f"new_mps: {mps}")
        return mps


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
            rho = self.calc_rho_back(istep)
            rho = rho / np.trace(rho)
            logger.info(f"rho at step {istep}: {rho}")
            rhos.append(rho)
        return rhos


    def calc_rho_back(self, istep):
        '''
        系统和环境的mps(正负分开的情况下)缩合为一个密度矩阵
        '''
        sys_mps = self.system_mps(istep)
        calc_mps = self.back_env_mps(self.sys_mps,istep)
        assert len(sys_mps) == len(calc_mps)
        e0 = xp.eye(1)
        num = 0
        for mt1, mt2 in zip(sys_mps, calc_mps):
            if num == len(sys_mps)-2:
                break
            else:
                e0 = tensordot(e0, mt2, (0,0))
                e0 = tensordot(e0, mt1, ([0, 1], [0, 1]))
                num += 1

        tmp1 = sys_mps[-1].reshape(sys_mps[-1].shape[:2])
        tmp2 = calc_mps[-1].reshape(calc_mps[-1].shape[:2])
        rho = oe.contract("ab,acd,de,bcg,ge->ce",
            e0,asxp(calc_mps[-2]),asxp(tmp2),asxp(sys_mps[-2]),asxp(tmp1), 
            backend=OE_BACKEND)
        
        return asnumpy(rho)
