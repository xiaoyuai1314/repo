import numpy as np
import scipy
import logging
from mps_compress import mps_compress,MPSNormalizer
import sys, os
sys.path.append('/home/xiaoyu/HelloWorld/venv/lib/python3.12/site-packages/Renormalizer_20240823')
import itertools
import logging
logger = logging.getLogger("renormalizer")
from renormalizer.utils.constant import *
from tnpi_2 import InfluenceFunctionalCoeff
import gc
import opt_einsum as oe
from renormalizer.mps.matrix import asxp
from renormalizer.mps.backend import np, xp, OE_BACKEND


class tempo:
    def __init__(self, h0, dt, rho0, eta, nsteps, chi, problem = "sbm", mpo_scheme = 1, dump_res = True, nmems = None, precision = np.complex128):

         # precision dtype for complex tensors, and a matching real dtype for logs/exps
        self.dtype = precision
        # choose matching real dtype
        self.real_dtype = np.longdouble
        self.mpo_scheme = mpo_scheme
        self.h0 = np.array(h0, dtype=self.dtype)
        self.rho0 = np.array(rho0, dtype=self.dtype)
        self.dt = self.real_dtype(dt)
        self.problem = problem
        self.eta = np.array(eta, dtype=self.dtype)
        self.nsteps = nsteps
        self.dump_res = dump_res
        self.ndim = self.h0.shape[0]
        self.log_norm_history = []  # 记录每步 log(norm)
        self.rho = [self.rho0.astype(self.dtype)]
        self.sys_mps_cache = None
        self.nmems = nmems  # 记忆长度（可选）
        self.chi = chi

        self.rho_1,self.propagator = self.system_mps_one()
        self.indexed_by_j = self.ADT_indices()

    def system_mps_one(self):
        ndim = self.ndim
        #print("ndim",ndim)
        ndim_2 = ndim**2
        
        #the propagator for the SBM model
        w, v = scipy.linalg.eigh(self.h0)
        propagator_f = v @ np.diag(np.exp(-1j*w*self.dt)) @ v.T
        propagator_b = v @ np.diag(np.exp(1j*w*self.dt)) @ v.T
        # 构造Liouville空间中的传播子
        propagator = np.einsum("ab,cd->acbd", propagator_f, propagator_b).reshape(ndim_2, ndim_2)
        #单步演化后的密度矩阵
        rho_reshaped = self.rho0.reshape(1, ndim_2)
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

        node0=np.einsum('ab,bc -> ac',self.rho0.reshape(1,ndim_2),propagator)
        node0=np.einsum('ab,bcd -> acd',node0,branch_f)
        node1=np.einsum('ab,bcd -> acd',propagator,branch_f)

        branch_N =np.eye(ndim_2)
        u_N,s_N,vt_N=np.linalg.svd(branch_N.reshape(ndim_3,ndim),full_matrices=0)
        branch_fN=u_N.reshape(ndim_2,ndim,ndim)
        branch_bN=np.diag(s_N).dot(vt_N)
        branch_bN=branch_bN.reshape(ndim,ndim,1)
        nodeN=np.einsum('ab,bcd -> acd',propagator,branch_fN)
        node0_N=np.einsum('ab,bc -> ac',self.rho0.reshape(1,ndim_2),propagator)
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

        for i, mps in enumerate(mps_list):
            if mps.dtype != self.dtype:
                mps_list[i] = mps.astype(self.dtype)

        return mps_list
    
    def ADT_indices(self, time_point = None):
        '''
        以字典形式返回演化过程中的指标索引
        '''
        nsteps = self.nsteps
        all_indices = {}
        for j in range(1, nsteps + 1):
            if self.nmems is not None:
                start_i = max(1, j - self.nmems)
            else:
                start_i = 1
                
            indices = []
            for i in range(start_i, j + 1):
                k = j - i
                indices.append([i, j, k])
            all_indices[j] = indices

        return  all_indices.get(time_point, []) if time_point is not None else all_indices
    
    def add(self, tensors, name = 'mps'):#在 Python 中，函数参数的默认值需要是一个具体的值（如字符串、数字等），不能直接使用未定义的变量。所以得使用字符串的形式
        '''
        I:mps或者mpo的列表
        o:为堆叠后的结果
        '''

        new_mps = []
        if name == 'mps':
            assert len(tensors[0][1].shape) == 3

            first_tensors = [mps[0] for mps in tensors]
            new_mps.append(np.hstack(first_tensors))

            for i in range(1, len(tensors[0]) - 1):
                current_tensors = [mps[i] for mps in tensors]
                
                merged_tensor = current_tensors[0]
                for j in range(1, len(current_tensors)):
                    mta = merged_tensor
                    mtb = current_tensors[j]
                    
                    new_ms = np.zeros(
                        [mta.shape[0] + mtb.shape[0], 
                        mta.shape[1], 
                        mta.shape[2] + mtb.shape[2]],
                        dtype=np.complex128
                    )
                    
                    new_ms[:mta.shape[0], :, :mta.shape[2]] = mta
                    new_ms[mta.shape[0]:, :, mta.shape[2]:] = mtb
                    
                    merged_tensor = new_ms
                
                new_mps.append(merged_tensor)
                
            last_tensors = [mps[-1] for mps in tensors]
            new_mps.append(np.vstack(last_tensors))

        elif name == 'mpo':
            assert len(tensors[0][0].shape) == 4

            first_tensors = [mpo[0] for mpo in tensors]
            new_mps.append(np.concatenate((first_tensors), axis=3))
        
            for i in range(1, len(tensors[0]) - 1):
            
                current_tensors = [mpo[i] for mpo in tensors]
                
                merged_tensor = current_tensors[0]
                for j in range(1, len(current_tensors)):
                    mta = merged_tensor
                    mtb = current_tensors[j]
                    new_ms = np.zeros(
                        [
                            mta.shape[0] + mtb.shape[0],
                            mta.shape[1],
                            mta.shape[1],
                            mta.shape[3] + mtb.shape[3],
                            
                        ],
                        dtype=np.complex128
                    )
                    new_ms[: mta.shape[0], :, :, : mta.shape[3]] = mta[:, :, :, :]
                    new_ms[mta.shape[0] :, :, :, mta.shape[3] :] = mtb[:, :, :, :]

                    merged_tensor = new_ms

                new_mps.append(merged_tensor)

            last_tensors = [mpo[-1] for mpo in tensors]
            new_mps.append(np.concatenate((last_tensors),axis=0))
        
        else:
            assert False
        
        return new_mps


    def influence_func_mpo(self, n = 1):
            '''
            使用renormalizer构造影响泛函的mpos
            '''
            ndim = self.ndim
            mpos = []
            indices = self.ADT_indices(n)
            eta = self.eta

            for idim in range(ndim):
                for deltas in [1,0,-1]:
                    mpo_list = []
                    occ = np.zeros(ndim)
                    occ[idim] = 1
                    for item in indices:
                        i,j,k = item
                        if k != 0:
                            mo_sf = np.diag(np.exp(-deltas*(eta[k,0]*occ))).reshape(1,ndim,ndim,1)
                            mo_sb = np.diag(np.exp(-deltas*(-eta[k,0].conj()*occ))).reshape(1,ndim,ndim,1)
                            mpo_list.extend([mo_sf, mo_sb])

                    mo_srf = np.zeros([1,ndim,ndim,ndim], dtype=np.complex128)
                    mo_srb = np.zeros([ndim,ndim,ndim,1], dtype=np.complex128)
                    for i in range(ndim):
                        mo_srb[i,i,i,0] = 1
                    
                    if deltas == 1:
                        for i in range(ndim):
                            if i != idim:
                                mo_srf[0,idim,idim,i] = np.exp(-eta[0,0])
                    elif deltas == 0:
                        for i in range(ndim):
                            for j in range(ndim):
                                if i != idim and j != idim:
                                    mo_srf[0,i,i,j] = 1
                        mo_srf[0,idim,idim,idim] = 1
                    elif deltas == -1:
                        for i in range(ndim):
                            if i != idim:
                                mo_srf[0,i,i,idim] = np.exp(-eta[0,0].conj())
                    mpo_list.extend([mo_srf, mo_srb])
                    mpos.append(mpo_list)
            new_mpos = [self.add(mpos[3*i:3*i+3], 'mpo') for i in range(ndim)]
                
            return new_mpos

    def influence_func_mpo2(self,n):

        mposs = self.influence_func_mpo(n)
        a,b = mposs[0][-2],mposs[0][-1]

        a = np.einsum('abcd->abd',a)
        b = np.einsum('abcd->abd',b)
        del mposs[0][-2:]
        mposs[0].extend([a,b])

        return mposs
    
    def constraction(self, B, A, a):
        """
        将MPS张量与影响泛函MPO进行收缩
        
        参数:
        B: 影响泛函MPO列表
        A: MPS张量列表
        
        返回:
        收缩后的MPS张量列表
        收缩的对应指标尽量和文献一致，是对物理指标的收缩，对mps和mpo的在收缩过程中的不同位置分类收缩，加上归一化和压缩
        """
        A_mps = []
        chi = self.chi
        N = self.ndim
        assert len(B) - len(A) == 2
        n = min(len(A), len(B))

        for i in range(n):
            new_tensor = np.einsum('abc,debf->adecf',A[i], B[i])
            new_tensor = new_tensor.reshape(A[i].shape[0]*B[i].shape[0], B[i].shape[1], A[i].shape[2] * B[i].shape[3]) 
            A_mps.append(new_tensor)
           
        a,b = B[-2],B[-1]
        A_mps.extend([a,b])

        d = mps_compress(A_mps,chi)
        A_mps = d.contraction_mps(equal_distr=True)

        # ===========================
        
        return A_mps
    
    def constraction_2(self, n, mpo = None, mps = None):
        '''
        I:输入为对应的时间步,如果时间步大于2,传入的是一层mps和一层mpo。收缩完成后压缩,在程序外部进行循环调用.(n时间参数主要是为了n=1的特殊情况)
        P:将每一层的电子态mpo和传入的mps进行收缩,每收缩一层进行一层压缩
        O:单层的MPO
        '''
        chi = self.chi
        mpo_a = []
        if n == 1:
            mpo_n = []
            l = len(mpo)
            assert l == 2
            for i in range(l):
                b = np.einsum('abc,debf->adecf',mps[i],mpo[i]).reshape(mps[i].shape[0]*mpo[i].shape[0],mpo[i].shape[1],mps[i].shape[2]*mpo[i].shape[3])
                mpo_n.append(b)
                
            d = mps_compress(mpo_n,chi)
            mpo_a = d.orthogonalization()
   
        else:
            N = len(mpo)
            assert len(mpo) == len (mps) 
            mpo_n = []
            for a in range(N):
                b = np.einsum('abc,debf->adecf',mps[a],mpo[a]).reshape(mps[a].shape[0]*mpo[a].shape[0],\
                                                                        mpo[a].shape[1],mpo[a].shape[3]*mps[a].shape[2])
                mpo_n.append(b)
            d = mps_compress(mpo_n,chi)
            mpo_a = d.contraction_mps(equal_distr=True)#因为这里压缩完成后就是一个列表

        '''
        normalizer = MPSNormalizer(mpo_a)
        mpo_a, norm = normalizer.normalize(mode="even")
        self.log_norm_history.append(np.log(norm))
        '''

        return mpo_a
    
    
    def system_envolve(self, step = 1, sys_mps_cache  = None):
        '''修改后的单个时间步演化的环境部分的mps'''

        N = self.nsteps
        n = step
       

        if n == 1:
            assert sys_mps_cache is None
            mps = self.influence_func_mpo2(n)[0]

            '''
            normalizer = MPSNormalizer(mps)
            mps, norm = normalizer.normalize(mode="even")
            self.log_norm_history.append(np.log(norm))
            '''

            for i in range(1, self.ndim):
                mpo = self.influence_func_mpo2(n)[i]
                mps = self.constraction_2(n, mpo, mps)
            logger.info(f"1th step/{N}th contraction successful!")

        else:
            env_mps = sys_mps_cache.copy()
            tensor_B = self.influence_func_mpo2(n)[0]
            mps = self.constraction(tensor_B, env_mps, n) 

            for i in range(1,self.ndim):
                mpo = self.influence_func_mpo2(n)[i]
                mps = self.constraction_2(n, mpo, mps)
            logger.info(f"{n}th step/{N}th contraction successful!")

        for i, m in enumerate(mps):
            if m.dtype != self.dtype:
                mps[i] = m.astype(self.dtype)
            
        env_mps = mps
        self.env_mps = env_mps
        assert len(env_mps) == 2 * n
        
        return env_mps
    
    def contract_mps_to_density_matrix(self):
        
        '''
        系统和环境的mps(正负分开的情况下)缩合为一个密度矩阵
        '''
        
        N = self.nsteps
        ndim = self.ndim
        self.last_rho = []#这里尝试捕捉出现错误的密度矩阵，和相关的迹。捕捉后再来验证除法

        for step in range(1 , N + 1):
            try:
                if step == 1:
                    sys_mps = self.system_mps(step)
                    calc_mps = self.system_envolve(step)

                    e0 = np.eye(1, dtype=self.dtype)
                    tmp1 = sys_mps[-1].reshape(sys_mps[-1].shape[:2]).astype(self.dtype)
                    tmp2 = calc_mps[-1].reshape(calc_mps[-1].shape[:2]).astype(self.dtype)
                    a = oe.contract("ab,acd,de,bcg,ge->ce",
                        e0,calc_mps[-2],tmp2,sys_mps[-2],tmp1).astype(self.dtype)

                    self.last_rho.append(a)
                    self.rho.append(a)
                    logger.info(f"{step}th rho:{a},{np.trace(a)}")

                else:
                    sys_mps = self.system_mps(step)
                    calc_mps = self.system_envolve(step , self.env_mps)

                    if len(self.last_rho) == 2:
                        del self.last_rho[0]

                    e0 = np.eye(1, dtype=self.dtype)
                    num = 0
                    for mt1, mt2 in zip(sys_mps, calc_mps):
                        if num == len(sys_mps)-2:
                            break
                        else:
                            e0 = np.tensordot(e0, mt2, (0,0))
                            e0 = np.tensordot(e0, mt1, ([0, 1], [0, 1]))
                            num += 1
                    tmp1 = sys_mps[-1].reshape(sys_mps[-1].shape[:2]).astype(self.dtype)
                    tmp2 = calc_mps[-1].reshape(calc_mps[-1].shape[:2]).astype(self.dtype)
                    a = oe.contract("ab,acd,de,bcg,ge->ce",
                        e0,calc_mps[-2],tmp2,sys_mps[-2],tmp1).astype(self.dtype)

                    self.last_rho.append(a)

                    self.rho.append(a)
                    #logger.info(f"[Step {step}/{N}] ρ trace={np.trace(a):.6e}, log_factor={log_factor:.3f},dtype={a_real.dtype}")
                    logger.info(f"{step}th rho:{a},{np.trace(a)}")

                    assert len(self.last_rho) == 2

            except Exception as e:
                logger.error(f"Error at step {step}: {str(e)}")

                self.process()
                raise  
            
        self.process()
        return  self.rho

    def process(self):

        rho = self.rho
        chi = self.chi
        last_rho = self.last_rho

        if self.dump_res == True:
            dump_dict = dict()
            dump_dict["rho"] = rho
            dump_dict["last_rho"] = last_rho
            np.savez(f"{chi}_dt_rho",**dump_dict)
