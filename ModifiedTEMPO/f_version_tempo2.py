import numpy as np
import scipy
import logging
import itertools
import logging
import sys 
sys.path.append('/home/xiaoyu/HelloWorld/venv/lib/python3.12/site-packages/Renormalizer_20240823')
logger = logging.getLogger("renormalizer")
from mps_compress import mps_compress
from renormalizer.utils.constant import *
from tnpi_2 import InfluenceFunctionalCoeff
import gc
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
from renormalizer.mps.lib import _sum



class tempo:
    def __init__(self, h0, dt, rho0, eta, nsteps, chi, problem = "sbm",  dump_res = False, nmems = None,compress_config: CompressConfig = None,):
        self.h0 = h0
        self.rho0 = rho0
        self.dt = dt
        self.problem = problem
        self.eta = eta
        
        self.nsteps = nsteps
        self.chi = chi#加上这一部分的原因是为了执行tempo中间mps过度的压缩
        self.dump_res = dump_res
        self.ndim = self.h0.shape[0]
        self.rho = [rho0]
        self.sys_mps_cache = None
        self.compress_config = compress_config
        self.nmems = nmems  # 记忆长度（可选）
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
        basis = []
        
        m = len(self.ADT_indices(a))
        for k in range(1,m+1):
            basis.append(Ba.BasisMultiElectron([f"{i}_{k}+" for i in range(self.ndim)], [0,]*self.ndim))
            basis.append(Ba.BasisMultiElectron([f"{i}_{k}-" for i in range(self.ndim)], [0,]*self.ndim))
 
        model = Model(basis, [])
        mps = Mps.from_mp(model, mps_list)
        logger.info(f"system_mps: {mps}")

        return mps
    
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
        indices = self.ADT_indices(n)
        a,b = mposs[0][-2],mposs[0][-1]

        a = oe.contract('abcd->abd',a)
        b = oe.contract('abcd->abd',b)
        del mposs[0][-2:]
        mposs[0].extend([a,b])

        '''
        basis = []
        m = len(self.ADT_indices(n))

        for k in range(1,m+1):
            basis.append(Ba.BasisMultiElectron([f"{i}_{k}+" for i in range(self.ndim)], [0,]*self.ndim))
            basis.append(Ba.BasisMultiElectron([f"{i}_{k}-" for i in range(self.ndim)], [0,]*self.ndim))
 
        model = Model(basis, [])
        env_mps = [Mps.from_mp(model,mposs[0])]
        mposs = [Mpo.from_mp(model, x) for x in mposs[1:]]
        env_mps.compress_config = self.compress_config
      
        new_mpos = [_sum(mpos[3*i:3*i+3], compress=False) for i in range(self.ndim)]
        for i in range(self.ndim):
                mps = new_mpos[i].contract(mps)
            logger.info(f"system_mps: {mps}")
        '''
        return mposs
    


    
    def system_envolve(self, n):
        
        tenosrs = self.influence_func_mpo2(n)
        ndim = self.ndim
        chi = self.chi

        if n == 1:
            basis = []
            for k in range(1,n+1):
                basis.append(Ba.BasisMultiElectron([f"{i}_{k}+" for i in range(self.ndim)], [0,]*self.ndim))
                basis.append(Ba.BasisMultiElectron([f"{i}_{k}-" for i in range(self.ndim)], [0,]*self.ndim))

            model = Model(basis, [])
            env_mps = Mps.from_mp(model,tenosrs[0])
            env_mps.compress_config = self.compress_config

            mposs = [Mpo.from_mp(model, x) for x in tenosrs[1:]]
            #print(len(mposs))
            for i in range(ndim - 1):
                env_mps = mposs[i].contract(env_mps)

            #env_mps = env_mps.normalize(kind="mps_only")#只对mps范数进行归一
            
            self.env_mps = env_mps
            
            
        else:
            assert n>1
            ini_mps1 = self.env_mps.copy()._mp #褪去所在的类

            b = tenosrs[0]

            ini_mps2 = tenosrs[0][-2:]
            mps_0 = []
            for i in range(len(tenosrs[0]) - 2):
                a = oe.contract('abc,debf->adecf',ini_mps1[i],b[i]).reshape(ini_mps1[i].shape[0]*b[i].shape[0],b[i].shape[1],b[i].shape[3]*ini_mps1[i].shape[2])
                mps_0.append(a)
            mps_0.extend(ini_mps2)

            a = mps_compress(mps_0,chi)
            mps_a = a.contraction_mps()#因为这里压缩完成后就是一个列表
            assert len(mps_0) == len(tenosrs[1])

            basis = []
            for k in range(1,n+1):
                basis.append(Ba.BasisMultiElectron([f"{i}_{k}+" for i in range(self.ndim)], [0,]*self.ndim))
                basis.append(Ba.BasisMultiElectron([f"{i}_{k}-" for i in range(self.ndim)], [0,]*self.ndim))

            model = Model(basis, [])
            env_mps = Mps.from_mp(model,mps_a)
            env_mps.compress_config = self.compress_config

            mposs = [Mpo.from_mp(model, x) for x in tenosrs[1:]]
            for i in range(ndim - 1):
                env_mps = mposs[i].contract(env_mps)

            #env_mps = env_mps.normalize(kind="mps_only")
            
            self.env_mps = env_mps

        logger.info(f"env_mps: {env_mps}")

        return env_mps#这里的错误是因为return语句没有对整齐，导致返回值只有n>1的情况，所以的主意习惯
        

    def contract_mps_to_density_matrix(self):
        
        '''
        系统和环境的mps(正负分开的情况下)缩合为一个密度矩阵
        '''
        self.rho = []
        N = self.nsteps
        ndim = self.ndim

        for step in range(1 , N+1):
            if step == 1:
                sys_mps = self.system_mps(step)
                calc_mps = self.system_envolve(step)

                #how to deal with |s3+s3->? sys_mps[-1],mps[-2]
                e0 = np.eye(1)
                tmp1 = sys_mps[-1].reshape(sys_mps[-1].shape[:2])
                tmp2 = calc_mps[-1].reshape(calc_mps[-1].shape[:2])
                a = oe.contract("ab,acd,de,bcg,ge->ce",
                    e0,calc_mps[-2],tmp2,sys_mps[-2],tmp1)
                
                #a = a / np.trace(a)
                logger.info(f"{step-1}th rho:{a},{np.trace(a)}")

            else:
                try:
                    sys_mps = self.system_mps(step)
                    calc_mps = self.system_envolve(step)
                except Exception as e:
                    logger.error(f"Error at step {step}: {e}")
                    #raise e 会立即中断当前流程，将异常向上传递，导致后面的 break 无法执行（属于 “不可达代码”）。
                    return self.process() # 关键：异常时返回部分结果
                    

                    
                e0 = np.eye(1)
                num = 0
                for mt1, mt2 in zip(sys_mps, calc_mps):
                    if num == len(sys_mps)-2:
                        break
                    else:
                        e0 = np.tensordot(e0, mt2, (0,0))
                        e0 = np.tensordot(e0, mt1, ([0, 1], [0, 1]))
                        num += 1

                #how to deal with |s3+s3->? sys_mps[-1],mps[-2]
                tmp1 = sys_mps[-1].reshape(sys_mps[-1].shape[:2])
                tmp2 = calc_mps[-1].reshape(calc_mps[-1].shape[:2])
                a = oe.contract("ab,acd,de,bcg,ge->ce",
                    e0,calc_mps[-2],tmp2,sys_mps[-2],tmp1)
                #a = a / np.trace(a)
                logger.info(f"{step-1}th rho:{a},{np.trace(a)}")


            self.rho.append(a)

        return self.process()

    def process(self):
        
        rho = self.rho
        chi = self.chi

        if self.dump_res == True:
            dump_dict = {
                "rho": rho,
            }
            np.savez(f"{chi}_rho",**dump_dict)
