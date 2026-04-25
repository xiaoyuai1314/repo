import sys, os

sys.path.append('/home/xiaoyu/HelloWorld/venv/lib/python3.12/site-packages/Renormalizer_20240823')
sys.path.append('/home/xiaoyu/HelloWorld/tempo/主程序')
import numpy as np
import numpy.linalg as LA
from scipy.linalg import svd,qr  # 导入SciPy的SVD函数
import mpmath
import scipy.integrate
from scipy.integrate import quad
import logging
logger = logging.getLogger("renormalizer")


class mps_guess:
    """
    创建一个简单的MPS类，支持生成随机MPS、后续可扩展期望值计算、归一化等功能
    """
    def __init__(self, d, chi, N):
        self.d = d  # 物理指标维度
        self.chi = chi  # 截断维度（键维度）
        self.N = N  # 格点数
        self.tensors = self._generate_random_mps()  # 初始化时生成随机MPS张量列表
    
    def _generate_random_mps(self):
        """生成随机的矩阵乘积态（MPS）张量列表"""
        tensors = [0] * self.N
        # 第一个张量：左边界维度为1
        tensors[0] = np.random.rand(1, self.d, min(self.chi, self.d))
        for k in range(1, self.N):
            prev_right_dim = tensors[k-1].shape[2]
            # 计算当前张量的右维度（截断为chi）
            right_dim = min(
                self.chi,
                prev_right_dim * self.d,
                self.chi ** (self.N - k - 1)
            )
            tensors[k] = np.random.rand(prev_right_dim, self.d, right_dim)
        return tensors


class mps_compress:
    '''
    该类主要用于将给定的一个mps进行虚拟维数的裁剪，先将mps收缩为一个大的张量，然后重新进行
    svd分解压缩。
    '''
    def __init__(self, mps, chi = None):

        self.mps = mps
        self.chi = chi 

    def mps_guess(self, d, chi, N):
        '''
        初猜矩阵乘积态,d为物理指标维数，chi为截断维数，N为格点数,返回一个随机的mps
        '''
        A = [0] * N 
        
        A[0] = np.random.rand(1, d, min(chi, d))
        
        for k in range(1, N):  
            right_dim = min(
                chi, 
                A[k-1].shape[2] * d,  
                chi ** (N - k - 1)
            )
            A[k] = np.random.rand(A[k-1].shape[2], d, right_dim)
        
        return A  # 返回生成的MPS列表

    def tt_product(self):
        """
        Tensor-train product
        :param tensors: tensors in the TT form
        :return: tensor
        """
        mps = self.mps
        x = np.tensordot(mps[0], mps[1], [[mps[0].ndim-1], [0]])
        for n in range(len(mps)-2):
            x = np.tensordot(x, mps[n+2], [[x.ndim - 1], [0]])
        x = np.squeeze(x)#移除维度1

        return x
        
    
    def ttd(self):
        """
        :param x: tensor to be decomposed
        :param chi: dimension cut-off. Use QR decomposition when chi=None;
                    use SVD but don't truncate when chi=-1
        :return tensors: tensors in the TT form
        :return lm: singular values in each decomposition (calculated when chi is not None)
        """
        chi = self.chi
        x = self.tt_product()
        dims = x.shape
        ndim = x.ndim
        dimL = 1
        tensors = list()
        lm = list()
        for n in range(ndim-1):
            if chi is None:  # No truncation
                q, x = np.linalg.qr(x.reshape(dimL*dims[n], -1))
                dimL1 = x.shape[0]
            else:
                q, s, v = np.linalg.svd(x.reshape(dimL*dims[n], -1))
                if chi > 0:
                    dc = min(chi, s.size)
                else:
                    dc = s.size
                q = q[:, :dc]
                s = s[:dc]
                lm.append(s)
                x = np.diag(s).dot(v[:dc, :])
                dimL1 = dc
            tensors.append(q.reshape(dimL, dims[n], dimL1))
            dimL = dimL1
        tensors.append(x.reshape(dimL, dims[-1]))
        tensors[0] = tensors[0][0, :, :]
        return tensors, lm
    

    def orthogonalization(self):
        """从左向右正交化（左规范形式）"""
        mps = [t.copy() for t in self.mps]
        n = len(mps)
        
        for i in range(n - 1):
            Dl, d, Dr = mps[i].shape
            matrix = mps[i].reshape(Dl * d, Dr)
            Q, R = qr(matrix, mode='economic')
            new_shape = (Dl, d, Q.shape[1])
            mps[i] = Q.reshape(new_shape)
            # 把R乘到下一个张量上
            next_tensor = mps[i + 1]
            Dl2, d2, Dr2 = next_tensor.shape
            merged = np.tensordot(R, next_tensor, axes=(1, 0))
            mps[i + 1] = merged.reshape(R.shape[0], d2, Dr2)
        return mps
    

    @staticmethod
    def equal_distribution_svd(matrix, chi_trunc):
        """
        等分布SVD分解：将奇异值拆分为 sqrt(S) * sqrt(S)
        """
        U, S, Vt = svd(matrix, full_matrices=False)
        chi_trunc = min(chi_trunc, len(S))
        U = U[:, :chi_trunc]
        S = S[:chi_trunc]
        Vt = Vt[:chi_trunc, :]

        sqrtS = np.sqrt(S)
        sqrtS_mat = np.diag(sqrtS)

        # 左右均分奇异值，减少数值梯度
        U_eq = U @ sqrtS_mat
        V_eq = sqrtS_mat @ Vt
        return U_eq, V_eq

    def contraction_mps(self, normalize=False, equal_distr=False):
        """
        压缩MPS（右规范化），可选等分布奇异值。
        """
        mps = self.orthogonalization()
        chi = self.chi
        n = len(mps)
        mps_new = [t.copy() for t in mps]

        for i in reversed(range(1, n)):
            Dl, d, Dr = mps_new[i].shape
            matrix = mps_new[i].reshape(Dl, d * Dr)
            chi_trunc = min(chi or matrix.shape[0], matrix.shape[0], matrix.shape[1])

            if equal_distr:
                U_eq, V_eq = self.equal_distribution_svd(matrix, chi_trunc)
                # 右张量
                mps_new[i] = V_eq.reshape(chi_trunc, d, Dr)
                # 左张量乘 U_eq
                prev_tensor = mps_new[i - 1]
                Dl_prev, d_prev, Dr_prev = prev_tensor.shape
                new_prev = np.tensordot(prev_tensor, U_eq, axes=(2, 0))
                mps_new[i - 1] = new_prev
            else:
                U, S, Vt = svd(matrix, full_matrices=False)
                chi_trunc = min(chi_trunc, len(S))
                U = U[:, :chi_trunc]
                S = S[:chi_trunc]
                Vt = Vt[:chi_trunc, :]
                mps_new[i] = Vt.reshape(chi_trunc, d, Dr)
                prev_tensor = mps_new[i - 1]
                new_prev = np.tensordot(prev_tensor, U @ np.diag(S), axes=(2, 0))
                mps_new[i - 1] = new_prev
        logger.info(f"After compression, MPS bond dimensions: {[t.shape for t in mps_new]}")

        if normalize:
            mps_new = self.mps_global_normalize(mps_new)
        return mps_new
     
    
    def mps_global_normalize(self, mps):

        norm_sq = self.mps_inner_product(mps, mps)
        # 取实部，因为模长平方是实数

        norm_sq_real = norm_sq.real  
        norm = np.sqrt(norm_sq_real + 1e-16)  # 1e-16为保护项
        normalized_mps = [tensor.copy() for tensor in mps]

        normalized_mps[0] /= norm  
        return normalized_mps

    def mps_inner_product(self,psi, phi):

        L = len(psi)
        
        # 初始化收缩结果（左环境）：从1x1单位矩阵开始
        contract_result = np.eye(1, dtype=np.complex128)  # 复数类型适配量子态
        
        for i in range(L):
            psi_conj = np.conj(psi[i])
            
            phi_tensor = phi[i]

            contract_result = np.tensordot(contract_result, psi_conj, axes=[1, 0])  
            contract_result = np.tensordot(contract_result, phi_tensor, axes=[[0, 1], [0, 1]])  
        
        # 最终收缩结果为1x1矩阵，取唯一元素作为内积
        return contract_result.item()
    

class MPSNormalizer:
    """
    用于对 MPS 进行全局归一化，同时返回范数。
    """
    def __init__(self, mps):
        self.mps = mps

    def normalize(self, mode="even", eps=1e-300, verbose=False):
        """
        对 MPS 进行归一化，返回 (归一化后的MPS, 范数)
        """
        mps = [A.copy() for A in self.mps]
        norm_sq = self.mps_inner_product(mps, mps)
        norm_sq_real = np.real(norm_sq)
        norm = np.sqrt(norm_sq_real + eps)

        if verbose:
            print(f"[Normalize] Norm = {norm:.6e}")

        L = len(mps)
        if mode == "even":
            scale = norm ** (1.0 / L)
            mps = [A / scale for A in mps]
        elif mode == "first":
            mps[0] /= norm
        else:
            raise ValueError("mode must be 'first' or 'even'")

        return mps, norm

    def mps_inner_product(self, psi, phi):
        """计算 MPS 内积 <psi|phi>。"""
        L = len(psi)
        env = np.eye(1, dtype=np.complex128)
        for i in range(L):
            A_conj = np.conj(psi[i])
            B = phi[i]
            env = np.tensordot(env, A_conj, axes=[1, 0])
            env = np.tensordot(env, B, axes=[[0, 1], [0, 1]])
        return env.squeeze()



class HolsteinErrorAnalyzer:

    def __init__(self, rho_tempo, rho_heom, time_steps):
        self.time_steps = np.asarray(time_steps) 
        self.rho_tempo = rho_tempo
        self.rho_heom = rho_heom
        self.nstates = len(rho_tempo[0])   
        self.P_tempo = self.extract_populations(rho_tempo)
        self.P_heom = self.extract_populations(rho_heom)

        self.sum_abs_diff = np.sum(np.abs(self.P_tempo - self.P_heom), axis=1)

    def extract_populations(self, rho):
        """提取密度矩阵的对角元（布居分布）"""
        return np.array([np.diag(rho_matrix) for rho_matrix in rho])
    
    def integrand_total(self, t_prime):
        """计算积分的被积函数 ∑|Pⁿ(t') - P_refⁿ(t')|"""
        total = 0.0
        for state_idx in range(self.nstates):
            # 插值获取t'时刻的布居
            P_interp = np.interp(t_prime, self.time_steps, self.P_tempo[:, state_idx])
            P_ref_interp = np.interp(t_prime, self.time_steps, self.P_heom[:, state_idx])
            total += np.abs(P_interp - P_ref_interp)
        return total
    
    def calculate_error(self, trapezoid=True):
        """
        计算随时间变化的误差：
        - 梯形法：用梯形公式数值积分∫₀ᵗ ∑|P - P_ref| dt'
        - 积分法：用scipy.integrate.quad精确积分
        最终误差均归一化为 (1/(nstates * t)) * 积分结果
        """
        time_points = self.time_steps
        error_list = []
        
        for i, t_current in enumerate(time_points):
            if t_current == 0:
                error_list.append(0.0)
                continue
            
            if trapezoid:
                # 梯形法：取0到t_current的时间片段
                idx = np.where(time_points <= t_current)[0]  
                t_segment = time_points[idx]
                sum_diff_segment = self.sum_abs_diff[idx]
                integral = np.trapz(sum_diff_segment, t_segment)
            else:
                # 精确积分法
                integral, _ = quad(self.integrand_total, 0, t_current)
            
            error = integral / (self.nstates * t_current)
            error_list.append(error)
        
        return np.array(error_list)
    

class InfluenceFunctionalCoeff:

    """
        初始化影响泛函系数计算器
        
        参数:
        beta: 逆温度参数
        Jw: 谱密度函数
        dt: 时间步长
        wrange: 频率范围 [下限, 上限]
        N: 时间步数
        order: 展开阶数，支持1或2
        two_side: 是否计算正负频率范围的积分
    """
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
          
            return val1+val2, error1+error2
        else:
            val, error = mpmath.quadsubdiv(func, [self.wrange[0], self.wrange[1]], error=True)
       
            val, error = scipy.integrate.quad(
                      func, self.wrange[0], self.wrange[1], complex_func=True,
                      limit=1000, epsabs=self.epsabs, epsrel=self.epsrel)
     
            return val, error