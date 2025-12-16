"""
期权定价计算函数库
包含三种定价方法，与核心代码完全一致
"""

import numpy as np
from scipy.stats import norm
import time

# ==================== BSM解析解 ====================
def black_scholes(S, K, T, r, sigma):
    # 1. 计算d1和d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # 2. 计算看涨期权价格
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    # 3. 计算看跌期权价格  
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return call_price, put_price
# ==================== 蒙特卡洛模拟 ====================

def monte_carlo_option_price(S, K, T, r, sigma, n_simulations=1000000):

    # 1. 生成随机股价路径（一次生成，call/put共用，提升效率）
    z = np.random.standard_normal(n_simulations)  # 正态随机数
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    
    # 2. 分别计算看涨/看跌到期收益
    call_payoff = np.maximum(ST - K, 0)  # 看涨收益
    put_payoff = np.maximum(K - ST, 0)   # 看跌收益
    
    # 3. 贴现求当前价格
    call_price = np.exp(-r * T) * np.mean(call_payoff)
    put_price = np.exp(-r * T) * np.mean(put_payoff)
    
    # 4. 计算标准误差
    call_se = np.std(call_payoff) / np.sqrt(n_simulations)
    put_se = np.std(put_payoff) / np.sqrt(n_simulations)
    
    return call_price, put_price, call_se, put_se

# ==================== 有限差分法 ====================
def finite_difference_price(S0, K, T, r, sigma, Nt=1000, Ns=100): 
    # 1. 网格参数（满足CFL条件：dt ≤ dS²/(σ²Smax² + r dS)，保证稳定）
    dt = T / Nt
    Smax = 3 * K
    dS = Smax / Ns
    S_grid = np.linspace(0, Smax, Ns+1)
    
    # 验证CFL稳定性条件（关键！显式法必须满足）
    cfl_condition = dt <= (dS**2) / (sigma**2 * Smax**2 + r * dS)
    if not cfl_condition:
        raise ValueError(f"步长不稳定！需增大Nt（当前Nt={Nt}），或减小Ns（当前Ns={Ns}）")
    
    # 2. 初始化价格网格
    V_call = np.zeros((Ns+1, Nt+1))
    V_put = np.zeros((Ns+1, Nt+1))
    
    # 3. 到期日边界条件
    V_call[:, Nt] = np.maximum(S_grid - K, 0)
    V_put[:, Nt] = np.maximum(K - S_grid, 0)
    
    # 4. 预计算显式差分系数
    alpha = np.zeros(Ns+1)
    beta = np.zeros(Ns+1)
    gamma = np.zeros(Ns+1)
    for i in range(Ns+1):
        S = S_grid[i]
        alpha[i] = 0.5 * dt * (sigma**2 * i**2 - r * i)   # 下系数
        beta[i]  = 1 - dt * (sigma**2 * i**2 + r)        # 中系数
        gamma[i] = 0.5 * dt * (sigma**2 * i**2 + r * i)   # 上系数
    
    # 5. 时间逆序计算（显式法：直接逐点计算，无需解方程）
    for j in range(Nt-1, -1, -1):
        # 计算看涨期权
        for i in range(1, Ns):  # 跳过边界点
            V_call[i, j] = alpha[i] * V_call[i-1, j+1] + beta[i] * V_call[i, j+1] + gamma[i] * V_call[i+1, j+1]
        # 看涨边界条件
        V_call[0, j] = 0  # S=0，看涨价值为0
        V_call[Ns, j] = Smax - K * np.exp(-r * (T - j*dt))  # S→∞
        
        # 计算看跌期权
        for i in range(1, Ns):  # 跳过边界点
            V_put[i, j] = alpha[i] * V_put[i-1, j+1] + beta[i] * V_put[i, j+1] + gamma[i] * V_put[i+1, j+1]
        # 看跌边界条件
        V_put[0, j] = K * np.exp(-r * (T - j*dt))  # S=0
        V_put[Ns, j] = 0  # S→∞，看跌价值为0
    
    # 6. 插值得到S0对应的价格
    call_price = np.interp(S0, S_grid, V_call[:, 0])
    put_price = np.interp(S0, S_grid, V_put[:, 0])
    
    return call_price, put_price, S_grid, V_call, V_put

