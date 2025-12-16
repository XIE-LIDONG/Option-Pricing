#欧式期权定价

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

#美式期权定价
def binomial_american(S, K, T, r, sigma, option_type='put', n=100):

    # 1. 计算参数
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))  # 上涨
    d = 1 / u                         # 下跌
    p = (np.exp(r * dt) - d) / (u - d)  # 风险中性概率
    disc = np.exp(-r * dt)  # 贴现因子
    
    # 2. 构建股价树（只存储最后一步）
    stock_tree = np.zeros(n + 1)
    for i in range(n + 1):
        stock_tree[i] = S * (u ** (n - i)) * (d ** i)
    
    # 3. 计算到期日价值
    option_tree = np.zeros(n + 1)
    if option_type == 'call':
        option_tree = np.maximum(stock_tree - K, 0)
    else:  # put
        option_tree = np.maximum(K - stock_tree, 0)
    
    # 4. 反向递推
    for step in range(n - 1, -1, -1):
        for i in range(step + 1):
            # 计算下一期的股价
            stock_price = S * (u ** (step - i)) * (d ** i)
            
            # 继续持有价值
            hold_value = disc * (p * option_tree[i] + (1-p) * option_tree[i+1])
            
            # 立即行权价值
            if option_type == 'call':
                exercise_value = max(stock_price - K, 0)
            else:
                exercise_value = max(K - stock_price, 0)
            
            # 取最大值
            option_tree[i] = max(hold_value, exercise_value)
    
    return option_tree[0]

#三叉数法
def trinomial_american(S, K, T, r, sigma, option_type='put', n=50):
    """
    三叉树法定价美式期权（极简版）
    
    三叉树：每个节点有三种可能（上、平、下）
    比二叉树收敛更快（更少步数达到相同精度）
    """
    # 1. 计算参数
    dt = T / n
    dx = sigma * np.sqrt(3 * dt)  # 价格步长
    
    u = np.exp(dx)      # 上涨因子
    d = 1 / u           # 下跌因子
    m = 1.0             # 平盘因子
    
    # 风险中性概率
    nu = r - 0.5 * sigma**2
    pu = 0.5 * ((sigma**2 * dt + nu**2 * dt**2) / dx**2 + nu * dt / dx)
    pm = 1 - (sigma**2 * dt + nu**2 * dt**2) / dx**2
    pd = 0.5 * ((sigma**2 * dt + nu**2 * dt**2) / dx**2 - nu * dt / dx)
    
    disc = np.exp(-r * dt)  # 贴现因子
    
    # 2. 构建股价数组（三叉树更宽）
    max_moves = 2 * n + 1  # 最多可能的位置数
    stock_prices = np.zeros(max_moves)
    
    # 初始位置在中间
    mid = n
    stock_prices[mid] = S
    
    # 填充股价
    for i in range(1, n + 1):
        # 上涨路径
        stock_prices[mid + i] = stock_prices[mid + i - 1] * u
        # 下跌路径  
        stock_prices[mid - i] = stock_prices[mid - i + 1] * d
    
    # 3. 计算到期日价值
    option_values = np.zeros(max_moves)
    
    if option_type == 'call':
        for i in range(max_moves):
            option_values[i] = max(stock_prices[i] - K, 0)
    else:  # put
        for i in range(max_moves):
            option_values[i] = max(K - stock_prices[i], 0)
    
    # 4. 反向递推
    for step in range(n - 1, -1, -1):
        new_values = np.zeros(max_moves)
        
        for i in range(n - step, n + step + 1):
            if stock_prices[i] == 0:
                continue
                
            # 继续持有价值（考虑三种可能）
            hold_value = disc * (pu * option_values[i+1] + 
                                pm * option_values[i] + 
                                pd * option_values[i-1])
            
            # 立即行权价值
            if option_type == 'call':
                exercise_value = max(stock_prices[i] - K, 0)
            else:
                exercise_value = max(K - stock_prices[i], 0)
            
            # 美式期权：取最大值
            new_values[i] = max(hold_value, exercise_value)
        
        option_values = new_values.copy()
    
    # 返回中间节点的值
    return option_values[mid]


#LSM方法

def lsm_american_fixed(S, K, T, r, sigma, option_type='put', n_paths=50000, n_steps=100):

    dt = T / n_steps
    discount = np.exp(-r * dt)
    
    # 1. 模拟股价路径
    np.random.seed(42)  # 固定随机种子，结果可重现
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S
    
    for t in range(1, n_steps + 1):
        z = np.random.standard_normal(n_paths)
        paths[:, t] = paths[:, t-1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        )
    
    # 2. 计算现金流矩阵（从到期日开始）
    cash_flow = np.zeros((n_paths, n_steps + 1))
    
    # 到期日现金流
    if option_type.lower() == 'call':
        cash_flow[:, -1] = np.maximum(paths[:, -1] - K, 0)
    else:  # put
        cash_flow[:, -1] = np.maximum(K - paths[:, -1], 0)
    
    # 3. 反向递推
    for t in range(n_steps - 1, 0, -1):
        # 找出价内路径
        if option_type.lower() == 'call':
            in_the_money = paths[:, t] > K
        else:  # put
            in_the_money = paths[:, t] < K
        
        if np.sum(in_the_money) > 10:
            # 这些路径的股价
            X = paths[in_the_money, t]
            
            # 计算未来现金流的贴现值
            future_value = np.zeros(np.sum(in_the_money))
            for future_t in range(t + 1, n_steps + 1):
                future_value += cash_flow[in_the_money, future_t] * np.exp(-r * (future_t - t) * dt)
            
            # 回归：使用多项式基函数
            # 对于看跌：通常用 S, S², S³
            # 对于看涨：通常用 (S-K), (S-K)², (S-K)³
            if option_type.lower() == 'call':
                moneyness = X - K
                X_poly = np.column_stack([
                    np.ones_like(X),
                    moneyness,
                    moneyness**2,
                    moneyness**3
                ])
            else:  # put
                moneyness = K - X
                X_poly = np.column_stack([
                    np.ones_like(X),
                    moneyness,
                    moneyness**2,
                    moneyness**3
                ])
            
            try:
                # 最小二乘回归
                beta = np.linalg.lstsq(X_poly, future_value, rcond=None)[0]
                continuation_value = X_poly @ beta
                
                # 立即行权价值
                if option_type.lower() == 'call':
                    immediate_value = np.maximum(X - K, 0)
                else:
                    immediate_value = np.maximum(K - X, 0)
                
                # 比较决定
                should_exercise = immediate_value > continuation_value
                
                # 更新现金流
                cash_flow[in_the_money, t] = np.where(
                    should_exercise,
                    immediate_value,
                    0
                )
                
                # 如果执行，未来现金流清零
                if np.any(should_exercise):
                    exercise_indices = np.where(in_the_money)[0][should_exercise]
                    for future_t in range(t + 1, n_steps + 1):
                        cash_flow[exercise_indices, future_t] = 0
                        
            except Exception as e:
                # 回归失败，跳过
                pass
    
    # 4. 计算期权价格
    option_price = 0
    for t in range(1, n_steps + 1):
        option_price += np.mean(cash_flow[:, t]) * np.exp(-r * t * dt)
    
    return option_price

#差分法

def american_option_fdm(S0, K, T, r, sigma, option_type='call', Nt=2000, Ns=100):
    dt = T / Nt
    Smax = 3 * K
    dS = Smax / Ns
    S_grid = np.linspace(0, Smax, Ns+1)

    # CFL稳定性检查
    cfl_condition = dt <= (dS**2) / (sigma**2 * Smax**2 + r * dS)
    if not cfl_condition:
        raise ValueError(f"步长不稳定！需增大Nt（当前{Nt}）或减小Ns（当前{Ns}）")

    # 初始化价值网格
    V = np.zeros((Ns+1, Nt+1))

    # 到期日边界
    if option_type == 'call':
        V[:, Nt] = np.maximum(S_grid - K, 0)
    else:
        V[:, Nt] = np.maximum(K - S_grid, 0)

    # 差分系数
    alpha = np.zeros(Ns+1)
    beta = np.zeros(Ns+1)
    gamma = np.zeros(Ns+1)
    for i in range(Ns+1):
        alpha[i] = 0.5 * dt * (sigma**2 * i**2 - r * i)
        beta[i]  = 1 - dt * (sigma**2 * i**2 + r)
        gamma[i] = 0.5 * dt * (sigma**2 * i**2 + r * i)

    # 时间逆序迭代（核心：提前行权判断）
    for j in range(Nt-1, -1, -1):
        # 计算持有价值
        for i in range(1, Ns):
            V[i, j] = alpha[i]*V[i-1, j+1] + beta[i]*V[i, j+1] + gamma[i]*V[i+1, j+1]

        # 美式期权：提前行权对比
        if option_type == 'call':
            early_exercise = np.maximum(S_grid - K, 0)
        else:
            early_exercise = np.maximum(K - S_grid, 0)
        V[:, j] = np.maximum(V[:, j], early_exercise)

        # 边界条件
        V[0, j] = early_exercise[0]
        V[Ns, j] = early_exercise[-1]

    # 插值得到当前价格
    option_price = np.interp(S0, S_grid, V[:, 0])
    return option_price


