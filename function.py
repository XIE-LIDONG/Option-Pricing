import numpy as np
from scipy.stats import norm
import time

# ==================== BSM Analytical Solution ====================
def black_scholes(S, K, T, r, sigma):
    # 1. Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # 2. Calculate call option price
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    # 3. Calculate put option price  
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return call_price, put_price

# ==================== Monte Carlo Simulation ====================
def monte_carlo_option_price(S, K, T, r, sigma, n_simulations=1000000):
    # 1. Generate random stock price paths (generated once for both call/put to improve efficiency)
    z = np.random.standard_normal(n_simulations)  # Normal random numbers
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    
    # 2. Calculate payoffs for call/put at maturity separately
    call_payoff = np.maximum(ST - K, 0)  # Call option payoff
    put_payoff = np.maximum(K - ST, 0)   # Put option payoff
    
    # 3. Discount to get current price
    call_price = np.exp(-r * T) * np.mean(call_payoff)
    put_price = np.exp(-r * T) * np.mean(put_payoff)
    
    # 4. Calculate standard error
    call_se = np.std(call_payoff) / np.sqrt(n_simulations)
    put_se = np.std(put_payoff) / np.sqrt(n_simulations)
    
    return call_price, put_price, call_se, put_se

# ==================== Finite Difference Method ====================
def finite_difference_price(S0, K, T, r, sigma, Nt=1000, Ns=100): 
    # 1. Grid parameters (satisfy CFL condition: dt ≤ dS²/(σ²Smax² + r dS) to ensure stability)
    dt = T / Nt
    Smax = 3 * K
    dS = Smax / Ns
    S_grid = np.linspace(0, Smax, Ns+1)
    
    # Verify CFL stability condition (critical! Must be satisfied for explicit method)
    cfl_condition = dt <= (dS**2) / (sigma**2 * Smax**2 + r * dS)
    if not cfl_condition:
        raise ValueError(f"Step size unstable! Need to increase Nt (current Nt={Nt}), or decrease Ns (current Ns={Ns})")
    
    # 2. Initialize price grids
    V_call = np.zeros((Ns+1, Nt+1))
    V_put = np.zeros((Ns+1, Nt+1))
    
    # 3. Boundary conditions at maturity
    V_call[:, Nt] = np.maximum(S_grid - K, 0)
    V_put[:, Nt] = np.maximum(K - S_grid, 0)
    
    # 4. Pre-calculate explicit finite difference coefficients
    alpha = np.zeros(Ns+1)
    beta = np.zeros(Ns+1)
    gamma = np.zeros(Ns+1)
    for i in range(Ns+1):
        S = S_grid[i]
        alpha[i] = 0.5 * dt * (sigma**2 * i**2 - r * i)   # Lower coefficient
        beta[i]  = 1 - dt * (sigma**2 * i**2 + r)        # Middle coefficient
        gamma[i] = 0.5 * dt * (sigma**2 * i**2 + r * i)   # Upper coefficient
    
    # 5. Backward time calculation (explicit method: direct point-wise calculation without solving equations)
    for j in range(Nt-1, -1, -1):
        # Calculate call option
        for i in range(1, Ns):  # Skip boundary points
            V_call[i, j] = alpha[i] * V_call[i-1, j+1] + beta[i] * V_call[i, j+1] + gamma[i] * V_call[i+1, j+1]
        # Call option boundary conditions
        V_call[0, j] = 0  # S=0, call value = 0
        V_call[Ns, j] = Smax - K * np.exp(-r * (T - j*dt))  # S→∞
        
        # Calculate put option
        for i in range(1, Ns):  # Skip boundary points
            V_put[i, j] = alpha[i] * V_put[i-1, j+1] + beta[i] * V_put[i, j+1] + gamma[i] * V_put[i+1, j+1]
        # Put option boundary conditions
        V_put[0, j] = K * np.exp(-r * (T - j*dt))  # S=0
        V_put[Ns, j] = 0  # S→∞, put value = 0
    
    # 6. Interpolate to get price at S0
    call_price = np.interp(S0, S_grid, V_call[:, 0])
    put_price = np.interp(S0, S_grid, V_put[:, 0])
    
    return call_price, put_price, S_grid, V_call, V_put

# American Option Pricing
def binomial_american(S, K, T, r, sigma, option_type='put', n=100):
    # 1. Calculate parameters
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = 1 / u                         # Down factor
    p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability
    disc = np.exp(-r * dt)  # Discount factor
    
    # 2. Build stock price tree (only store the last step)
    stock_tree = np.zeros(n + 1)
    for i in range(n + 1):
        stock_tree[i] = S * (u ** (n - i)) * (d ** i)
    
    # 3. Calculate maturity values
    option_tree = np.zeros(n + 1)
    if option_type == 'call':
        option_tree = np.maximum(stock_tree - K, 0)
    else:  # put
        option_tree = np.maximum(K - stock_tree, 0)
    
    # 4. Backward recursion
    for step in range(n - 1, -1, -1):
        for i in range(step + 1):
            # Calculate stock price at next period
            stock_price = S * (u ** (step - i)) * (d ** i)
            
            # Hold value
            hold_value = disc * (p * option_tree[i] + (1-p) * option_tree[i+1])
            
            # Immediate exercise value
            if option_type == 'call':
                exercise_value = max(stock_price - K, 0)
            else:
                exercise_value = max(K - stock_price, 0)
            
            # Take maximum value
            option_tree[i] = max(hold_value, exercise_value)
    
    return option_tree[0]

# Trinomial Tree Method
def trinomial_american(S, K, T, r, sigma, option_type='put', n=50):
    """
    Trinomial Tree for American Option Pricing (Simplified Version)
    
    Trinomial Tree: Each node has three possibilities (up, flat, down)
    Faster convergence than Binomial Tree (fewer steps for same accuracy)
    """
    # 1. Calculate parameters
    dt = T / n
    dx = sigma * np.sqrt(3 * dt)  # Price step size
    
    u = np.exp(dx)      # Up factor
    d = 1 / u           # Down factor
    m = 1.0             # Flat factor
    
    # Risk-neutral probabilities
    nu = r - 0.5 * sigma**2
    pu = 0.5 * ((sigma**2 * dt + nu**2 * dt**2) / dx**2 + nu * dt / dx)
    pm = 1 - (sigma**2 * dt + nu**2 * dt**2) / dx**2
    pd = 0.5 * ((sigma**2 * dt + nu**2 * dt**2) / dx**2 - nu * dt / dx)
    
    disc = np.exp(-r * dt)  # Discount factor
    
    # 2. Build stock price array (wider for trinomial tree)
    max_moves = 2 * n + 1  # Maximum possible positions
    stock_prices = np.zeros(max_moves)
    
    # Initial position in the middle
    mid = n
    stock_prices[mid] = S
    
    # Fill stock prices
    for i in range(1, n + 1):
        # Up paths
        stock_prices[mid + i] = stock_prices[mid + i - 1] * u
        # Down paths  
        stock_prices[mid - i] = stock_prices[mid - i + 1] * d
    
    # 3. Calculate maturity values
    option_values = np.zeros(max_moves)
    
    if option_type == 'call':
        for i in range(max_moves):
            option_values[i] = max(stock_prices[i] - K, 0)
    else:  # put
        for i in range(max_moves):
            option_values[i] = max(K - stock_prices[i], 0)
    
    # 4. Backward recursion
    for step in range(n - 1, -1, -1):
        new_values = np.zeros(max_moves)
        
        for i in range(n - step, n + step + 1):
            if stock_prices[i] == 0:
                continue
                
            # Hold value (consider three possibilities)
            hold_value = disc * (pu * option_values[i+1] + 
                                pm * option_values[i] + 
                                pd * option_values[i-1])
            
            # Immediate exercise value
            if option_type == 'call':
                exercise_value = max(stock_prices[i] - K, 0)
            else:
                exercise_value = max(K - stock_prices[i], 0)
            
            # American option: take maximum value
            new_values[i] = max(hold_value, exercise_value)
        
        option_values = new_values.copy()
    
    # Return value at middle node
    return option_values[mid]

# LSM Method
def lsm_american_fixed(S, K, T, r, sigma, option_type='put', n_paths=50000, n_steps=100):
    dt = T / n_steps
    discount = np.exp(-r * dt)
    
    # 1. Simulate stock price paths
    np.random.seed(42)  # Fix random seed for reproducibility
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S
    
    for t in range(1, n_steps + 1):
        z = np.random.standard_normal(n_paths)
        paths[:, t] = paths[:, t-1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        )
    
    # 2. Calculate cash flow matrix (start from maturity)
    cash_flow = np.zeros((n_paths, n_steps + 1))
    
    # Cash flow at maturity
    if option_type.lower() == 'call':
        cash_flow[:, -1] = np.maximum(paths[:, -1] - K, 0)
    else:  # put
        cash_flow[:, -1] = np.maximum(K - paths[:, -1], 0)
    
    # 3. Backward recursion
    for t in range(n_steps - 1, 0, -1):
        # Identify in-the-money paths
        if option_type.lower() == 'call':
            in_the_money = paths[:, t] > K
        else:  # put
            in_the_money = paths[:, t] < K
        
        if np.sum(in_the_money) > 10:
            # Stock prices for these paths
            X = paths[in_the_money, t]
            
            # Calculate discounted future cash flows
            future_value = np.zeros(np.sum(in_the_money))
            for future_t in range(t + 1, n_steps + 1):
                future_value += cash_flow[in_the_money, future_t] * np.exp(-r * (future_t - t) * dt)
            
            # Regression: use polynomial basis functions
            # For put: typically use S, S², S³
            # For call: typically use (S-K), (S-K)², (S-K)³
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
                # Least squares regression
                beta = np.linalg.lstsq(X_poly, future_value, rcond=None)[0]
                continuation_value = X_poly @ beta
                
                # Immediate exercise value
                if option_type.lower() == 'call':
                    immediate_value = np.maximum(X - K, 0)
                else:
                    immediate_value = np.maximum(K - X, 0)
                
                # Decision rule
                should_exercise = immediate_value > continuation_value
                
                # Update cash flows
                cash_flow[in_the_money, t] = np.where(
                    should_exercise,
                    immediate_value,
                    0
                )
                
                # If exercised, set future cash flows to zero
                if np.any(should_exercise):
                    exercise_indices = np.where(in_the_money)[0][should_exercise]
                    for future_t in range(t + 1, n_steps + 1):
                        cash_flow[exercise_indices, future_t] = 0
                        
            except Exception as e:
                # Skip if regression fails
                pass
    
    # 4. Calculate option price
    option_price = 0
    for t in range(1, n_steps + 1):
        option_price += np.mean(cash_flow[:, t]) * np.exp(-r * t * dt)
    
    return option_price

# Finite Difference Method (American)
def american_option_fdm(S0, K, T, r, sigma, option_type='call', Nt=2000, Ns=100):
    dt = T / Nt
    Smax = 3 * K
    dS = Smax / Ns
    S_grid = np.linspace(0, Smax, Ns+1)

    # CFL stability check
    cfl_condition = dt <= (dS**2) / (sigma**2 * Smax**2 + r * dS)
    if not cfl_condition:
        raise ValueError(f"Step size unstable! Need to increase Nt (current Nt={Nt}) or decrease Ns (current Ns={Ns})")

    # Initialize value grid
    V = np.zeros((Ns+1, Nt+1))

    # Boundary conditions at maturity
    if option_type == 'call':
        V[:, Nt] = np.maximum(S_grid - K, 0)
    else:
        V[:, Nt] = np.maximum(K - S_grid, 0)

    # Finite difference coefficients
    alpha = np.zeros(Ns+1)
    beta = np.zeros(Ns+1)
    gamma = np.zeros(Ns+1)
    for i in range(Ns+1):
        alpha[i] = 0.5 * dt * (sigma**2 * i**2 - r * i)
        beta[i]  = 1 - dt * (sigma**2 * i**2 + r)
        gamma[i] = 0.5 * dt * (sigma**2 * i**2 + r * i)

    # Backward time iteration (core: early exercise check)
    for j in range(Nt-1, -1, -1):
        # Calculate hold value
        for i in range(1, Ns):
            V[i, j] = alpha[i]*V[i-1, j+1] + beta[i]*V[i, j+1] + gamma[i]*V[i+1, j+1]

        # American option: early exercise comparison
        if option_type == 'call':
            early_exercise = np.maximum(S_grid - K, 0)
        else:
            early_exercise = np.maximum(K - S_grid, 0)
        V[:, j] = np.maximum(V[:, j], early_exercise)

        # Boundary conditions
        V[0, j] = early_exercise[0]
        V[Ns, j] = early_exercise[-1]

    # Interpolate to get current price
    option_price = np.interp(S0, S_grid, V[:, 0])
    return option_price
