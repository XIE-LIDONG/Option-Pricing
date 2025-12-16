import streamlit as st
import numpy as np
import time
import pandas as pd

st.set_page_config(
    page_title="Option Pricing Calculator",
    page_icon="📊",
    layout="wide"
)

# Title + Signature (right-aligned)
st.markdown(
    """
    <div style='display: flex; justify-content: space-between; align-items: center;'>
        <h1 style='margin: 0;'>📊 Option Pricing Calculator</h1>
        <p style='margin: 0;'>By XIE LI DONG</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ==================== Import calculation functions ====================
try:
    # Import European functions
    from function import black_scholes, monte_carlo_option_price, finite_difference_price
    # Import American functions
    from function import binomial_american, trinomial_american, lsm_american_fixed, american_option_fdm
    st.success("✅ Calculation engines loaded successfully!")
except ImportError as e:
    st.error(f"❌ Failed to import calculation functions: {e}")
    st.stop()

# ==================== Sidebar - Parameter Input ====================
with st.sidebar:
    st.header("⚙️ Parameter Settings")
    
    # Core parameters
    S = st.number_input("Current Stock Price (S)", value=100.0, min_value=0.1, step=10.0)
    K = st.number_input("Strike Price (K)", value=100.0, min_value=0.1, step=10.0)
    T = st.number_input("Time to Maturity (years)", value=1.0, min_value=0.01, max_value=50.0, step=0.25)
    r = st.number_input("Risk-free Rate (%)", value=5.0, min_value=0.0, max_value=50.0, step=0.5) / 100
    sigma = st.number_input("Volatility (%)", value=20.0, min_value=0.1, max_value=200.0, step=5.0) / 100
    
    st.markdown("---")
    
    # Option Type (American/European) - Core new feature
    st.subheader("📌 Option Type")
    option_style = st.radio(
        "Select Option Style",
        ("European Option", "American Option"),
        horizontal=True
    )
    
    st.markdown("---")
    
    # Output type selection
    st.subheader("📋 Output Selection")
    show_call = st.checkbox("Show Call Option Price", value=True)
    show_put = st.checkbox("Show Put Option Price", value=True)
    
    # Calculation method selection (dynamic based on option style)
    st.subheader("🔧 Calculation Engine Selection")
    if option_style == "European Option":
        # European methods (original)
        use_bsm = st.checkbox("BSM Analytical Solution", value=True)
        use_mc = st.checkbox("Monte Carlo Simulation (European)", value=True)
        use_fd = st.checkbox("Finite Difference (European)", value=True)
        # Hide American methods
        use_binomial = use_trinomial = use_lsm = use_am_fd = False
    else:
        # American methods (new)
        use_binomial = st.checkbox("Binomial Tree (American)", value=True)
        use_trinomial = st.checkbox("Trinomial Tree (American)", value=True)
        use_lsm = st.checkbox("LSM Monte Carlo (American)", value=True)
        use_am_fd = st.checkbox("Finite Difference (American)", value=True)
        # Hide European methods
        use_bsm = use_mc = use_fd = False
    
    # Advanced parameters (dynamic)
    with st.expander("Advanced Parameters"):
        if option_style == "European Option":
            if use_mc:
                n_simulations = st.slider("Monte Carlo Simulations", 1000, 1000000, 50000, step=5000)
            if use_fd:
                Nt = st.number_input("Finite Difference Time Steps (Nt)", value=500, min_value=10, max_value=5000, step=100)
                Ns = st.number_input("Finite Difference Price Steps (Ns)", value=100, min_value=10, max_value=1000, step=50)
        else:
            # American advanced params
            if use_binomial:
                binomial_n = st.slider("Binomial Tree Steps", 50, 500, 100, step=10)
            if use_trinomial:
                trinomial_n = st.slider("Trinomial Tree Steps", 20, 200, 50, step=10)
            if use_lsm:
                lsm_paths = st.slider("LSM Monte Carlo Paths", 10000, 100000, 20000, step=5000)
                lsm_steps = st.slider("LSM Time Steps", 20, 100, 50, step=10)
            if use_am_fd:
                am_Nt = st.number_input("American FD Time Steps (Nt)", value=2000, min_value=100, max_value=5000, step=200)
                am_Ns = st.number_input("American FD Price Steps (Ns)", value=100, min_value=50, max_value=1000, step=50)
    
    # Calculate button
    st.markdown("---")
    calculate_button = st.button("🚀 Calculate", type="primary", use_container_width=True)

# ==================== Main Display Area ====================
if calculate_button:
    # Check at least one output is selected
    if not (show_call or show_put):
        st.error("Please select at least one option type to display!")
        st.stop()
    
    # Check at least one method is selected
    selected_methods = []
    if option_style == "European Option":
        if use_bsm: selected_methods.append("BSM")
        if use_mc: selected_methods.append("Monte Carlo")
        if use_fd: selected_methods.append("Finite Difference")
    else:
        if use_binomial: selected_methods.append("Binomial Tree")
        if use_trinomial: selected_methods.append("Trinomial Tree")
        if use_lsm: selected_methods.append("LSM Monte Carlo")
        if use_am_fd: selected_methods.append("American Finite Difference")
    
    if not selected_methods:
        st.error(f"Please select at least one calculation method for {option_style}!")
        st.stop()
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Store results
    results = {}
    total_steps = len(selected_methods)
    current_step = 0
    
    # ==================== European Option Calculations ====================
    if option_style == "European Option":
        # 1. BSM Analytical
        if use_bsm:
            status_text.text("Calculating BSM analytical solution...")
            try:
                start_time = time.time()
                call_price, put_price = black_scholes(S, K, T, r, sigma)
                computation_time = time.time() - start_time
                
                results['BSM Analytical'] = {
                    'call_price': call_price,
                    'put_price': put_price,
                    'time': computation_time
                }
                current_step += 1
                progress_bar.progress(current_step / total_steps)
            except Exception as e:
                st.error(f"BSM calculation error: {e}")
        
        # 2. Monte Carlo (European)
        if use_mc:
            status_text.text("Running Monte Carlo simulation (European)...")
            try:
                start_time = time.time()
                call_price, put_price, call_se, put_se = monte_carlo_option_price(
                    S, K, T, r, sigma, n_simulations
                )
                computation_time = time.time() - start_time
                
                results['Monte Carlo (European)'] = {
                    'call_price': call_price,
                    'put_price': put_price,
                    'call_se': call_se,
                    'put_se': put_se,
                    'time': computation_time,
                    'n_simulations': n_simulations
                }
                current_step += 1
                progress_bar.progress(current_step / total_steps)
            except Exception as e:
                st.error(f"Monte Carlo calculation error: {e}")
        
        # 3. Finite Difference (European)
        if use_fd:
            status_text.text("Running Finite Difference calculation (European)...")
            try:
                start_time = time.time()
                call_price, put_price, S_grid, V_call, V_put = finite_difference_price(
                    S, K, T, r, sigma, Nt, Ns
                )
                computation_time = time.time() - start_time
                
                results['Finite Difference (European)'] = {
                    'call_price': call_price,
                    'put_price': put_price,
                    'time': computation_time,
                    'Nt': Nt,
                    'Ns': Ns
                }
                current_step += 1
                progress_bar.progress(current_step / total_steps)
            except Exception as e:
                st.error(f"Finite Difference calculation error: {e}")
    
    # ==================== American Option Calculations ====================
    else:
        # 1. Binomial Tree (American)
        if use_binomial:
            status_text.text("Calculating Binomial Tree (American)...")
            try:
                start_time = time.time()
                # American call (no early exercise advantage for non-dividend stock)
                call_price = binomial_american(S, K, T, r, sigma, 'call', binomial_n)
                put_price = binomial_american(S, K, T, r, sigma, 'put', binomial_n)
                computation_time = time.time() - start_time
                
                results['Binomial Tree (American)'] = {
                    'call_price': call_price,
                    'put_price': put_price,
                    'time': computation_time,
                    'steps': binomial_n
                }
                current_step += 1
                progress_bar.progress(current_step / total_steps)
            except Exception as e:
                st.error(f"Binomial Tree calculation error: {e}")
        
        # 2. Trinomial Tree (American)
        if use_trinomial:
            status_text.text("Calculating Trinomial Tree (American)...")
            try:
                start_time = time.time()
                call_price = trinomial_american(S, K, T, r, sigma, 'call', trinomial_n)
                put_price = trinomial_american(S, K, T, r, sigma, 'put', trinomial_n)
                computation_time = time.time() - start_time
                
                results['Trinomial Tree (American)'] = {
                    'call_price': call_price,
                    'put_price': put_price,
                    'time': computation_time,
                    'steps': trinomial_n
                }
                current_step += 1
                progress_bar.progress(current_step / total_steps)
            except Exception as e:
                st.error(f"Trinomial Tree calculation error: {e}")
        
        # 3. LSM Monte Carlo (American)
        if use_lsm:
            status_text.text("Running LSM Monte Carlo (American)...")
            try:
                start_time = time.time()
                call_price = lsm_american_fixed(S, K, T, r, sigma, 'call', lsm_paths, lsm_steps)
                put_price = lsm_american_fixed(S, K, T, r, sigma, 'put', lsm_paths, lsm_steps)
                computation_time = time.time() - start_time
                
                results['LSM Monte Carlo (American)'] = {
                    'call_price': call_price,
                    'put_price': put_price,
                    'time': computation_time,
                    'paths': lsm_paths,
                    'steps': lsm_steps
                }
                current_step += 1
                progress_bar.progress(current_step / total_steps)
            except Exception as e:
                st.error(f"LSM Monte Carlo calculation error: {e}")
        
        # 4. Finite Difference (American)
        if use_am_fd:
            status_text.text("Running Finite Difference (American)...")
            try:
                start_time = time.time()
                call_price = american_option_fdm(S, K, T, r, sigma, 'call', am_Nt, am_Ns)
                put_price = american_option_fdm(S, K, T, r, sigma, 'put', am_Nt, am_Ns)
                computation_time = time.time() - start_time
                
                results['Finite Difference (American)'] = {
                    'call_price': call_price,
                    'put_price': put_price,
                    'time': computation_time,
                    'Nt': am_Nt,
                    'Ns': am_Ns
                }
                current_step += 1
                progress_bar.progress(current_step / total_steps)
            except Exception as e:
                st.error(f"American Finite Difference calculation error: {e}")
    
    # ==================== Finalize Progress ====================
    status_text.text("Calculation completed!")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    # ==================== Display Results ====================
    st.markdown("---")
    st.header(f"📈 {option_style} Calculation Results")
    
    # Display input parameters
    st.subheader("Input Parameters")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Stock Price (S)", f"{S:.2f}")
    with col2:
        st.metric("Strike Price (K)", f"{K:.2f}")
    with col3:
        st.metric("Time to Maturity", f"{T:.2f} years")
    with col4:
        st.metric("Risk-free Rate", f"{r*100:.2f}%")
    with col5:
        st.metric("Volatility", f"{sigma*100:.2f}%")
    
    st.markdown("---")
    
    # Display Call Option Results
    if show_call:
        st.subheader("Call Option Price")
        call_data = []
        for method, result in results.items():
            note = ""
            if 'call_se' in result:
                note = f"±{1.96*result['call_se']:.4f} (95% CI)"
            elif 'steps' in result:
                note = f"Steps: {result['steps']}"
            elif 'Nt' in result:
                note = f"Nt={result['Nt']}, Ns={result['Ns']}"
            
            call_data.append({
                'Calculation Method': method,
                'Price (USD)': result['call_price'],
                'Computation Time (s)': result['time'],
                'Notes': note
            })
        
        df_call = pd.DataFrame(call_data)
        st.dataframe(df_call.style.format({
            'Price (USD)': '{:.4f}',
            'Computation Time (s)': '{:.4f}'
        }), use_container_width=True)
    
    # Display Put Option Results
    if show_put:
        st.subheader("Put Option Price")
        put_data = []
        for method, result in results.items():
            note = ""
            if 'put_se' in result:
                note = f"±{1.96*result['put_se']:.4f} (95% CI)"
            elif 'steps' in result:
                note = f"Steps: {result['steps']}"
            elif 'Nt' in result:
                note = f"Nt={result['Nt']}, Ns={result['Ns']}"
            
            put_data.append({
                'Calculation Method': method,
                'Price (USD)': result['put_price'],
                'Computation Time (s)': result['time'],
                'Notes': note
            })
        
        df_put = pd.DataFrame(put_data)
        st.dataframe(df_put.style.format({
            'Price (USD)': '{:.4f}',
            'Computation Time (s)': '{:.4f}'
        }), use_container_width=True)
    
    # Difference analysis (only for European)
    if option_style == "European Option" and len(results) > 1 and 'BSM Analytical' in results:
        st.markdown("---")
        st.subheader("🔍 Difference from BSM Analytical Solution")
        
        bsm_call = results['BSM Analytical']['call_price']
        bsm_put = results['BSM Analytical']['put_price']
        
        diff_data = []
        for method, result in results.items():
            if method != 'BSM Analytical':
                call_diff = result['call_price'] - bsm_call
                put_diff = result['put_price'] - bsm_put
                
                call_rel_diff = (call_diff / bsm_call * 100) if bsm_call > 0 else 0
                put_rel_diff = (put_diff / bsm_put * 100) if bsm_put > 0 else 0
                
                diff_data.append({
                    'Calculation Method': method,
                    'Call Option Absolute Difference': call_diff,
                    'Put Option Absolute Difference': put_diff,
                    'Call Relative Difference (%)': call_rel_diff,
                    'Put Relative Difference (%)': put_rel_diff
                })
        
        if diff_data:
            df_diff = pd.DataFrame(diff_data)
            st.dataframe(df_diff.style.format({
                'Call Option Absolute Difference': '{:.4f}',
                'Put Option Absolute Difference': '{:.4f}',
                'Call Relative Difference (%)': '{:.2f}',
                'Put Relative Difference (%)': '{:.2f}'
            }), use_container_width=True)

else:
    # Initial state display
    st.markdown("---")
    st.info("👈 Please set parameters on the left, select calculation methods and output types, then click 'Calculate'")
    
    # Display examples (English)
    with st.expander("📚 Example Usage"):
        st.markdown("""
        ### 1. Basic European Option Calculation
        - Use default parameters (S=100, K=100, T=1 year, r=5%, σ=20%)
        - Select "BSM Analytical Solution"
        - Results: Call ≈10.45, Put ≈5.57
        
        ### 2. American Put Option Test
        - Select "American Option"
        - Set K=108 (In-the-money put)
        - Select "Binomial Tree (American)"
        - Result: Put ≈10.60 (higher than European put ≈9.8)
        
        ### 3. Convergence Test (Monte Carlo)
        - Select "European Option" + "Monte Carlo Simulation"
        - Increase simulations to 1,000,000
        - Observe results converging to BSM values
        
        ### 4. Deep In-the-money American Put
        - Set K=127 (S=100)
        - All American methods should return ≈27.0 (immediate exercise value)
        """)
