import streamlit as st
import numpy as np
import time
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Option Pricing Calculator",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Option Pricing Calculator")

# ==================== Import calculation functions ====================
try:
    from function import black_scholes, monte_carlo_option_price, finite_difference_price
    st.success("✅ Calculation engines loaded successfully!")
except ImportError as e:
    st.error(f"❌ Failed to import calculation functions: {e}")
    st.stop()

# ==================== Sidebar - Parameter Input ====================
with st.sidebar:
    st.header("⚙️ Parameter Settings")
    
    # Parameter inputs
    S = st.number_input("Current Stock Price (S)", value=100.0, min_value=0.1, step=10.0)
    K = st.number_input("Strike Price (K)", value=100.0, min_value=0.1, step=10.0)
    T = st.number_input("Time to Maturity (years)", value=1.0, min_value=0.01, max_value=50.0, step=0.25)
    r = st.number_input("Risk-free Rate (%)", value=5.0, min_value=0.0, max_value=50.0, step=0.5) / 100
    sigma = st.number_input("Volatility (%)", value=20.0, min_value=0.1, max_value=200.0, step=5.0) / 100
    
    st.markdown("---")
    
    # Output type selection
    st.subheader("📋 Output Selection")
    show_call = st.checkbox("Show Call Option Price", value=True)
    show_put = st.checkbox("Show Put Option Price", value=True)
    
    # Calculation method selection
    st.subheader("🔧 Calculation Engine Selection")
    use_bsm = st.checkbox("BSM Analytical Solution", value=True)
    use_mc = st.checkbox("Monte Carlo Simulation", value=True)
    use_fd = st.checkbox("Finite Difference Method", value=True)
    
    # Advanced parameters
    with st.expander("Advanced Parameters"):
        if use_mc:
            n_simulations = st.slider("Monte Carlo Simulations", 1000, 1000000, 50000, step=5000)
        if use_fd:
            Nt = st.number_input("Finite Difference Time Steps (Nt)", value=500, min_value=10, max_value=5000, step=100)
            Ns = st.number_input("Finite Difference Price Steps (Ns)", value=100, min_value=10, max_value=1000, step=50)
    
    # Calculate button
    st.markdown("---")
    calculate_button = st.button("🚀 Calculate", type="primary", use_container_width=True)

# ==================== Main Display Area ====================
if calculate_button:
    # Check at least one method is selected
    if not (use_bsm or use_mc or use_fd):
        st.error("Please select at least one calculation method!")
        st.stop()
    
    # Check at least one output is selected
    if not (show_call or show_put):
        st.error("Please select at least one option type to display!")
        st.stop()
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Store results
    results = {}
    
    # 1. Calculate BSM
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
            progress_bar.progress(33)
        except Exception as e:
            st.error(f"BSM calculation error: {e}")
    
    # 2. Calculate Monte Carlo
    if use_mc:
        status_text.text("Running Monte Carlo simulation...")
        try:
            start_time = time.time()
            call_price, put_price, call_se, put_se = monte_carlo_option_price(
                S, K, T, r, sigma, n_simulations
            )
            computation_time = time.time() - start_time
            
            results['Monte Carlo'] = {
                'call_price': call_price,
                'put_price': put_price,
                'call_se': call_se,
                'put_se': put_se,
                'time': computation_time,
                'n_simulations': n_simulations
            }
            progress_bar.progress(66)
        except Exception as e:
            st.error(f"Monte Carlo calculation error: {e}")
    
    # 3. Calculate Finite Difference
    if use_fd:
        status_text.text("Running Finite Difference calculation...")
        try:
            start_time = time.time()
            call_price, put_price, S_grid, V_call, V_put = finite_difference_price(
                S, K, T, r, sigma, Nt, Ns
            )
            computation_time = time.time() - start_time
            
            results['Finite Difference'] = {
                'call_price': call_price,
                'put_price': put_price,
                'time': computation_time,
                'Nt': Nt,
                'Ns': Ns
            }
            progress_bar.progress(100)
        except Exception as e:
            st.error(f"Finite Difference calculation error: {e}")
    
    status_text.text("Calculation completed!")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    # ==================== Display Results ====================
    st.markdown("---")
    st.header("📈 Calculation Results")
    
    # Display parameters
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
    
    # Display price results
    if show_call:
        st.subheader("Call Option Price Comparison")
        call_data = []
        for method, result in results.items():
            note = f"±{1.96*result.get('call_se', 0):.4f}" if 'call_se' in result else "-"
            call_data.append({
                'Calculation Method': method,
                'Price': result['call_price'],
                'Computation Time (s)': result['time'],
                'Note': note
            })
        
        df_call = pd.DataFrame(call_data)
        st.dataframe(df_call.style.format({
            'Price': '{:.4f}',
            'Computation Time (s)': '{:.4f}'
        }), use_container_width=True)
    
    if show_put:
        st.subheader("Put Option Price Comparison")
        put_data = []
        for method, result in results.items():
            note = f"±{1.96*result.get('put_se', 0):.4f}" if 'put_se' in result else "-"
            put_data.append({
                'Calculation Method': method,
                'Price': result['put_price'],
                'Computation Time (s)': result['time'],
                'Note': note
            })
        
        df_put = pd.DataFrame(put_data)
        st.dataframe(df_put.style.format({
            'Price': '{:.4f}',
            'Computation Time (s)': '{:.4f}'
        }), use_container_width=True)
    
    # Difference analysis (if multiple methods)
    if len(results) > 1 and 'BSM Analytical' in results:
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
                    'Call Option Difference': call_diff,
                    'Put Option Difference': put_diff,
                    'Call Relative Difference (%)': call_rel_diff,
                    'Put Relative Difference (%)': put_rel_diff
                })
        
        if diff_data:
            df_diff = pd.DataFrame(diff_data)
            st.dataframe(df_diff.style.format({
                'Call Option Difference': '{:.4f}',
                'Put Option Difference': '{:.4f}',
                'Call Relative Difference (%)': '{:.2f}',
                'Put Relative Difference (%)': '{:.2f}'
            }), use_container_width=True)
    
    # Calculation suggestions
    st.markdown("---")
    st.subheader("💡 Calculation Suggestions")
    
    if len(results) == 1:
        method = list(results.keys())[0]
        st.info(f"Currently using only **{method}**. Consider enabling multiple methods for comparison and validation.")
    else:
        # Find the fastest calculation method
        fastest_method = min(results.items(), key=lambda x: x[1]['time'])[0]
        st.info(f"**{fastest_method}** is the fastest. BSM analytical solution is the most precise, Monte Carlo is suitable for complex derivatives, and Finite Difference is ideal for American options.")

else:
    # Initial state display
    st.markdown("---")
    st.info("👈 Please set parameters on the left, select calculation methods and output types, then click 'Calculate'")
    
    # Display examples
    with st.expander("📚 Example Usage"):
        st.markdown("""
        1. **Basic Usage**:
           - Use default parameters
           - Check all calculation methods and output types
           - Click "Calculate"
        
        2. **Quick Verification**:
           - S=100, K=100, T=1 year
           - r=5%, σ=20%
           - Check only BSM analytical solution
           - Results should be: Call≈10.45, Put≈5.57
        
        3. **Precision Test**:
           - Check Monte Carlo simulation
           - Increase simulations to 1 million
           - Observe if results converge to BSM
        """)

# Footer
st.markdown("---")
st.caption("🎓 Option Pricing Calculator | Using BSM, Monte Carlo, Finite Difference Methods")
