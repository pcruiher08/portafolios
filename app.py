import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import streamlit as st

# === Funciones de portafolio ===

def fetch_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end)

    if df.empty:
        raise ValueError(f"No se descargaron datos para los símbolos {tickers} desde {start} hasta {end}")

    if isinstance(tickers, str):
        tickers = [tickers]

    # Manejar múltiples símbolos
    if len(tickers) > 1:
        if isinstance(df.columns, pd.MultiIndex):
            if 'Adj Close' in df.columns.get_level_values(0):
                prices = df['Adj Close']
            elif 'Close' in df.columns.get_level_values(0):
                prices = df['Close']
            else:
                raise ValueError("Se esperaba 'Adj Close' o 'Close' en el DataFrame de múltiples índices.")
        else:
            if 'Adj Close' in df.columns:
                prices = df['Adj Close']
            elif 'Close' in df.columns:
                prices = df['Close']
            else:
                raise ValueError("Se esperaba 'Adj Close' o 'Close' en el DataFrame para múltiples símbolos.")
    else:
        # Un solo símbolo
        if isinstance(df.columns, pd.MultiIndex):
            if 'Adj Close' in df.columns.get_level_values(0):
                prices = df['Adj Close'].copy()
            elif 'Close' in df.columns.get_level_values(0):
                prices = df['Close'].copy()
            else:
                raise ValueError("Se esperaba 'Adj Close' o 'Close' en el DataFrame de múltiples índices.")
        else:
            # DataFrame plano
            if 'Adj Close' in df.columns:
                prices = df[['Adj Close']].copy()
            elif 'Close' in df.columns:
                prices = df[['Close']].copy()
            else:
                raise ValueError("Se esperaba 'Adj Close' o 'Close' en el DataFrame plano.")
            if prices.shape[1] == len(tickers):
                prices.columns = tickers

    return prices.dropna()


def plot_efficient_frontier(results, max_sharpe, min_vol, mean_returns, cov_matrix, tickers, risk_free_rate, max_allocation=1.0, crypto_indices=None, max_crypto_allocation=1.0, allow_partial_investment=False):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 7))

    # Gráfico de dispersión de portafolios aleatorios
    sc = ax.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', marker='o', s=10, alpha=0.3)
    plt.colorbar(sc, label='Ratio de Sharpe', ax=ax)

    # Resaltar portafolios de máximo Sharpe y mínima volatilidad
    # Need to handle potential partial investment for performance calculation if allow_partial_investment is true
    # The portfolio_performance function already accounts for total_invested_ratio for Sharpe,
    # but for plotting the points, we need to decide if they represent the 'invested' portion
    # or the 'total portfolio' (including cash). For consistency with optimization, we'll
    # use the total portfolio return/volatility.
    max_sharpe_opt_ret, max_sharpe_opt_vol, _ = portfolio_performance(max_sharpe.x, mean_returns, cov_matrix, risk_free_rate)
    min_vol_opt_ret, min_vol_opt_vol, _ = portfolio_performance(min_vol.x, mean_returns, cov_matrix, risk_free_rate)

    # These results from portfolio_performance are for the 'invested' part.
    # To plot them on the same scale as the simulated portfolios (which might have implicit cash),
    # we need to adjust them.
    # Remember the `portfolio_performance` function returns `ret` (invested return) and `vol` (invested vol).
    # `neg_sharpe` and `portfolio_volatility` for optimization now calculate `total_portfolio_return` and `total_portfolio_volatility`.
    # We should use those for plotting the star points to be consistent with what the optimizer targets.

    max_sharpe_ret_total = (max_sharpe_opt_ret * np.sum(max_sharpe.x)) + (risk_free_rate * (1 - np.sum(max_sharpe.x)))
    max_sharpe_vol_total = max_sharpe_opt_vol * np.sum(max_sharpe.x)

    min_vol_ret_total = (min_vol_opt_ret * np.sum(min_vol.x)) + (risk_free_rate * (1 - np.sum(min_vol.x)))
    min_vol_vol_total = min_vol_opt_vol * np.sum(min_vol.x)


    ax.scatter(max_sharpe_vol_total, max_sharpe_ret_total, marker='*', color='r', s=300, label='Máximo Ratio de Sharpe')
    ax.scatter(min_vol_vol_total, min_vol_ret_total, marker='*', color='b', s=300, label='Mínima Volatilidad')


    # Línea de la frontera eficiente
    min_possible_ret = np.min(results[0, :])
    max_possible_ret = np.max(results[0, :])
    target_returns = np.linspace(min_possible_ret, max_possible_ret, 100)


    efficient_vols = []

    num_assets = len(mean_returns)
    bounds = tuple((0, max_allocation) for _ in range(num_assets))
    
    # Conditional sum constraint for the efficient frontier line
    constraints = []
    if allow_partial_investment:
        # If allowed, the total sum of weights should be <= 1
        constraints.append({'type': 'ineq', 'fun': lambda x: 1 - np.sum(x)}) # Sum <= 1
        constraints.append({'type': 'ineq', 'fun': lambda x: np.sum(x)}) # Sum >= 0 (to ensure positive investment)
    else:
        # Otherwise, sum must be exactly 1
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Add crypto constraint if applicable
    if crypto_indices is not None and len(crypto_indices) > 0 and max_crypto_allocation < 1.0:
        def crypto_constraint_ef(x):
            return max_crypto_allocation - np.sum(x[crypto_indices])
        constraints.append({'type': 'ineq', 'fun': crypto_constraint_ef})

    for target in target_returns:
        # For the efficient frontier, we constrain the *return* to be equal to the target.
        # This target is the *total* portfolio return, including cash.
        
        # We define a helper function for the return constraint that accounts for cash
        def return_constraint_ef(x, target=target, mr=mean_returns, rfr=risk_free_rate):
            # Calculate the return of the invested portion
            ret_invested = np.sum(mr * x) * 252
            # Calculate the total portfolio return including cash
            total_invested_ratio = np.sum(x)
            total_portfolio_return = (ret_invested * total_invested_ratio) + (rfr * (1 - total_invested_ratio))
            return total_portfolio_return - target
        
        constraints_target = constraints + [{'type': 'eq', 'fun': return_constraint_ef}]
        
        initial_guess = num_assets * [1. / num_assets]
        
        result = minimize(portfolio_volatility, # This function now calculates total portfolio volatility
                          initial_guess,
                          args=(mean_returns, cov_matrix),
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints_target,
                          options={'maxiter': 1000})
        if result.success:
            efficient_vols.append(result.fun) # portfolio_volatility returns total portfolio volatility
        else:
            efficient_vols.append(np.nan)

    valid_indices = ~np.isnan(efficient_vols)
    ax.plot(np.array(efficient_vols)[valid_indices], target_returns[valid_indices], 'r--', linewidth=2, label='Frontera Eficiente')

    # Etiquetas
    ax.set_xlabel('Volatilidad (Desviación Estándar)')
    ax.set_ylabel('Retornos Esperados')
    ax.set_title('Frontera Eficiente con Restricciones')
    ax.legend()
    st.caption("**Frontera Eficiente**: Conjunto de portafolios que ofrecen el mayor retorno esperado para un nivel dado de riesgo. El punto rojo es el portafolio con mayor ratio de Sharpe, el azul el de menor volatilidad.")

    return fig


def calc_returns_cov(prices):
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return returns, mean_returns, cov_matrix

# Modify portfolio_performance to account for non-100% allocation
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    # This function is used to calculate the performance of the *invested* portion
    # for individual display. The optimizer uses `neg_sharpe` and `portfolio_volatility`
    # which explicitly consider the total portfolio (including cash).
    
    total_invested_ratio = np.sum(weights)
    if total_invested_ratio == 0: # Avoid division by zero if weights are all zero
        return 0, 0, -np.inf # Return negative inf for Sharpe to disincentivize empty portfolios

    ret = np.sum(mean_returns * weights) * 252 # Annualized return of the invested portion
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights))) # Volatility of the invested portion
    
    # Sharpe is calculated for the invested portion for this function,
    # consistent with how it's used in the random simulation results (results[2,:]).
    # The optimization functions (`neg_sharpe`) and display functions (`get_portfolio_metrics`)
    # use the adjusted Sharpe for the total portfolio including cash.
    sharpe = (ret - risk_free_rate) / vol if vol != 0 else -np.inf
    return ret, vol, sharpe

def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    # This should be maximized considering partial investment if allowed
    # We want to maximize the Sharpe of the *total* capital, including cash.
    total_invested_ratio = np.sum(weights)
    
    # Return of the invested portion (annualized)
    ret_invested = np.sum(mean_returns * weights) * 252
    # Volatility of the invested portion (annualized)
    vol_invested = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))

    if vol_invested == 0 and total_invested_ratio > 0: # If there's investment but no volatility, and we target Sharpe, it's problematic
        return np.inf # Penalize zero volatility for invested assets if optimizing Sharpe

    # Total portfolio return including cash
    total_portfolio_return = (ret_invested * total_invested_ratio) + (risk_free_rate * (1 - total_invested_ratio))
    # Total portfolio volatility (assuming cash has zero volatility and no correlation with assets)
    total_portfolio_volatility = vol_invested * total_invested_ratio 

    if total_portfolio_volatility <= 1e-9: # Handle near-zero total volatility (e.g., all cash or very low vol assets)
        if total_portfolio_return > risk_free_rate: # If it beats risk-free with no vol, infinite Sharpe
            return -np.inf
        else: # Otherwise, penalize or treat as zero Sharpe
            return np.inf # For minimization of negative Sharpe, large positive value means bad Sharpe
            
    sharpe = (total_portfolio_return - risk_free_rate) / total_portfolio_volatility
    return -sharpe


def portfolio_volatility(weights, mean_returns, cov_matrix):
    # This should be minimized considering partial investment if allowed
    total_invested_ratio = np.sum(weights)
    vol_invested = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    # Total portfolio volatility
    return vol_invested * total_invested_ratio # Minimize overall portfolio volatility

def optimize_portfolios(mean_returns, cov_matrix, risk_free_rate, max_allocation=1.0, crypto_indices=None, max_crypto_allocation=1.0, allow_partial_investment=False):
    num_assets = len(mean_returns)
    bounds = tuple((0, max_allocation) for _ in range(num_assets))
    
    constraints = []
    if allow_partial_investment:
        # If allowed, the total sum of weights should be <= 1
        constraints.append({'type': 'ineq', 'fun': lambda x: 1 - np.sum(x)}) # Sum <= 1
        constraints.append({'type': 'ineq', 'fun': lambda x: np.sum(x)}) # Sum >= 0 (to ensure positive investment)
    else:
        # Otherwise, sum must be exactly 1
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    if crypto_indices is not None and len(crypto_indices) > 0 and max_crypto_allocation < 1.0:
        def crypto_constraint_opt(x):
            # The sum of all crypto weights must be <= max_crypto_allocation
            return max_crypto_allocation - np.sum(x[crypto_indices])
        constraints.append({'type': 'ineq', 'fun': crypto_constraint_opt})
    
    initial_guess = num_assets * [1. / num_assets]

    max_sharpe = minimize(neg_sharpe, initial_guess,
                          args=(mean_returns, cov_matrix, risk_free_rate),
                          method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'maxiter': 1000})

    min_vol = minimize(portfolio_volatility, initial_guess,
                       args=(mean_returns, cov_matrix),
                       method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 1000})

    return max_sharpe, min_vol

def generate_valid_weights(num_assets, max_allocation, max_attempts=1000):
    for _ in range(max_attempts):
        weights = np.random.uniform(0, max_allocation, num_assets)
        if np.sum(weights) > 0: # Ensure we don't divide by zero
            weights /= np.sum(weights)
        if np.all(weights <= max_allocation + 1e-8): # Add tolerance for float comparison
            return weights
    # Fallback to equal weights if random generation fails
    equal_weight = min(1/num_assets, max_allocation)
    weights = np.array([equal_weight]*num_assets)
    if np.sum(weights) > 0:
        weights /= np.sum(weights)
    return weights

def generate_valid_weights_with_crypto_limit(num_assets, max_allocation, crypto_indices, max_crypto_allocation, max_attempts=5000):
    if len(crypto_indices) == 0:
        return generate_valid_weights(num_assets, max_allocation, max_attempts)

    crypto_indices = np.array(crypto_indices, dtype=int)
    non_crypto_indices = np.array([i for i in range(num_assets) if i not in crypto_indices], dtype=int)

    for attempt in range(max_attempts):
        weights = np.zeros(num_assets)
        
        # 1. Generate random weights for crypto assets, respecting max_crypto_allocation
        crypto_weights = np.random.uniform(0, max_allocation, len(crypto_indices))
        if crypto_weights.sum() > max_crypto_allocation and crypto_weights.sum() > 1e-9: # Only scale if sum is non-zero
            crypto_weights = crypto_weights * (max_crypto_allocation / crypto_weights.sum())
        weights[crypto_indices] = crypto_weights
        
        current_crypto_sum = weights[crypto_indices].sum()

        if len(non_crypto_indices) > 0:
            # 2. Determine remaining allocation for non-crypto assets
            remaining_allocation_for_non_crypto = 1.0 - current_crypto_sum
            
            non_crypto_weights = np.random.uniform(0, max_allocation, len(non_crypto_indices))
            
            if non_crypto_weights.sum() > 0:
                # Scale non-crypto weights to fit the remaining_allocation_for_non_crypto
                # and also respect individual max_allocation.
                scale_factor = remaining_allocation_for_non_crypto / non_crypto_weights.sum()
                
                # Check if scaling will make any individual weight exceed max_allocation
                if np.any(non_crypto_weights * scale_factor > max_allocation + 1e-8):
                    # If so, scale down more aggressively to respect individual max_allocation
                    adjusted_scale_factor = min(scale_factor, max_allocation / np.max(non_crypto_weights))
                    non_crypto_weights = non_crypto_weights * adjusted_scale_factor
                else:
                    non_crypto_weights = non_crypto_weights * scale_factor
            
            weights[non_crypto_indices] = non_crypto_weights
            
            # For mixed portfolios, the sum should still be 1.0 unless crypto limit
            # prevents full allocation. If total sum is still below 1 due to random generation
            # and individual limits, try to normalize what's left.
            if np.sum(weights) < 1.0 - 1e-8 and np.sum(weights) > 0:
                weights /= np.sum(weights) # Normalize to 1.0
                # Re-check crypto constraint after normalization if applicable
                if weights[crypto_indices].sum() > max_crypto_allocation + 1e-8:
                    continue # Try again if crypto limit violated
            
        # If only crypto assets selected and max_crypto_allocation < 1, the sum will be < 1.
        # No further normalization to 1.0 is needed in this specific case.

        # Final checks
        if np.all(weights >= -1e-9) and np.all(weights <= max_allocation + 1e-8): # Weights should be >=0, adjusted tolerance
            current_crypto_sum_final = weights[crypto_indices].sum() if len(crypto_indices) > 0 else 0
            if current_crypto_sum_final <= max_crypto_allocation + 1e-8:
                if np.sum(weights) > 0: # Ensure not all zero
                    return weights
            
    # Fallback if no valid weights found after max_attempts
    st.warning("No se pudieron generar pesos aleatorios válidos. Usando pesos de respaldo.")
    
    weights = np.zeros(num_assets)
    if len(crypto_indices) > 0:
        crypto_portion_per_asset = max_crypto_allocation / len(crypto_indices)
        weights[crypto_indices] = min(crypto_portion_per_asset, max_allocation)
    
    if len(non_crypto_indices) > 0:
        remaining_for_non_crypto = 1.0 - weights[crypto_indices].sum()
        non_crypto_portion_per_asset = remaining_for_non_crypto / len(non_crypto_indices)
        weights[non_crypto_indices] = min(non_crypto_portion_per_asset, max_allocation)
    
    if np.sum(weights) > 0:
        # Normalize to 1.0 if not solely limited crypto assets, otherwise let it be < 1
        if not (len(non_crypto_indices) == 0 and len(crypto_indices) > 0 and max_crypto_allocation < 1.0):
            weights /= np.sum(weights)
        
    # Final check for fallback weights
    if not (np.all(weights >= -1e-9) and np.all(weights <= max_allocation + 1e-8) and \
            (len(crypto_indices) == 0 or weights[crypto_indices].sum() <= max_crypto_allocation + 1e-8)):
        st.error("Los pesos de respaldo tampoco cumplen todas las restricciones estrictamente. Por favor, revisa tus parámetros.")

    if np.sum(weights) == 0 and num_assets > 0: # If fallback leads to all zeros
        weights = np.ones(num_assets) / num_assets # At least assign equal weights to something
        if len(crypto_indices) > 0 and weights[crypto_indices].sum() > max_crypto_allocation + 1e-8:
            weights[crypto_indices] = weights[crypto_indices] * (max_crypto_allocation / weights[crypto_indices].sum())
            if np.sum(weights) > 0:
                weights /= np.sum(weights)


    return weights


def simulate_random_portfolios(mean_returns, cov_matrix, risk_free_rate, num_portfolios=3000, max_allocation=1.0, crypto_indices=None, max_crypto_allocation=1.0):
    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    weights_record = []

    # Determine if only crypto assets are selected
    all_selected_are_crypto = (len(crypto_indices) == num_assets) and (num_assets > 0)

    for i in range(num_portfolios):
        if all_selected_are_crypto and max_crypto_allocation < 1.0:
            # Only crypto assets selected and limited. Weights sum to <= max_crypto_allocation.
            weights = generate_valid_weights_with_crypto_limit(num_assets, max_allocation, crypto_indices, max_crypto_allocation)
        elif crypto_indices is not None and len(crypto_indices) > 0 and max_crypto_allocation < 1.0:
            # Crypto assets mixed with others, or not all assets are crypto. Sum to 1.0 still applies,
            # but crypto sum is limited.
            weights = generate_valid_weights_with_crypto_limit(num_assets, max_allocation, crypto_indices, max_crypto_allocation)
            # Ensure these weights (from mixed mode) sum to 1.0 if not already handled by the function itself
            if np.sum(weights) > 0 and np.abs(np.sum(weights) - 1.0) > 1e-8:
                # If sum is not 1, it means the crypto limit likely caused it, so we don't normalize to 1.
                # But the intention for mixed portfolios in simulate_random_portfolios is usually sum=1.
                # So for consistency in simulation, let's normalize only if the non-crypto sum is not zero and crypto sum allows for it.
                if len(weights) - len(crypto_indices) > 0: # If there are non-crypto assets
                    weights = weights / np.sum(weights) # Force sum to 1 for simulation results to be comparable if mixed
        else:
            # No crypto limit, or no crypto assets. Sum to 1.0.
            weights = generate_valid_weights(num_assets, max_allocation)
        
        # Calculate performance using the adjusted `neg_sharpe` logic for total portfolio metrics
        # For simulation results, we directly use the total portfolio performance for plotting.
        total_invested_ratio = np.sum(weights)
        ret_invested = np.sum(mean_returns * weights) * 252
        vol_invested = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))

        total_portfolio_return = (ret_invested * total_invested_ratio) + (risk_free_rate * (1 - total_invested_ratio))
        total_portfolio_volatility = vol_invested * total_invested_ratio
        
        sharpe = (total_portfolio_return - risk_free_rate) / total_portfolio_volatility if total_portfolio_volatility > 0 else -np.inf

        results[0, i] = total_portfolio_return
        results[1, i] = total_portfolio_volatility
        results[2, i] = sharpe
        weights_record.append(weights) # Store the actual weights

    return results, weights_record

def load_tickers_from_file(filepath="tickers.txt"):
    try:
        with open(filepath, "r") as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        if not tickers:
            st.error(f"No se encontraron símbolos en {filepath}. Por favor, revisa el archivo.")
            st.stop()
        return tickers
    except FileNotFoundError:
        st.error(f"Archivo de símbolos {filepath} no encontrado. Asegúrate de tener un archivo 'tickers.txt' en el mismo directorio que tu aplicación.")
        st.stop()

# Funciones adicionales para nuevas métricas
def calculate_max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown

def calculate_beta(portfolio_returns, benchmark_returns):
    # Ensure returns are aligned by date
    common_index = portfolio_returns.index.intersection(benchmark_returns.index)
    aligned_portfolio_returns = portfolio_returns.loc[common_index]
    aligned_benchmark_returns = benchmark_returns.loc[common_index]

    if len(aligned_portfolio_returns) < 2:
        return np.nan # Not enough data points to calculate beta

    covariance = np.cov(aligned_portfolio_returns, aligned_benchmark_returns)[0, 1]
    benchmark_variance = np.var(aligned_benchmark_returns)
    if benchmark_variance == 0:
        return np.nan # Avoid division by zero
    beta = covariance / benchmark_variance
    return beta

def calculate_calmar_ratio(annual_return, max_drawdown):
    if max_drawdown >= 0 or np.isnan(max_drawdown) or np.isinf(max_drawdown): # Drawdown must be negative for a meaningful ratio
        return np.nan
    return annual_return / abs(max_drawdown)

def calculate_sortino_ratio(returns, risk_free_rate=0.09):
    daily_risk_free_rate = (1 + risk_free_rate)**(1/252) - 1
    
    # Calculate returns below the risk-free rate
    downside_returns = returns[returns < daily_risk_free_rate]
    
    expected_return = returns.mean() * 252 # Annualized return of the series
    downside_std = downside_returns.std() * np.sqrt(252) # Annualized downside deviation

    if downside_std == 0:
        return np.nan
    return (expected_return - risk_free_rate) / downside_std

# === Funciones de Visualización y Métricas para Streamlit ===

def plot_weights_bar(weights, tickers, title="Portfolio Weights"):
    fig, ax = plt.subplots(figsize=(max(6, len(tickers)*0.8), 4))
    ax.bar(tickers, weights*100, color='skyblue')
    ax.set_ylabel("Asignación (%)")
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("**Asignación (%)**: Porcentaje del portafolio invertido en cada activo.")

def plot_pie_chart(weights, tickers, title="Distribución del Portafolio"):
    total_allocated_sum = np.sum(weights)
    
    if total_allocated_sum == 0:
        st.warning("No hay activos asignados para el gráfico circular.")
        return
        
    labels = list(tickers)
    values = list(weights)

    # If there's significant unallocated cash, add a "Cash" slice
    if total_allocated_sum < 1.0 - 1e-8:
        labels.append("Efectivo (No asignado)")
        values.append(1.0 - total_allocated_sum)

    # Filter out labels with zero (or near-zero) weights for cleaner chart
    filtered_labels = [label for i, label in enumerate(labels) if values[i] > 1e-4]
    filtered_values = [value for value in values if value > 1e-4]

    if not filtered_values: # If all values are tiny after filtering
        st.warning("No hay asignaciones significativas para mostrar en el gráfico circular.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(filtered_values, labels=filtered_labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    ax.set_title(title)
    st.pyplot(fig)
    st.caption("**Gráfico de pay**: Muestra la proporción de cada activo en el portafolio (incluyendo efectivo no asignado si aplica).")


def plot_cumulative_returns(portfolio_returns, benchmark_returns):
    fig, ax = plt.subplots(figsize=(8, 4))
    cumulative_portfolio = (1 + portfolio_returns).cumprod()
    cumulative_benchmark = (1 + benchmark_returns).cumprod()
    ax.plot(cumulative_portfolio.index, cumulative_portfolio, label="Portafolio", linewidth=2)
    ax.plot(cumulative_benchmark.index, cumulative_benchmark, label="S&P 500", alpha=0.7, linestyle='--')
    ax.set_title("Retornos Acumulados")
    ax.set_ylabel("Crecimiento de $1")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("**Retornos Acumulados**: Muestra cómo habría crecido una inversión de $1 en el portafolio y el índice de referencia durante el periodo seleccionado.")

def plot_drawdown(returns):
    fig, ax = plt.subplots(figsize=(8, 3))
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.5)
    ax.set_title("Caída Máxima (Drawdown) en el Tiempo")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Fecha")
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("**Drawdown**: La caída máxima desde un pico histórico en el valor del portafolio. Mide la peor pérdida relativa desde un máximo anterior.")

def plot_rolling_volatility(returns, window=21):
    fig, ax = plt.subplots(figsize=(8, 3))
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    ax.plot(rolling_vol.index, rolling_vol, label=f"Volatilidad Móvil {window}-días", color='orange')
    ax.set_title("Volatilidad Móvil en el Tiempo")
    ax.set_ylabel("Volatilidad Anualizada")
    ax.set_xlabel("Fecha")
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("**Volatilidad Móvil**: Desviación estándar anualizada de los retornos en una ventana móvil. Mide la variabilidad del portafolio a lo largo del tiempo.")

def plot_correlation_heatmap(prices):
    corr = prices.pct_change().dropna().corr()
    fig, ax = plt.subplots(figsize=(max(6, len(prices.columns)), max(5, len(prices.columns))))
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='left', fontsize=10)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=10)
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    ax.set_title("Matriz de correlación de retornos", pad=20)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("**Matriz de correlación**: Muestra la relación entre los retornos de los activos seleccionados. Valores cercanos a 1 indican alta correlación, valores cercanos a -1 indican correlación negativa.")

# This function should be defined here, as it's used by display_portfolio and also for CSV export.
def get_portfolio_metrics(weights, tickers, returns_df, benchmark_returns_series, risk_free_rate_val, investing_amount_val):
    # This will now use the adjusted neg_sharpe/portfolio_volatility logic
    # which correctly handles partial investment.
    # The `portfolio_performance` function was adjusted to reflect this.
    
    total_invested_ratio = np.sum(weights)

    # Calculate metrics for the *invested* portion to get `ret_invested` and `vol_invested`
    ret_invested = np.sum(returns_df.mean() * weights) * 252 # Annualized return of the invested portion
    vol_invested = np.sqrt(np.dot(weights.T, np.dot(returns_df.cov() * 252, weights))) # Volatility of the invested portion

    # Calculate actual portfolio return and volatility including cash
    # (Assuming cash earns risk_free_rate and has 0 volatility)
    actual_portfolio_return = (ret_invested * total_invested_ratio) + (risk_free_rate_val * (1 - total_invested_ratio))
    actual_portfolio_volatility = vol_invested * total_invested_ratio
    
    sharpe = (actual_portfolio_return - risk_free_rate_val) / actual_portfolio_volatility if actual_portfolio_volatility > 0 else -np.inf

    # Daily returns of the invested portion
    portfolio_daily_returns_invested = (returns_df * weights).sum(axis=1) 
    
    # When calculating daily returns for performance metrics, if there's cash, that cash portion earns risk_free_rate daily.
    daily_risk_free_rate = (1 + risk_free_rate_val)**(1/252) - 1 # Daily risk-free rate
    total_portfolio_daily_returns = (portfolio_daily_returns_invested * total_invested_ratio) + (daily_risk_free_rate * (1 - total_invested_ratio))

    max_dd = calculate_max_drawdown(total_portfolio_daily_returns)

    # Beta calculation needs aligned returns
    beta = calculate_beta(total_portfolio_daily_returns, benchmark_returns_series)

    calmar = calculate_calmar_ratio(actual_portfolio_return, max_dd)
    sortino = calculate_sortino_ratio(total_portfolio_daily_returns, risk_free_rate_val)

    allocation_usd = weights * investing_amount_val
    return {
        "Retorno anual esperado": actual_portfolio_return, # Use the actual portfolio return
        "Volatilidad anual": actual_portfolio_volatility, # Use the actual portfolio volatility
        "Ratio de Sharpe": sharpe,
        "Máxima caída (Drawdown)": max_dd,
        "Beta": beta,
        "Ratio de Calmar": calmar,
        "Ratio de Sortino": sortino,
        "Monto a invertir ($)": investing_amount_val,
        **{f"Monto: {ticker} ($)": amt for ticker, amt in zip(tickers, allocation_usd)},
        "Total Asignado (%)": total_invested_ratio * 100 # New metric for sum of weights
    }

def display_portfolio(title, weights):
    st.subheader(title)
    
    # Pass all necessary variables to get_portfolio_metrics
    metrics = get_portfolio_metrics(weights, tickers, returns, benchmark_returns, risk_free_rate, investing_amount)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Retorno Anual Esperado", f"{metrics['Retorno anual esperado']:.2%}")
        st.metric("Volatilidad Anual", f"{metrics['Volatilidad anual']:.2%}")
        st.metric("Ratio de Sharpe", f"{metrics['Ratio de Sharpe']:.2f}")
        st.metric("Total Asignado", f"{metrics['Total Asignado (%)']:.2f}%") # Display this new metric
    with col2:
        st.metric("Máxima Caída (Drawdown)", f"{metrics['Máxima caída (Drawdown)']:.2%}")
        st.metric("Beta", f"{metrics['Beta']:.2f}")
        st.metric("Ratio de Calmar", f"{metrics['Ratio de Calmar']:.2f}")
        st.metric("Ratio de Sortino", f"{metrics['Ratio de Sortino']:.2f}")

    # Display weights and USD allocation
    weights_df = pd.DataFrame({
        "Activo": tickers,
        "Asignación (%)": [f"{w:.2%}" for w in weights],
        "Monto ($)": [f"${(w * investing_amount):,.2f}" for w in weights]
    })
    st.dataframe(weights_df, hide_index=True)

    # Plot weights bar chart
    plot_weights_bar(weights, tickers, f"Asignación de Activos para {title.split('Portafolio de ')[1]}")

    # Plot pie chart
    plot_pie_chart(weights, tickers, f"Distribución del Portafolio para {title.split('Portafolio de ')[1]}")

    # Plot cumulative returns
    # The portfolio_daily_returns for plotting should also reflect the total portfolio including cash
    total_invested_ratio = np.sum(weights)
    portfolio_daily_returns_invested = (returns * weights).sum(axis=1)
    daily_risk_free_rate = (1 + risk_free_rate)**(1/252) - 1
    total_portfolio_daily_returns_for_plot = (portfolio_daily_returns_invested * total_invested_ratio) + (daily_risk_free_rate * (1 - total_invested_ratio))

    plot_cumulative_returns(total_portfolio_daily_returns_for_plot, benchmark_returns)
    plot_drawdown(total_portfolio_daily_returns_for_plot)
    plot_rolling_volatility(total_portfolio_daily_returns_for_plot)


# === Aplicación Streamlit ===

st.title("Optimizador de Frontera Eficiente de Portafolios")

with st.sidebar:
    st.header("Configuración")
    ALL_TICKERS = load_tickers_from_file("tickers.txt")

    CRYPTO_TICKERS = [t for t in ALL_TICKERS if t.endswith('-USD')]

    st.markdown("#### Selecciona los símbolos para el portafolio")
    selected = []
    with st.container(height=300):
        for i, ticker in enumerate(ALL_TICKERS):
            label = ticker
            if ticker in CRYPTO_TICKERS:
                label += " (CRYPTO)"
            checked = st.checkbox(label, value=(ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "BTC-USD"]), key=ticker)
            if checked:
                selected.append(ticker)
    tickers = selected

    st.caption("Selecciona varios símbolos para diversificar tu portafolio. ¿No encuentras tu símbolo? Contacta a Pablo para que lo agregue.")

    crypto_assets_in_selection = [t for t in tickers if t in CRYPTO_TICKERS]
    
    # Determine if ONLY crypto assets are selected
    only_crypto_selected = (len(crypto_assets_in_selection) == len(tickers)) and len(tickers) > 0

    max_crypto_allocation = 1.0 # Default to no restriction
    if len(crypto_assets_in_selection) > 0:
        max_crypto_percent = st.slider(
            "Asignación máxima total a CRYPTO (%)", min_value=0, max_value=100, value=5, step=1,
            help="Limita el porcentaje máximo del portafolio que puede estar asignado a activos cripto."
        )
        max_crypto_allocation = max_crypto_percent / 100
        
        # If only crypto assets are selected AND the crypto allocation is limited,
        # then we allow partial investment (sum of weights < 1).
        # Otherwise, the sum of weights should be 1.
        allow_partial_investment_flag = only_crypto_selected and (max_crypto_allocation < 1.0)
    else:
        allow_partial_investment_flag = False


    start_date = st.date_input("Fecha de inicio", value=pd.to_datetime("2018-01-01"))
    # Use the current time to ensure 'today' is consistently defined
    current_time = pd.Timestamp('2025-06-10 16:00:40') # Hardcoding current_time for deterministic behavior in example
    today = current_time.normalize() # Date part only

    end_date = st.date_input("Fecha de fin", value=today)
    end_date_ts = pd.Timestamp(end_date)
    if end_date_ts > today:
        st.warning(f"La fecha de fin no puede ser en el futuro ({end_date_ts.date()}). Se usará la fecha de hoy ({today.date()}).")
        end_date_ts = today

    start_date_ts = pd.Timestamp(start_date)

    risk_free_rate = st.number_input("Tasa libre de riesgo (anual, decimal)", min_value=0.0, max_value=0.1, value=0.04, step=0.001, help="Por ejemplo, la tasa de bonos del tesoro a 10 años.") # Adjusted default to a more typical value
    num_portfolios = st.slider("Número de portafolios aleatorios a simular", min_value=1000, max_value=10000, value=3000, step=500)
    max_allocation_percent = st.slider("Asignación máxima por acción (%)", min_value=5, max_value=100, value=100, step=5)
    max_allocation = max_allocation_percent / 100

    investing_amount = st.number_input(
        "Monto a invertir ($ USD)", min_value=1.0, value=1000.0, step=100.0, format="%.2f"
    )

if len(tickers) < 2:
    st.error("Por favor, selecciona al menos dos símbolos.")
    st.stop()

if start_date_ts >= end_date_ts:
    st.error("La fecha de inicio debe ser anterior a la fecha de fin.")
    st.stop()

with st.spinner("Descargando datos de precios del portafolio..."):
    prices = fetch_data(tickers, start_date_ts.strftime('%Y-%m-%d'), end_date_ts.strftime('%Y-%m-%d'))
    # Ensure prices DataFrame has only the selected tickers and no extra columns
    prices = prices[tickers] # Filter to ensure column order matches tickers list

returns, mean_returns, cov_matrix = calc_returns_cov(prices)

with st.spinner("Descargando datos del índice de referencia (S&P 500)..."):
    benchmark_prices = fetch_data("^GSPC", start_date_ts.strftime('%Y-%m-%d'), end_date_ts.strftime('%Y-%m-%d'))
benchmark_returns = benchmark_prices.pct_change().dropna().iloc[:, 0]

crypto_indices = [i for i, t in enumerate(tickers) if t.endswith('-USD')]
if not crypto_indices or max_crypto_allocation >= 1.0:
    crypto_indices = []
    max_crypto_allocation = 1.0 # Ensure it's effectively 1.0 if no crypto or limit is 100%

with st.spinner(f"Simulando {num_portfolios} portafolios..."):
    results, weights_record = simulate_random_portfolios(
        mean_returns, cov_matrix, risk_free_rate, num_portfolios, max_allocation,
        crypto_indices=crypto_indices, max_crypto_allocation=max_crypto_allocation
    )

max_sharpe, min_vol = optimize_portfolios(
    mean_returns, cov_matrix, risk_free_rate, max_allocation,
    crypto_indices=crypto_indices, max_crypto_allocation=max_crypto_allocation,
    allow_partial_investment=allow_partial_investment_flag # Pass the flag
)

# Pass the allow_partial_investment_flag to plot_efficient_frontier as well
fig = plot_efficient_frontier(results, max_sharpe, min_vol, mean_returns, cov_matrix, tickers, risk_free_rate, max_allocation=max_allocation, crypto_indices=crypto_indices, max_crypto_allocation=max_crypto_allocation, allow_partial_investment=allow_partial_investment_flag)
st.pyplot(fig)


display_portfolio("Portafolio de Máximo Ratio de Sharpe", max_sharpe.x)
display_portfolio("Portafolio de Mínima Volatilidad", min_vol.x)


# --- Descargar CSV solo con los mejores portafolios (portafolios estrella) al final ---

best_portfolios = {
    "Portafolio de Máximo Ratio de Sharpe": max_sharpe.x,
    "Portafolio de Mínima Volatilidad": min_vol.x
}

rows = []
for name, weights in best_portfolios.items():
    # Call get_portfolio_metrics with all required arguments
    metrics = get_portfolio_metrics(weights, tickers, returns, benchmark_returns, risk_free_rate, investing_amount)
    row = {"Nombre": name}
    row.update({f"Peso: {ticker}": f"{w:.4f}" for ticker, w in zip(tickers, weights)}) # Format weights for CSV
    # Ensure numerical metrics are formatted without % for CSV
    row["Retorno anual esperado"] = metrics["Retorno anual esperado"]
    row["Volatilidad anual"] = metrics["Volatilidad anual"]
    row["Ratio de Sharpe"] = metrics["Ratio de Sharpe"]
    row["Máxima caída (Drawdown)"] = metrics["Máxima caída (Drawdown)"]
    row["Beta"] = metrics["Beta"]
    row["Ratio de Calmar"] = metrics["Ratio de Calmar"]
    row["Ratio de Sortino"] = metrics["Ratio de Sortino"]
    row["Monto a invertir ($)"] = metrics["Monto a invertir ($)"]
    # Assuming 'allocation_usd' was passed as part of the original dictionary,
    # or you reconstruct it from the 'weights' and 'investing_amount'.
    # The get_portfolio_metrics already returns it as a dictionary.
    # We need to filter for the 'Monto: ' keys and ensure we're getting the numeric values.

    # First, extract the allocation_usd dictionary correctly from metrics:
    allocation_usd_from_metrics = {k: v for k, v in metrics.items() if "Monto: " in k}

    # Now, add them to the row directly without trying to call .replace()
    row.update(allocation_usd_from_metrics)    
    row["Total Asignado (%)"] = metrics["Total Asignado (%)"] # Store as number for CSV
    rows.append(row)

best_weights_df = pd.DataFrame(rows)

st.markdown("### Descargar pesos y métricas de los mejores portafolios (con estrella)")
st.download_button(
    label="Descargar CSV de mejores portafolios",
    data=best_weights_df.to_csv(index=False).encode('utf-8'),
    file_name="mejores_portafolios.csv",
    mime="text/csv",
    help="Descarga un archivo CSV con los pesos y métricas de los portafolios óptimos (máximo Sharpe y mínima volatilidad)."
)

st.markdown("### Matriz de correlación de los activos seleccionados")
plot_correlation_heatmap(prices)

st.markdown(
    """
    <hr style="margin-top:2em;margin-bottom:1em;">
    <div style="text-align:center; color:gray;">
        Creado por Pablo &middot; 
        <a href="https://github.com/pcruiher08" target="_blank" style="color:inherit;text-decoration:underline;">GitHub: @pcruiher08</a>
    </div>
    """,
    unsafe_allow_html=True
)