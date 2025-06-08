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
            raise ValueError("Se esperaban columnas de múltiples índices para varios símbolos.")
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
            # Solo establecer columnas si el número coincide
            if prices.shape[1] == len(tickers):
                prices.columns = tickers

    return prices.dropna()


def plot_efficient_frontier(results, max_sharpe, min_vol, mean_returns, cov_matrix, tickers, risk_free_rate, max_allocation=1.0):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 7))

    # Gráfico de dispersión de portafolios aleatorios
    sc = ax.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', marker='o', s=10, alpha=0.3)
    plt.colorbar(sc, label='Ratio de Sharpe', ax=ax)

    # Resaltar portafolios de máximo Sharpe y mínima volatilidad
    max_sharpe_ret, max_sharpe_vol, _ = portfolio_performance(max_sharpe.x, mean_returns, cov_matrix, risk_free_rate)
    min_vol_ret, min_vol_vol, _ = portfolio_performance(min_vol.x, mean_returns, cov_matrix, risk_free_rate)

    ax.scatter(max_sharpe_vol, max_sharpe_ret, marker='*', color='r', s=300, label='Máximo Ratio de Sharpe')
    ax.scatter(min_vol_vol, min_vol_ret, marker='*', color='b', s=300, label='Mínima Volatilidad')

    # Línea de la frontera eficiente
    target_returns = np.linspace(min_vol_ret, max(results[0, :]), 100)
    efficient_vols = []

    num_assets = len(mean_returns)
    bounds = tuple((0, max_allocation) for _ in range(num_assets))
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    ]

    for target in target_returns:
        constraints_target = constraints + [{'type': 'eq', 'fun': lambda x, target=target: np.dot(x, mean_returns) * 252 - target}]
        result = minimize(portfolio_volatility,
                          num_assets * [1. / num_assets],
                          args=(mean_returns, cov_matrix),
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints_target)
        if result.success:
            efficient_vols.append(result.fun)
        else:
            efficient_vols.append(np.nan)

    ax.plot(efficient_vols, target_returns, 'r--', linewidth=2, label='Frontera Eficiente')

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

def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    ret = np.sum(mean_returns * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = (ret - risk_free_rate) / vol
    return ret, vol, sharpe

def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix, 0)[1]

def optimize_portfolios(mean_returns, cov_matrix, risk_free_rate, max_allocation=1.0):
    num_assets = len(mean_returns)
    bounds = tuple((0, max_allocation) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    initial_guess = num_assets * [1. / num_assets]

    max_sharpe = minimize(neg_sharpe, initial_guess,
                          args=(mean_returns, cov_matrix, risk_free_rate),
                          method='SLSQP', bounds=bounds, constraints=constraints)

    min_vol = minimize(portfolio_volatility, initial_guess,
                       args=(mean_returns, cov_matrix),
                       method='SLSQP', bounds=bounds, constraints=constraints)

    return max_sharpe, min_vol

def generate_valid_weights(num_assets, max_allocation, max_attempts=1000):
    for _ in range(max_attempts):
        weights = np.random.uniform(0, max_allocation, num_assets)
        weights /= np.sum(weights)
        if np.all(weights <= max_allocation):
            return weights
    # en caso de no encontrar, usar pesos iguales
    equal_weight = min(1/num_assets, max_allocation)
    weights = np.array([equal_weight]*num_assets)
    weights /= np.sum(weights)
    return weights

def simulate_random_portfolios(mean_returns, cov_matrix, risk_free_rate, num_portfolios=3000, max_allocation=1.0):
    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = generate_valid_weights(num_assets, max_allocation)
        weights_record.append(weights)

        ret, vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
        results[0, i] = ret
        results[1, i] = vol
        results[2, i] = sharpe

    return results, weights_record


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
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(weights, labels=tickers, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    ax.set_title(title)
    st.pyplot(fig)
    st.caption("**Gráfico de pay**: Muestra la proporción de cada activo en el portafolio.")

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

def load_tickers_from_file(filepath="tickers.txt"):
    try:
        with open(filepath, "r") as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        if not tickers:
            st.error(f"No se encontraron símbolos en {filepath}. Por favor, revisa el archivo.")
            st.stop()
        return tickers
    except FileNotFoundError:
        st.error(f"Archivo de símbolos {filepath} no encontrado.")
        st.stop()


# Funciones adicionales para nuevas métricas

def calculate_max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown

def calculate_beta(portfolio_returns, benchmark_returns):
    covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    beta = covariance / benchmark_variance
    return beta

def calculate_calmar_ratio(annual_return, max_drawdown):
    if max_drawdown == 0:
        return np.nan
    return annual_return / abs(max_drawdown)

def calculate_sortino_ratio(returns, risk_free_rate=0.09):
    downside_returns = returns[returns < 0]
    expected_return = returns.mean() * 252
    downside_std = downside_returns.std() * np.sqrt(252)
    if downside_std == 0:
        return np.nan
    return (expected_return - risk_free_rate) / downside_std


# === Aplicación Streamlit ===

st.title("Optimizador de Frontera Eficiente de Portafolios")

with st.sidebar:
    st.header("Configuración")
    ALL_TICKERS = load_tickers_from_file("tickers.txt")

    # Mejor UX de selección de símbolos: muestra todos los símbolos como checkboxes en un contenedor desplazable, sin necesidad de escribir
    st.markdown("#### Selecciona los símbolos para el portafolio")
    max_show = 30  # Número de símbolos a mostrar antes de permitir desplazamiento
    selected = []
    with st.container():
        for i, ticker in enumerate(ALL_TICKERS):
            if i == max_show:
                st.markdown("---")
                st.markdown("Desplázate para ver más símbolos...")
            checked = st.checkbox(ticker, value=(ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "BTC-USD"]), key=ticker)
            if checked:
                selected.append(ticker)
    tickers = selected

    st.caption("Selecciona varios símbolos para diversificar tu portafolio. ¿No encuentras tu símbolo? Contacta a Pablo para que lo agregue.")

    start_date = st.date_input("Fecha de inicio", value=pd.to_datetime("2018-01-01"))
    today = pd.to_datetime("today").normalize()
    end_date = st.date_input("Fecha de fin", value=today)
    end_date_ts = pd.Timestamp(end_date)
    if end_date_ts > today:
        st.warning("La fecha de fin no puede ser en el futuro. Se usará la fecha de hoy.")
        end_date_ts = today

    # Convertir start_date a pd.Timestamp para comparación y uso
    start_date_ts = pd.Timestamp(start_date)

    risk_free_rate = st.number_input("Tasa libre de riesgo (anual, decimal)", min_value=0.0, max_value=0.1, value=0.09, step=0.001)
    num_portfolios = st.slider("Número de portafolios aleatorios a simular", min_value=1000, max_value=10000, value=3000, step=500)
    max_allocation_percent = st.slider("Asignación máxima por acción (%)", min_value=5, max_value=100, value=100, step=5)
    max_allocation = max_allocation_percent / 100

if len(tickers) < 2:
    st.error("Por favor, selecciona al menos dos símbolos.")
    st.stop()

if start_date_ts >= end_date_ts:
    st.error("La fecha de inicio debe ser anterior a la fecha de fin.")
    st.stop()

if max_allocation * len(tickers) < 1.0:
    st.error("La asignación máxima por acción es muy baja para los símbolos seleccionados. Aumenta la asignación máxima o selecciona más símbolos.")
    st.stop()

# Cargar datos de precios del portafolio
with st.spinner("Descargando datos de precios del portafolio..."):
    prices = fetch_data(tickers, start_date_ts.strftime('%Y-%m-%d'), end_date_ts.strftime('%Y-%m-%d'))

returns, mean_returns, cov_matrix = calc_returns_cov(prices)

# Cargar datos del índice de referencia (S&P 500)
with st.spinner("Descargando datos del índice de referencia (S&P 500)..."):
    benchmark_prices = fetch_data("^GSPC", start_date_ts.strftime('%Y-%m-%d'), end_date_ts.strftime('%Y-%m-%d'))
benchmark_returns = benchmark_prices.pct_change().dropna().iloc[:, 0]

# Ejecutar simulaciones
with st.spinner(f"Simulando {num_portfolios} portafolios..."):
    results, weights_record = simulate_random_portfolios(mean_returns, cov_matrix, risk_free_rate, num_portfolios, max_allocation)

# Optimizar portafolios
max_sharpe, min_vol = optimize_portfolios(mean_returns, cov_matrix, risk_free_rate, max_allocation)

def display_portfolio(title, weights):
    st.subheader(title)
    aligned_tickers = prices.columns.tolist()
    if len(weights) != len(aligned_tickers):
        st.error(f"Desajuste de longitud: pesos ({len(weights)}) vs símbolos ({len(aligned_tickers)})")
        return

    df = pd.DataFrame({
        'Símbolo': aligned_tickers,
        'Asignación (%)': weights * 100
    })
    st.table(df.style.format({"Asignación (%)": "{:.2f}%"}))
    st.caption("**Asignación (%)**: Porcentaje del portafolio invertido en cada activo.")

    ret, vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
    st.write(f"Retorno anual esperado: **{ret:.2%}**")
    st.caption("**Retorno anual esperado**: Suma ponderada de los retornos medios diarios anualizados de los activos del portafolio.")

    st.write(f"Volatilidad anual: **{vol:.2%}**")
    st.caption("**Volatilidad anual**: Desviación estándar anualizada de los retornos del portafolio. Mide el riesgo total.")

    st.write(f"Ratio de Sharpe: **{sharpe:.4f}**")
    st.caption("**Ratio de Sharpe**: (Retorno del portafolio - tasa libre de riesgo) / volatilidad. Mide el retorno ajustado al riesgo.")

    portfolio_returns = (returns * weights).sum(axis=1)

    max_dd = calculate_max_drawdown(portfolio_returns)
    st.write(f"Máxima caída (Drawdown): **{max_dd:.2%}**")
    st.caption("**Máxima caída (Drawdown)**: Mayor pérdida porcentual desde un máximo histórico hasta un mínimo posterior.")

    beta = calculate_beta(portfolio_returns.loc[benchmark_returns.index], benchmark_returns)
    st.write(f"Beta del portafolio vs. S&P 500: **{beta:.3f}**")
    st.caption("**Beta**: Covarianza entre los retornos del portafolio y el índice de referencia dividida por la varianza del índice. Mide la sensibilidad del portafolio al mercado.")

    calmar = calculate_calmar_ratio(ret, max_dd)
    st.write(f"Ratio de Calmar: **{calmar:.3f}**")
    st.caption("**Ratio de Calmar**: Retorno anual esperado dividido por la máxima caída (drawdown). Mide el retorno ajustado a la peor pérdida.")

    sortino = calculate_sortino_ratio(portfolio_returns, risk_free_rate)
    st.write(f"Ratio de Sortino: **{sortino:.3f}**")
    st.caption("**Ratio de Sortino**: (Retorno esperado - tasa libre de riesgo) dividido por la desviación estándar de los retornos negativos. Mide el retorno ajustado al riesgo bajista.")

    # Organizar gráficos en columnas para mejor legibilidad
    col1, col2 = st.columns(2)
    with col1:
        plot_weights_bar(weights, prices.columns, f"{title} - Pesos de los activos")
    with col2:
        plot_pie_chart(weights, prices.columns, f"{title} - Distribución del Portafolio")

    st.markdown("#### Gráficos de desempeño del portafolio")
    plot_cumulative_returns(portfolio_returns.loc[benchmark_returns.index], benchmark_returns)
    plot_drawdown(portfolio_returns)
    plot_rolling_volatility(portfolio_returns)

display_portfolio("Portafolio de Máximo Ratio de Sharpe", max_sharpe.x)
display_portfolio("Portafolio de Mínima Volatilidad", min_vol.x)

fig = plot_efficient_frontier(results, max_sharpe, min_vol, mean_returns, cov_matrix, tickers, risk_free_rate, max_allocation=max_allocation)
st.pyplot(fig)

# --- Descargar CSV solo con los mejores portafolios (portafolios estrella) al final ---

# Preparar métricas para cada mejor portafolio
def get_portfolio_metrics(weights):
    ret, vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
    portfolio_returns = (returns * weights).sum(axis=1)
    max_dd = calculate_max_drawdown(portfolio_returns)
    beta = calculate_beta(portfolio_returns.loc[benchmark_returns.index], benchmark_returns)
    calmar = calculate_calmar_ratio(ret, max_dd)
    sortino = calculate_sortino_ratio(portfolio_returns, risk_free_rate)
    return {
        "Retorno anual esperado": ret,
        "Volatilidad anual": vol,
        "Ratio de Sharpe": sharpe,
        "Máxima caída (Drawdown)": max_dd,
        "Beta": beta,
        "Ratio de Calmar": calmar,
        "Ratio de Sortino": sortino
    }

best_portfolios = {
    "Portafolio de Máximo Ratio de Sharpe": max_sharpe.x,
    "Portafolio de Mínima Volatilidad": min_vol.x
}

# Construir DataFrame con pesos y métricas
rows = []
for name, weights in best_portfolios.items():
    metrics = get_portfolio_metrics(weights)
    row = {"Nombre": name}
    # Agregar pesos con nombres de columna como "Peso: TICKER"
    row.update({f"Peso: {ticker}": w for ticker, w in zip(prices.columns, weights)})
    # Agregar métricas
    row.update(metrics)
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

# Después de mostrar todos los portafolios/métricas, mostrar el heatmap de correlación para los símbolos seleccionados
#st.markdown("### Matriz de correlación de los activos seleccionados")
#plot_correlation_heatmap(prices)
