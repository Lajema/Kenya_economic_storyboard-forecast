import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression

# 1. Page Configuration
st.set_page_config(
    page_title="Kenya Economic Storyboard & Forecast",
    page_icon="🇰🇪",
    layout="wide"
)

# 2. Inject Custom Fonts (Lato) and Global Styles
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Lato:ital,wght@0,300;0,400;0,700;1,400&display=swap');
        
        html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4, h5, h6, label {
            font-family: 'Lato', sans-serif !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 2.2rem !important;
            font-weight: 700 !important;
        }
        [data-testid="stMetricLabel"] {
            font-weight: 400 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.85rem !important;
        }
        .translator-note {
            background-color: #f0f4f8;
            border-left: 4px solid #3b82f6;
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 15px;
        }
        .forecast-note {
            background-color: #fef3c7;
            border-left: 4px solid #d97706;
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 15px;
            color: #78350f;
        }
    </style>
""", unsafe_allow_html=True)

# 3. Data Loading Engine
@st.cache_data
def load_data():
    df = pd.read_csv("API_KEN_DS2_en_csv_v2_5938.csv", skiprows=4)
    df.columns = df.columns.str.strip()
    df = df.dropna(how='all', axis=1)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load the CSV file. Verify 'API_KEN_DS2_en_csv_v2_5938.csv' is in this directory. Error: {e}")
    st.stop()

# Data Extractor Helper
def get_indicator_timeline(indicator_name):
    row = df[df['Indicator Name'] == indicator_name]
    if row.empty:
        return pd.Series(dtype=float)
    year_cols = [col for col in df.columns if col.isdigit()]
    timeline = row[year_cols].T
    timeline.columns = ['Value']
    timeline.index = timeline.index.astype(int)
    timeline['Value'] = pd.to_numeric(timeline['Value'], errors='coerce')
    return timeline.dropna()

def get_latest_val_and_year(timeline_df, start, end):
    filtered = timeline_df[(timeline_df.index >= start) & (timeline_df.index <= end)]
    if not filtered.empty:
        idx = filtered.index[-1]
        return filtered.loc[idx, 'Value'], idx
    return None, None

# Forecasting Engine using Sklearn Linear Regression
def forecast_next_5_years(timeline_df):
    if timeline_df.empty or len(timeline_df) < 5:
        return pd.DataFrame()
    
    # Train on the last 15 years to catch modern structural trends rather than 1960s data
    recent_data = timeline_df.tail(15)
    X = recent_data.index.values.reshape(-1, 1)
    y = recent_data['Value'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    last_actual_year = timeline_df.index[-1]
    future_years = np.array(range(last_actual_year + 1, last_actual_year + 6)).reshape(-1, 1)
    predictions = model.predict(future_years)
    
    # Return formatted dataframe
    forecast_df = pd.DataFrame({
        'Year': future_years.flatten(),
        'Projected Value': predictions
    }).set_index('Year')
    return forecast_df

# 4. Global Timeline Navigation
year_columns = [int(col) for col in df.columns if col.isdigit()]
min_year, max_year = min(year_columns), max(year_columns)

st.sidebar.header("🗺️ Dashboard Navigation")
page = st.sidebar.radio(
    "Go To:", 
    [
        "1. Executive Summary", 
        "2. Growth & Stability Deep Dive", 
        "3. External Trade & Remittances",
        "4. Future Horizons (5-Year Forecasts)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Global Timeline")
selected_years = st.sidebar.slider(
    "Select Target Window",
    min_value=min_year,
    max_value=max_year,
    value=(2000, max_year)
)
start_yr, end_yr = selected_years

# Extract Universal Datasets
gdp_df = get_indicator_timeline("GDP growth (annual %)")
inflation_df = get_indicator_timeline("Inflation, consumer prices (annual %)")
exports_df = get_indicator_timeline("Merchandise exports (current US$)")
imports_df = get_indicator_timeline("Merchandise imports (current US$)")
remit_df = get_indicator_timeline("Personal remittances, received (current US$)")
reserves_df = get_indicator_timeline("Total reserves (includes gold, current US$)")
debt_service_df = get_indicator_timeline("Public and publicly guaranteed debt service (% of exports of goods, services and primary income)")
interest_rev_df = get_indicator_timeline("Interest payments (% of revenue)")
savings_gdp_df = get_indicator_timeline("Gross savings (% of GDP)")
capital_growth_df = get_indicator_timeline("Gross capital formation (annual % growth)")
food_df = get_indicator_timeline("Food exports (% of merchandise exports)")
manuf_df = get_indicator_timeline("Manufactures exports (% of merchandise exports)")

# ==========================================
# PAGE 1: EXECUTIVE SUMMARY
# ==========================================
if page == "1. Executive Summary":
    st.title("🏛️ Executive Summary: The Economic Pulse")
    st.markdown("""
    This dashboard translates decades of complex numbers into a simple story about Kenya’s economic journey. 
    Think of this page as a quick medical check-up of the country's financial vitals.
    """)
    st.markdown("---")
    
    # Snapshot Metrics Block
    gdp_val, gdp_yr = get_latest_val_and_year(gdp_df, start_yr, end_yr)
    inf_val, inf_yr = get_latest_val_and_year(inflation_df, start_yr, end_yr)
    rem_val, rem_yr = get_latest_val_and_year(remit_df, start_yr, end_yr)
    res_val, res_yr = get_latest_val_and_year(reserves_df, start_yr, end_yr)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label=f"GDP Growth ({gdp_yr if gdp_yr else 'N/A'})", value=f"{gdp_val:.2f}%" if gdp_val else "No Data")
        st.caption("How fast the economic pie is growing compared to last year.")
    with col2:
        st.metric(label=f"Inflation Rate ({inf_yr if inf_yr else 'N/A'})", value=f"{inf_val:.2f}%" if inf_val else "No Data")
        st.caption("How much more expensive everyday goods became this year.")
    with col3:
        st.metric(label=f"Diaspora Inflows ({rem_yr if rem_yr else 'N/A'})", value=f"${rem_val/1e9:.2f}B" if rem_val else "No Data")
        st.caption("Total money sent home by Kenyans living and working abroad.")
    with col4:
        st.metric(label=f"Forex Buffer ({res_yr if res_yr else 'N/A'})", value=f"${res_val/1e9:.2f}B" if res_val else "No Data")
        st.caption("The central bank's emergency rainy-day savings tank.")
        
    st.markdown("---")
    
    # Summary Visualization Combo
    st.subheader("The Tug-of-War: Economic Growth vs. Living Costs")
    col_summary_l, col_summary_r = st.columns([3, 2])
    
    with col_summary_l:
        combined_all = pd.DataFrame(index=range(start_yr, end_yr + 1)).join(gdp_df).rename(columns={'Value': 'GDP'}).join(inflation_df).rename(columns={'Value': 'Inflation'}).dropna(how='all')
        fig_summary = go.Figure()
        fig_summary.add_trace(go.Scatter(x=combined_all.index, y=combined_all['GDP'], name="Economic Growth Speed", line=dict(color='#10b981', width=3)))
        fig_summary.add_trace(go.Scatter(x=combined_all.index, y=combined_all['Inflation'], name="Price Increase Speed (Inflation)", line=dict(color='#ef4444', width=2, dash='dot')))
        fig_summary.update_layout(font_family="Lato", margin=dict(t=20, b=20, l=20, r=20), hovermode="x unified")
        st.plotly_chart(fig_summary, use_container_width=True)
        
    with col_summary_r:
        st.markdown("""
        <div class="translator-note">
            <strong>💡 Plain-English Translator Note:</strong><br>
            Imagine economic growth is your car's speed, and inflation is the wind resistance hitting your windshield. 
            <ul>
                <li>When the <strong>green line (growth)</strong> is high, businesses are expanding and jobs are being created.</li>
                <li>When the <strong>red dashed line (inflation)</strong> spikes, it means your money loses purchasing power—food, rent, and fuel are getting expensive too quickly.</li>
                <li><strong>The Goal:</strong> We want a high green line and a low red line. When they cross or get too close, citizens feel squeezed because prices are rising faster than the economy is expanding.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# PAGE 2: GROWTH & STABILITY DEEP DIVE
# ==========================================
elif page == "2. Growth & Stability Deep Dive":
    st.title("🔄 The Engine: Growth, Investment & Domestic Cushioning")
    st.markdown("Let's look under the hood to see what drives Kenya's economic stability: how national debt pressures eat into government budgets, and whether the country is saving enough to fund its own future.")
    st.markdown("---")
    
    col_dive1, col_dive2 = st.columns(2)
    
    with col_dive1:
        st.subheader("1. The Budget Tightrope: Debt Payments vs. Revenue")
        
        combined_fiscal = pd.DataFrame(index=range(start_yr, end_yr + 1)).join(debt_service_df).rename(columns={'Value': 'Debt Service'}).join(interest_rev_df).rename(columns={'Value': 'Interest Cost'}).dropna(how='all')
        fig_fiscal = go.Figure()
        fig_fiscal.add_trace(go.Scatter(x=combined_fiscal.index, y=combined_fiscal['Debt Service'], name="Debt Payments (% of Exports)", line=dict(color='#f59e0b', width=2.5)))
        fig_fiscal.add_trace(go.Scatter(x=combined_fiscal.index, y=combined_fiscal['Interest Cost'], name="Interest Cut from Revenues (%)", line=dict(color='#6b7280', width=2.5, dash='dash')))
        fig_fiscal.update_layout(font_family="Lato", margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_fiscal, use_container_width=True)
        
        st.markdown("""
        <div class="translator-note">
            <strong>💡 Plain-English Translator Note:</strong><br>
            Just like a household with too many credit cards, a government has to split its income between paying off loans and buying essentials.<br><br>
            The <strong>grey dashed line</strong> shows what percentage of all tax revenues collected goes <em>just</em> toward paying interest on loans. The higher this line goes, the less money the government has left to spend on public healthcare, public schools, and repairing roads.
        </div>
        """, unsafe_allow_html=True)
        
    with col_dive2:
        st.subheader("2. Building the Future: National Savings vs. Big Projects")
        
        combined_invest = pd.DataFrame(index=range(start_yr, end_yr + 1)).join(savings_gdp_df).rename(columns={'Value': 'Savings'}).join(capital_growth_df).rename(columns={'Value': 'Capital Growth'}).dropna(how='all')
        fig_invest = go.Figure()
        fig_invest.add_trace(go.Scatter(x=combined_invest.index, y=combined_invest['Savings'], name="National Piggy Bank Size (% of GDP)", fill='tozeroy', line_color='rgba(59, 130, 246, 0.3)'))
        fig_invest.add_trace(go.Scatter(x=combined_invest.index, y=combined_invest['Capital Growth'], name="Investment in Mega Projects (YoY % Growth)", mode='lines+markers', line=dict(color='#ec4899', width=2)))
        fig_invest.update_layout(font_family="Lato", margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_invest, use_container_width=True)
        
        st.markdown("""
        <div class="translator-note">
            <strong>💡 Plain-English Translator Note:</strong><br>
            To build railroads, power plants, and technology hubs, a country needs money. 
            <ul>
                <li>The <strong>blue shaded area</strong> is Kenya's "National Piggy Bank" (money saved domestically by citizens and corporations).</li>
                <li>The <strong>pink line</strong> shows the year-on-year growth speed of building these big physical assets.</li>
                <li><strong>The Story:</strong> When the pink line shoots way higher than our blue piggy bank can afford, the country has to borrow money from foreign lenders to fund those projects—which triggers the debt spikes seen in the left chart!</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# PAGE 3: EXTERNAL TRADE & REMITTANCES
# ==========================================
elif page == "3. External Trade & Remittances":
    st.title("🌐 The Bridges: Global Trade, Export Composition & Inflows")
    st.markdown("Kenya doesn't operate in a vacuum. This section maps out how much we buy from other nations, what items we sell to them, and the crucial lifelines that protect our local currency.")
    st.markdown("---")
    
    # Row 1: The Core Balance of Trade
    st.subheader("The Shopping Bill: Imports vs. Exports")
    
    combined_trade = pd.DataFrame(index=range(start_yr, end_yr + 1)).join(exports_df).rename(columns={'Value': 'Exports'}).join(imports_df).rename(columns={'Value': 'Imports'}).dropna(how='all')
    fig_trade = go.Figure()
    fig_trade.add_trace(go.Bar(x=combined_trade.index, y=combined_trade['Exports']/1e9, name="What We Sold Overseas (Exports)", marker_color='#3b82f6'))
    fig_trade.add_trace(go.Bar(x=combined_trade.index, y=combined_trade['Imports']/1e9, name="What We Bought Globally (Imports)", marker_color='#f97316'))
    fig_trade.update_layout(font_family="Lato", barmode='group', xaxis_title="Year", yaxis_title="Billions of USD", margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_trade, use_container_width=True)
    
    st.markdown("""
    <div class="translator-note" style="margin-top:-10px;">
        <strong>💡 Plain-English Translator Note:</strong><br>
        Think of this chart as Kenya's international store tab. The <strong>blue bars</strong> show the value of things Kenya sells overseas (like tea, flowers, and coffee). The <strong>orange bars</strong> show what we buy from abroad (like machinery, vehicles, and oil). <br>
        Because the orange bars are consistently much taller than the blue bars, Kenya runs a <strong>Trade Deficit</strong>. This means more money is flowing out of the country to pay for foreign goods than is coming in.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Row 2: Composition Shifts vs. Stabilization Reservoirs
    col_ext_l, col_ext_r = st.columns(2)
    
    with col_ext_l:
        st.subheader("What We Ship Out: Farm Produce vs. Factories")
        
        comp_df = pd.DataFrame(index=range(start_yr, end_yr + 1)).join(food_df).rename(columns={'Value': 'Food'}).join(manuf_df).rename(columns={'Value': 'Manufactures'}).dropna(how='all')
        fig_comp = px.line(comp_df, x=comp_df.index, y=["Food", "Manufactures"], labels={"value": "% Share of All Exports", "index": "Year", "variable": "Export Sector"}, color_discrete_sequence=["#10b981", "#8b5cf6"])
        fig_comp.update_layout(font_family="Lato", margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_comp, use_container_width=True)
        
        st.markdown("""
        <div class="translator-note">
            <strong>💡 Plain-English Translator Note:</strong><br>
            Developing economies want to transform from selling raw materials to selling high-value manufactured items. <br>
            This timeline tracks the percentage split of our physical exports. If the <strong>green line (food/agriculture)</strong> stays dominant while the <strong>purple line (manufactured goods)</strong> stays flat, it tells us the economy is still heavily dependent on farm output and vulnerable to weather/climate shocks.
        </div>
        """, unsafe_allow_html=True)
        
    with col_ext_r:
        st.subheader("The Invisible Heroes: Diaspora Money & Central Forex Cushions")
        
        combined_cushion = pd.DataFrame(index=range(start_yr, end_yr + 1)).join(remit_df).rename(columns={'Value': 'Remittances'}).join(reserves_df).rename(columns={'Value': 'Reserves'}).dropna(how='all')
        fig_cushion = go.Figure()
        fig_cushion.add_trace(go.Scatter(x=combined_cushion.index, y=combined_cushion['Remittances']/1e9, name="Money Sent by Kenyans Abroad", line=dict(color='#6366f1', width=3)))
        fig_cushion.add_trace(go.Scatter(x=combined_cushion.index, y=combined_cushion['Reserves']/1e9, name="Central Bank Backup Stash", fill='tonexty', line_color='rgba(14, 165, 233, 0.2)'))
        fig_cushion.update_layout(font_family="Lato", margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_cushion, use_container_width=True)
        
        st.markdown("""
        <div class="translator-note">
            <strong>💡 Plain-English Translator Note:</strong><br>
            Remember how our trade chart above showed that Kenya spends more foreign cash than it earns? How do we balance the books? Enter the diaspora lifeline! <br>
            The <strong>purple line</strong> tracks money wired home by Kenyans abroad. This hard currency directly refills the Central Bank's <strong>blue backup stash (Forex Reserves)</strong>. This stash is crucial because it ensures the country always has enough foreign money to buy critical imports like medicine and fuel without causing the Shilling to crash.
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# PAGE 4: FUTURE HORIZONS (5-YEAR FORECAST)
# ==========================================
elif page == "4. Future Horizons (5-Year Forecasts)":
    st.title("🔮 Future Horizons: Machine Learning 5-Year Trend Estimator")
    st.markdown("""
    This section uses a statistical forecasting model to predict where Kenya's core economic baselines are heading 
    over the next 5 years based on modern rolling data trends.
    """)
    st.markdown("---")
    
    # Forecast selections selector
    st.subheader("Select an Indicator to Peek into the Future")
    forecast_choice = st.selectbox(
        "Choose Target Variable:",
        ["Economic Growth Speed (GDP %)", "Money Sent by Kenyans Abroad (Diaspora Remittances)", "Central Bank Backup Stash (Forex Reserves)"]
    )
    
    # Map selection to actual dataframes
    if forecast_choice == "Economic Growth Speed (GDP %)":
        target_df = gdp_df
        title_text = "GDP Growth Projections (Annual %)"
        y_label = "Growth Percentage (%)"
        is_currency = False
    elif forecast_choice == "Money Sent by Kenyans Abroad (Diaspora Remittances)":
        target_df = remit_df
        title_text = "Diaspora Remittance Inflow Projections"
        y_label = "Value in Billions (USD)"
        is_currency = True
    else:
        target_df = reserves_df
        title_text = "Central Bank Foreign Exchange Reserves Cushion Projections"
        y_label = "Value in Billions (USD)"
        is_currency = True

    # Compute prediction
    f_df = forecast_next_5_years(target_df)
    
    if not f_df.empty:
        col_graph, col_expl = st.columns([3, 2])
        
        with col_graph:
            # Prepare plotting arrays
            hist_years = target_df.index
            hist_vals = target_df['Value'].values / 1e9 if is_currency else target_df['Value'].values
            
            pred_years = f_df.index
            pred_vals = f_df['Projected Value'].values / 1e9 if is_currency else f_df['Projected Value'].values
            
            fig_fc = go.Figure()
            # Historical line
            fig_fc.add_trace(go.Scatter(x=hist_years, y=hist_vals, name="Actual History", line=dict(color='#1e293b', width=2.5)))
            # Forecast dotted line
            fig_fc.add_trace(go.Scatter(x=pred_years, y=pred_vals, name="5-Year Prediction Model", line=dict(color='#d97706', width=3, dash='dot')))
            
            # Connect the gap between history and prediction line smoothly
            fig_fc.add_trace(go.Scatter(x=[hist_years[-1], pred_years[0]], y=[hist_vals[-1], pred_vals[0]], showlegend=False, line=dict(color='#d97706', width=2, dash='dot')))
            
            fig_fc.update_layout(font_family="Lato", title=title_text, xaxis_title="Year", yaxis_title=y_label, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_fc, use_container_width=True)
            
        with col_expl:
            st.markdown("""
            <div class="forecast-note">
                <strong>🧠 How does this prediction work?</strong><br>
                The dashboard isolates the last 15 years of actual data points and draws a mathematical trajectory vector forward. 
                It effectively isolates modern structural momentum to see what your next 5 years look like if structural habits remain stable.
            </div>
            """, unsafe_allow_html=True)
            
            # Print explicit numbers
            st.markdown("### 📊 Projected Figures Table")
            display_f = f_df.copy()
            if is_currency:
                display_f['Projected Value'] = display_f['Projected Value'].apply(lambda x: f"${x/1e9:.2f} Billion USD")
                st.table(display_f.reset_index())
            else:
                display_f['Projected Value'] = display_f['Projected Value'].apply(lambda x: f"{x:.2f}%")
                st.table(display_f.reset_index())
                
            st.markdown("""
            ⚠️ *Disclaimer: Predictive data models assume relative continuity. Major real-world shocks like weather anomalies, abrupt regulatory changes, or global supply market crashes cannot be anticipated by historical trends.*
            """)
    else:
        st.warning("Insufficient sequential milestones to compute a valid predictive output baseline.")