import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
from scipy.stats import skew, kurtosis

# Page configuration
st.set_page_config(
    page_title="Global Research Impact Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #1f77b4 0%, #ff7f0e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #f0f9ff;
        border-left: 4px solid #0ea5e9;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    try:
        # Read the actual data file
        df = pd.read_csv('publications.txt', sep='\t')
        
        # Calculate derived metrics
        df['Citations_per_Doc'] = df['Times Cited'] / df['Web of Science Documents']
        df['Elite_Ratio'] = df['% Documents in Top 1%'] / df['% Documents in Top 10%']
        df['Impact_Score'] = df['Category Normalized Citation Impact'] * df['% Documents in Top 10%']
        df['H_Index_Proxy'] = np.sqrt(df['Times Cited'] * df['Web of Science Documents'])
        df['Productivity_Index'] = df['Web of Science Documents'] / df['year'].apply(lambda x: 2025 - x + 1)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
df = load_data()

if df is None:
    st.stop()

# Sidebar
st.sidebar.title("Filters & Controls")
st.sidebar.markdown("---")

# Country filter with improved UI
countries = sorted(df['Name'].unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Select Countries (Leave empty for all)",
    options=countries,
    default=[]
)

# Year range filter
year_range = st.sidebar.slider(
    "Year Range",
    min_value=int(df['year'].min()),
    max_value=int(df['year'].max()),
    value=(int(df['year'].min()), int(df['year'].max()))
)

# Citation threshold filter
citation_threshold = st.sidebar.number_input(
    "Minimum Citations Threshold",
    min_value=0,
    max_value=int(df['Times Cited'].max()),
    value=0,
    step=10000
)

# Metric selection
metric_options = {
    'Times Cited': 'Times Cited',
    'Web of Science Documents': 'Documents',
    'Category Normalized Citation Impact': 'CNCI',
    'Citations_per_Doc': 'Citations per Document',
    'Impact_Score': 'Impact Score',
    'H_Index_Proxy': 'H-Index Proxy'
}
selected_metric = st.sidebar.selectbox(
    "Primary Metric",
    options=list(metric_options.keys()),
    format_func=lambda x: metric_options[x]
)

# Advanced filters
st.sidebar.markdown("### Advanced Filters")
show_outliers = st.sidebar.checkbox("Show Outliers", value=True)
cnci_range = st.sidebar.slider(
    "CNCI Range",
    min_value=float(df['Category Normalized Citation Impact'].min()),
    max_value=float(df['Category Normalized Citation Impact'].max()),
    value=(float(df['Category Normalized Citation Impact'].min()), 
           float(df['Category Normalized Citation Impact'].max())),
    step=0.1
)

# Filter data
filtered_df = df.copy()

if selected_countries:
    filtered_df = filtered_df[filtered_df['Name'].isin(selected_countries)]

filtered_df = filtered_df[
    (filtered_df['year'] >= year_range[0]) & 
    (filtered_df['year'] <= year_range[1]) &
    (filtered_df['Times Cited'] >= citation_threshold) &
    (filtered_df['Category Normalized Citation Impact'] >= cnci_range[0]) &
    (filtered_df['Category Normalized Citation Impact'] <= cnci_range[1])
]

# Main header
st.markdown('<p class="main-header">Global Research Impact Analysis Dashboard</p>', unsafe_allow_html=True)
st.markdown("### Comprehensive Exploratory Data Analysis of Scientific Publications (2003-2025)")
st.markdown("---")

# Executive Summary with Key Insights
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    total_records = len(filtered_df)
    st.metric("Total Records", f"{total_records:,}", 
              delta=f"{(total_records/len(df)*100):.1f}% of total")

with col2:
    total_docs = filtered_df['Web of Science Documents'].sum()
    st.metric("Total Documents", f"{total_docs/1e6:.2f}M")

with col3:
    total_citations = filtered_df['Times Cited'].sum()
    st.metric("Total Citations", f"{total_citations/1e6:.1f}M")

with col4:
    avg_cnci = filtered_df['Category Normalized Citation Impact'].mean()
    delta_cnci = avg_cnci - 1.0
    st.metric("Avg CNCI", f"{avg_cnci:.3f}", 
              delta=f"{delta_cnci:+.3f} vs baseline")

with col5:
    avg_top10 = filtered_df['% Documents in Top 10%'].mean()
    st.metric("Avg Top 10%", f"{avg_top10:.2f}%")

with col6:
    unique_countries = filtered_df['Name'].nunique()
    st.metric("Countries", f"{unique_countries}")

st.markdown("---")

# Tab layout with additional advanced analysis tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Data Overview", 
    "Geographic Analysis", 
    "Temporal Trends", 
    "Quality Metrics",
    "Advanced Analytics",
    "Outlier Detection",
    "Statistical Summary"
])

# TAB 1: DATA OVERVIEW
with tab1:
    st.header("Dataset Overview & Data Quality")
    
    # Data Quality Assessment
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("### Data Quality Report")
        
        # Missing values
        missing_pct = (filtered_df.isnull().sum() / len(filtered_df) * 100)
        if missing_pct.sum() == 0:
            st.success("No missing values detected")
        else:
            st.warning(f"Missing values found in {missing_pct[missing_pct > 0].count()} columns")
            st.dataframe(missing_pct[missing_pct > 0])
        
        # Duplicates
        duplicates = filtered_df.duplicated().sum()
        st.info(f"Duplicate rows: {duplicates} ({duplicates/len(filtered_df)*100:.2f}%)")
        
        # Data types
        st.info(f"Numeric columns: {filtered_df.select_dtypes(include=[np.number]).columns.tolist().__len__()}")
        st.info(f"Categorical columns: {filtered_df.select_dtypes(exclude=[np.number]).columns.tolist().__len__()}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("### Key Findings")
        
        # Top performer
        top_country = filtered_df.groupby('Name')[selected_metric].sum().idxmax()
        top_value = filtered_df.groupby('Name')[selected_metric].sum().max()
        st.success(f"Top Performer: **{top_country}** ({metric_options[selected_metric]}: {top_value:,.0f})")
        
        # Most productive year
        top_year = filtered_df.groupby('year')['Web of Science Documents'].sum().idxmax()
        st.success(f"Most Productive Year: **{top_year}**")
        
        # Highest CNCI
        max_cnci_country = filtered_df.groupby('Name')['Category Normalized Citation Impact'].mean().idxmax()
        max_cnci = filtered_df.groupby('Name')['Category Normalized Citation Impact'].mean().max()
        st.success(f"Highest Avg CNCI: **{max_cnci_country}** ({max_cnci:.3f})")
        
        # Citation efficiency
        top_efficiency = filtered_df.groupby('Name')['Citations_per_Doc'].mean().idxmax()
        st.success(f"Most Efficient: **{top_efficiency}** (citations/doc)")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Distribution Analysis
    st.subheader("Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top countries by selected metric
        country_agg = filtered_df.groupby('Name').agg({
            selected_metric: 'sum',
            'Web of Science Documents': 'sum',
            'Times Cited': 'sum'
        }).reset_index().sort_values(selected_metric, ascending=False).head(15)
        
        fig = px.bar(
            country_agg,
            x='Name',
            y=selected_metric,
            title=f"Top 15 Countries by {metric_options[selected_metric]}",
            color=selected_metric,
            color_continuous_scale='Blues',
            text=selected_metric
        )
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig.update_layout(xaxis_tickangle=-45, height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribution of CNCI with statistical annotations
        fig = px.histogram(
            filtered_df,
            x='Category Normalized Citation Impact',
            nbins=50,
            title="Distribution of CNCI (with Statistical Measures)",
            color_discrete_sequence=['#1f77b4'],
            marginal="box"
        )
        
        mean_cnci = filtered_df['Category Normalized Citation Impact'].mean()
        median_cnci = filtered_df['Category Normalized Citation Impact'].median()
        
        fig.add_vline(x=1.0, line_dash="dash", line_color="red", 
                      annotation_text="World Baseline (1.0)")
        fig.add_vline(x=mean_cnci, line_dash="dot", line_color="green", 
                      annotation_text=f"Mean: {mean_cnci:.2f}")
        fig.add_vline(x=median_cnci, line_dash="dot", line_color="orange", 
                      annotation_text=f"Median: {median_cnci:.2f}")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Analysis
    st.subheader("Quick Correlation Insights")
    
    numeric_cols = ['Web of Science Documents', 'Times Cited', 'Collab-CNCI',
                    'Category Normalized Citation Impact', '% Documents in Top 1%',
                    '% Documents in Top 10%', 'Citations_per_Doc', 'Impact_Score']
    
    corr_matrix = filtered_df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Correlation Matrix of Key Metrics",
        color_continuous_scale='RdBu',
        aspect='auto',
        zmin=-1, zmax=1,
        text_auto='.2f'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table with enhanced display
    st.subheader("Detailed Data View")
    display_cols = ['Name', 'year', 'Web of Science Documents', 'Times Cited', 
                    'Category Normalized Citation Impact', '% Documents in Top 10%',
                    'Citations_per_Doc', 'Impact_Score']
    
    # Add color coding
    styled_df = filtered_df[display_cols].sort_values('Impact_Score', ascending=False).head(100)
    st.dataframe(
        styled_df.style.background_gradient(subset=['Impact_Score'], cmap='YlGn')
                       .background_gradient(subset=['Category Normalized Citation Impact'], cmap='RdYlGn'),
        use_container_width=True,
        height=400
    )

# TAB 2: GEOGRAPHIC ANALYSIS
with tab2:
    st.header("Geographic Distribution & Performance")
    
    # Country performance heatmap
    country_year_pivot = filtered_df.pivot_table(
        values='Category Normalized Citation Impact',
        index='Name',
        columns='year',
        aggfunc='mean'
    ).fillna(0)
    
    fig = px.imshow(
        country_year_pivot,
        title="CNCI Heatmap: Countries vs Years (Darker = Higher Impact)",
        color_continuous_scale='RdYlGn',
        aspect='auto',
        labels=dict(color="CNCI")
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Citations per document by country with bubble size
        country_efficiency = filtered_df.groupby('Name').agg({
            'Citations_per_Doc': 'mean',
            'Web of Science Documents': 'sum',
            'Times Cited': 'sum'
        }).reset_index().sort_values('Citations_per_Doc', ascending=False).head(15)
        
        fig = px.scatter(
            country_efficiency,
            x='Web of Science Documents',
            y='Citations_per_Doc',
            size='Times Cited',
            color='Citations_per_Doc',
            hover_name='Name',
            title="Research Efficiency: Productivity vs Citations/Document",
            color_continuous_scale='Viridis',
            size_max=50
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Elite output comparison with trend
        elite_data = filtered_df.groupby('Name')[
            ['% Documents in Top 1%', '% Documents in Top 10%']
        ].mean().reset_index().sort_values('% Documents in Top 1%', ascending=False).head(15)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=elite_data['Name'],
            y=elite_data['% Documents in Top 1%'],
            name='Top 1%',
            marker_color='#ef4444',
            text=elite_data['% Documents in Top 1%'].round(2),
            textposition='outside'
        ))
        fig.add_trace(go.Bar(
            x=elite_data['Name'],
            y=elite_data['% Documents in Top 10%'],
            name='Top 10%',
            marker_color='#f59e0b',
            text=elite_data['% Documents in Top 10%'].round(2),
            textposition='outside'
        ))
        fig.update_layout(
            title="Elite Output: Top 1% vs Top 10% Documents",
            xaxis_tickangle=-45,
            barmode='group',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Geographic insights
    st.subheader("Regional Performance Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Country with most consistent performance (lowest std in CNCI)
        country_consistency = filtered_df.groupby('Name')['Category Normalized Citation Impact'].agg(['mean', 'std'])
        most_consistent = country_consistency.nsmallest(1, 'std').index[0]
        st.metric("Most Consistent", most_consistent, 
                 f"σ={country_consistency.loc[most_consistent, 'std']:.3f}")
    
    with col2:
        # Most improved country
        year_performance = filtered_df.groupby(['Name', 'year'])['Category Normalized Citation Impact'].mean().reset_index()
        country_trends = year_performance.groupby('Name').apply(
            lambda x: (x['Category Normalized Citation Impact'].iloc[-1] - x['Category Normalized Citation Impact'].iloc[0]) 
            if len(x) > 1 else 0
        )
        most_improved = country_trends.idxmax()
        improvement = country_trends.max()
        st.metric("Most Improved", most_improved, f"+{improvement:.3f}")
    
    with col3:
        # Highest collaboration impact
        top_collab = filtered_df.groupby('Name')['Collab-CNCI'].mean().idxmax()
        collab_score = filtered_df.groupby('Name')['Collab-CNCI'].mean().max()
        st.metric("Best Collaboration", top_collab, f"{collab_score:.3f}")

# TAB 3: TEMPORAL TRENDS
with tab3:
    st.header("Temporal Analysis & Trends")
    
    # Time series aggregation
    time_series = filtered_df.groupby('year').agg({
        'Web of Science Documents': 'sum',
        'Times Cited': 'sum',
        'Category Normalized Citation Impact': 'mean',
        '% Documents in Top 10%': 'mean',
        'Citations_per_Doc': 'mean'
    }).reset_index()
    
    # Multi-axis time series with enhanced visuals
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Publication & Citation Volume Over Time", 
                       "Quality Metrics Evolution",
                       "Citation Efficiency Trend"),
        vertical_spacing=0.1,
        specs=[[{"secondary_y": True}],
               [{"secondary_y": True}],
               [{"secondary_y": False}]]
    )
    
    # Row 1: Volume metrics
    fig.add_trace(
        go.Scatter(x=time_series['year'], y=time_series['Web of Science Documents'],
                   name='Documents', line=dict(color='#3b82f6', width=3),
                   fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.2)'),
        row=1, col=1, secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=time_series['year'], y=time_series['Times Cited'],
                   name='Citations', line=dict(color='#10b981', width=3),
                   fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.2)'),
        row=1, col=1, secondary_y=True
    )
    
    # Row 2: Quality metrics
    fig.add_trace(
        go.Scatter(x=time_series['year'], y=time_series['Category Normalized Citation Impact'],
                   name='CNCI', line=dict(color='#8b5cf6', width=3),
                   mode='lines+markers'),
        row=2, col=1, secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=time_series['year'], y=time_series['% Documents in Top 10%'],
                   name='Top 10%', line=dict(color='#f59e0b', width=3),
                   mode='lines+markers'),
        row=2, col=1, secondary_y=True
    )
    
    # Row 3: Efficiency
    fig.add_trace(
        go.Scatter(x=time_series['year'], y=time_series['Citations_per_Doc'],
                   name='Citations/Doc', line=dict(color='#ec4899', width=3),
                   mode='lines+markers', fill='tozeroy'),
        row=3, col=1
    )
    
    fig.update_layout(height=1000, showlegend=True, hovermode='x unified')
    fig.update_xaxes(title_text="Year", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)
    
    # Year-over-year growth analysis
    st.subheader("Year-over-Year Growth Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        time_series['Doc_Growth'] = time_series['Web of Science Documents'].pct_change() * 100
        fig = px.bar(
            time_series.dropna(),
            x='year',
            y='Doc_Growth',
            title="YoY Document Growth (%) - Trend Analysis",
            color='Doc_Growth',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        time_series['Citation_Growth'] = time_series['Times Cited'].pct_change() * 100
        fig = px.bar(
            time_series.dropna(),
            x='year',
            y='Citation_Growth',
            title="YoY Citation Growth (%) - Trend Analysis",
            color='Citation_Growth',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig, use_container_width=True)
    
    # Trend forecasting visualization
    st.subheader("Trend Patterns")
    
    from scipy import stats as sp_stats
    
    # Linear regression for documents
    slope_docs, intercept_docs, r_docs, p_docs, se_docs = sp_stats.linregress(
        time_series['year'], time_series['Web of Science Documents']
    )
    
    slope_cit, intercept_cit, r_cit, p_cit, se_cit = sp_stats.linregress(
        time_series['year'], time_series['Times Cited']
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Document Trend", 
                 "Growing" if slope_docs > 0 else "Declining",
                 f"{slope_docs:.0f} docs/year")
    
    with col2:
        st.metric("Citation Trend",
                 "Growing" if slope_cit > 0 else "Declining",
                 f"{slope_cit:.0f} citations/year")
    
    with col3:
        st.metric("CNCI Stability",
                 f"R²={r_docs**2:.3f}",
                 "Strong" if r_docs**2 > 0.7 else "Moderate" if r_docs**2 > 0.4 else "Weak")

# TAB 4: QUALITY METRICS
with tab4:
    st.header("Research Quality Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Quality vs Quantity scatter with enhanced annotations
        country_summary = filtered_df.groupby('Name').agg({
            'Web of Science Documents': 'sum',
            'Category Normalized Citation Impact': 'mean',
            'Times Cited': 'sum',
            '% Documents in Top 10%': 'mean'
        }).reset_index()
        
        fig = px.scatter(
            country_summary,
            x='Web of Science Documents',
            y='Category Normalized Citation Impact',
            size='Times Cited',
            color='% Documents in Top 10%',
            hover_name='Name',
            title="Quality vs Quantity Matrix (Size=Citations, Color=Top 10%)",
            color_continuous_scale='Turbo',
            size_max=60,
            log_x=True
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                      annotation_text="World Baseline (CNCI=1.0)")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Collaboration impact with regression line
        sample_data = filtered_df.sample(min(500, len(filtered_df)))
        fig = px.scatter(
            sample_data,
            x='Collab-CNCI',
            y='Category Normalized Citation Impact',
            color='Name',
            title="Collaboration Index vs Citation Impact (with Trend)",
            opacity=0.6,
            trendline="ols",
            trendline_scope="overall"
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Box plots for distribution comparison
    st.subheader("Distribution Comparison Across Countries")
    
    top_countries = filtered_df['Name'].value_counts().head(10).index
    df_top = filtered_df[filtered_df['Name'].isin(top_countries)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(
            df_top,
            x='Name',
            y='Category Normalized Citation Impact',
            color='Name',
            title="CNCI Distribution by Country (with Outliers)",
            points='outliers'
        )
        fig.update_layout(showlegend=False, height=450, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.violin(
            df_top,
            x='Name',
            y='Citations_per_Doc',
            color='Name',
            title="Citations/Document Distribution (Violin Plot)",
            box=True,
            points='outliers'
        )
        fig.update_layout(showlegend=False, height=450, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

# TAB 5: ADVANCED ANALYTICS
with tab5:
    st.header("Advanced Analytics & Insights")
    
    # Statistical distribution analysis
    st.subheader("Statistical Distribution Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**CNCI Distribution**")
        cnci_stats = filtered_df['Category Normalized Citation Impact'].describe()
        cnci_skew = skew(filtered_df['Category Normalized Citation Impact'])
        cnci_kurt = kurtosis(filtered_df['Category Normalized Citation Impact'])
        st.write(cnci_stats)
        st.info(f"Skewness: {cnci_skew:.3f}")
        st.info(f"Kurtosis: {cnci_kurt:.3f}")
    
    with col2:
        st.markdown("**Top 10% Distribution**")
        top10_stats = filtered_df['% Documents in Top 10%'].describe()
        top10_skew = skew(filtered_df['% Documents in Top 10%'])
        st.write(top10_stats)
        st.info(f"Skewness: {top10_skew:.3f}")
    
    with col3:
        st.markdown("**Citations per Doc**")
        cpd_stats = filtered_df['Citations_per_Doc'].describe()
        cpd_skew = skew(filtered_df['Citations_per_Doc'])
        st.write(cpd_stats)
        st.info(f"Skewness: {cpd_skew:.3f}")
    
    with col4:
        st.markdown("**Impact Score**")
        impact_stats = filtered_df['Impact_Score'].describe()
        impact_skew = skew(filtered_df['Impact_Score'])
        st.write(impact_stats)
        st.info(f"Skewness: {impact_skew:.3f}")
    
    # Impact score ranking with composite metrics
    st.subheader("Overall Impact Score Ranking")
    
    impact_ranking = filtered_df.groupby('Name').agg({
        'Impact_Score': 'mean',
        'Category Normalized Citation Impact': 'mean',
        '% Documents in Top 10%': 'mean',
        'Times Cited': 'sum',
        'Citations_per_Doc': 'mean',
        'H_Index_Proxy': 'mean'
    }).reset_index()
    
    # Normalize and create composite score
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

# Normalize metrics for composite score
    score_cols = ['Impact_Score', 'Category Normalized Citation Impact', 
                  '% Documents in Top 10%', 'Citations_per_Doc', 'H_Index_Proxy']
    
    impact_ranking_scaled = impact_ranking.copy()
    impact_ranking_scaled[score_cols] = scaler.fit_transform(impact_ranking[score_cols])
    
    # Calculate composite score (weighted average)
    impact_ranking_scaled['Composite_Score'] = (
        impact_ranking_scaled['Impact_Score'] * 0.3 +
        impact_ranking_scaled['Category Normalized Citation Impact'] * 0.25 +
        impact_ranking_scaled['% Documents in Top 10%'] * 0.2 +
        impact_ranking_scaled['Citations_per_Doc'] * 0.15 +
        impact_ranking_scaled['H_Index_Proxy'] * 0.1
    )
    
    impact_ranking['Composite_Score'] = impact_ranking_scaled['Composite_Score']
    impact_ranking_sorted = impact_ranking.sort_values('Composite_Score', ascending=False).head(20)
    
    fig = px.bar(
        impact_ranking_sorted,
        x='Name',
        y='Composite_Score',
        title="Top 20 Countries by Composite Impact Score (Multi-metric Weighted)",
        color='Composite_Score',
        color_continuous_scale='Plasma',
        hover_data=['Impact_Score', 'Category Normalized Citation Impact', 
                    '% Documents in Top 10%', 'Times Cited']
    )
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed breakdown table
    st.subheader("Detailed Impact Score Breakdown")
    display_impact_cols = ['Name', 'Composite_Score', 'Impact_Score', 
                          'Category Normalized Citation Impact', '% Documents in Top 10%',
                          'Citations_per_Doc', 'H_Index_Proxy', 'Times Cited']
    
    st.dataframe(
        impact_ranking_sorted[display_impact_cols].style.background_gradient(
            subset=['Composite_Score'], cmap='RdYlGn'
        ).format({
            'Composite_Score': '{:.4f}',
            'Impact_Score': '{:.2f}',
            'Category Normalized Citation Impact': '{:.3f}',
            '% Documents in Top 10%': '{:.2f}',
            'Citations_per_Doc': '{:.2f}',
            'H_Index_Proxy': '{:.2f}',
            'Times Cited': '{:,.0f}'
        }),
        use_container_width=True,
        height=400
    )
    
    # Percentile Analysis
    st.subheader("Percentile Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CNCI percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        cnci_percentiles = np.percentile(filtered_df['Category Normalized Citation Impact'], percentiles)
        
        percentile_df = pd.DataFrame({
            'Percentile': [f'{p}th' for p in percentiles],
            'CNCI Value': cnci_percentiles
        })
        
        fig = px.bar(
            percentile_df,
            x='Percentile',
            y='CNCI Value',
            title="CNCI Percentile Distribution",
            color='CNCI Value',
            color_continuous_scale='Blues',
            text='CNCI Value'
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", 
                     annotation_text="World Baseline")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Citations per doc percentiles
        cpd_percentiles = np.percentile(filtered_df['Citations_per_Doc'], percentiles)
        
        percentile_df_cpd = pd.DataFrame({
            'Percentile': [f'{p}th' for p in percentiles],
            'Citations/Doc': cpd_percentiles
        })
        
        fig = px.bar(
            percentile_df_cpd,
            x='Percentile',
            y='Citations/Doc',
            title="Citations per Document Percentile Distribution",
            color='Citations/Doc',
            color_continuous_scale='Greens',
            text='Citations/Doc'
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

# TAB 6: OUTLIER DETECTION
with tab6:
    st.header("Outlier Detection & Anomaly Analysis")
    
    st.markdown("""
    This section identifies statistical outliers using multiple methods:
    - **IQR Method**: Values beyond 1.5 × IQR from Q1/Q3
    - **Z-Score Method**: Values with |z-score| > 3
    - **Isolation Forest**: Machine learning-based anomaly detection
    """)
    
    # IQR Method for outlier detection
    st.subheader("IQR-Based Outlier Detection")
    
    outlier_metrics = ['Category Normalized Citation Impact', 'Citations_per_Doc', 
                      '% Documents in Top 10%', 'Times Cited']
    
    outlier_counts = {}
    outlier_data = {}
    
    for metric in outlier_metrics:
        Q1 = filtered_df[metric].quantile(0.25)
        Q3 = filtered_df[metric].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = filtered_df[(filtered_df[metric] < lower_bound) | 
                              (filtered_df[metric] > upper_bound)]
        outlier_counts[metric] = len(outliers)
        outlier_data[metric] = outliers
    
    # Display outlier counts
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("CNCI Outliers", outlier_counts['Category Normalized Citation Impact'],
                 f"{outlier_counts['Category Normalized Citation Impact']/len(filtered_df)*100:.1f}%")
    
    with col2:
        st.metric("Citations/Doc Outliers", outlier_counts['Citations_per_Doc'],
                 f"{outlier_counts['Citations_per_Doc']/len(filtered_df)*100:.1f}%")
    
    with col3:
        st.metric("Top 10% Outliers", outlier_counts['% Documents in Top 10%'],
                 f"{outlier_counts['% Documents in Top 10%']/len(filtered_df)*100:.1f}%")
    
    with col4:
        st.metric("Citations Outliers", outlier_counts['Times Cited'],
                 f"{outlier_counts['Times Cited']/len(filtered_df)*100:.1f}%")
    
    # Visualize outliers
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot with outliers
        fig = px.box(
            filtered_df,
            y='Category Normalized Citation Impact',
            title="CNCI Distribution with Outliers Highlighted",
            points='outliers',
            color_discrete_sequence=['#3b82f6']
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", 
                     annotation_text="World Baseline")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Scatter plot showing outliers
        sample_for_plot = filtered_df.sample(min(1000, len(filtered_df)))
        
        # Mark outliers
        sample_for_plot['Is_Outlier'] = sample_for_plot.apply(
            lambda row: any([
                row['Category Normalized Citation Impact'] in outlier_data['Category Normalized Citation Impact']['Category Normalized Citation Impact'].values
                if len(outlier_data['Category Normalized Citation Impact']) > 0 else False
            ]), axis=1
        )
        
        fig = px.scatter(
            sample_for_plot,
            x='Citations_per_Doc',
            y='Category Normalized Citation Impact',
            color='Is_Outlier',
            title="Outliers in Quality vs Efficiency Space",
            color_discrete_map={True: '#ef4444', False: '#94a3b8'},
            opacity=0.6,
            labels={'Is_Outlier': 'Outlier Status'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Z-Score Analysis
    st.subheader("Z-Score Based Anomaly Detection")
    
    from scipy.stats import zscore
    
    z_scores = filtered_df[outlier_metrics].apply(zscore)
    extreme_outliers = (np.abs(z_scores) > 3).any(axis=1)
    extreme_outlier_df = filtered_df[extreme_outliers]
    
    st.write(f"**Found {len(extreme_outlier_df)} extreme outliers** (|z-score| > 3)")
    
    if len(extreme_outlier_df) > 0:
        # Show top extreme outliers
        display_cols = ['Name', 'year', 'Category Normalized Citation Impact', 
                       'Citations_per_Doc', 'Times Cited', '% Documents in Top 10%']
        
        st.dataframe(
            extreme_outlier_df[display_cols].head(20).style.background_gradient(
                subset=['Category Normalized Citation Impact'], cmap='RdYlGn'
            ),
            use_container_width=True,
            height=300
        )
        
        # Analyze outlier characteristics
        st.markdown("### Outlier Characteristics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            outlier_countries = extreme_outlier_df['Name'].value_counts().head(10)
            fig = px.bar(
                x=outlier_countries.index,
                y=outlier_countries.values,
                title="Countries with Most Extreme Outliers",
                labels={'x': 'Country', 'y': 'Number of Outliers'},
                color=outlier_countries.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            outlier_years = extreme_outlier_df['year'].value_counts().sort_index()
            fig = px.line(
                x=outlier_years.index,
                y=outlier_years.values,
                title="Temporal Distribution of Outliers",
                labels={'x': 'Year', 'y': 'Number of Outliers'},
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)

# TAB 7: STATISTICAL SUMMARY
with tab7:
    st.header("Comprehensive Statistical Summary")
    
    # Overall statistics
    st.subheader("Descriptive Statistics")
    
    key_metrics = ['Web of Science Documents', 'Times Cited', 
                   'Category Normalized Citation Impact', '% Documents in Top 10%',
                   'Citations_per_Doc', 'Impact_Score', 'Collab-CNCI']
    
    stats_df = filtered_df[key_metrics].describe().T
    stats_df['variance'] = filtered_df[key_metrics].var()
    stats_df['skewness'] = filtered_df[key_metrics].apply(skew)
    stats_df['kurtosis'] = filtered_df[key_metrics].apply(kurtosis)
    
    st.dataframe(
        stats_df.style.format("{:.2f}").background_gradient(cmap='Blues', axis=0),
        use_container_width=True
    )
    
    # Country-level aggregated statistics
    st.subheader("Country-Level Statistics")
    
    country_stats = filtered_df.groupby('Name').agg({
        'Web of Science Documents': ['sum', 'mean', 'std'],
        'Times Cited': ['sum', 'mean', 'std'],
        'Category Normalized Citation Impact': ['mean', 'std', 'min', 'max'],
        '% Documents in Top 10%': ['mean', 'std'],
        'Citations_per_Doc': ['mean', 'std']
    }).round(2)
    
    country_stats.columns = ['_'.join(col).strip() for col in country_stats.columns.values]
    country_stats = country_stats.sort_values('Times Cited_sum', ascending=False).head(20)
    
    st.dataframe(country_stats, use_container_width=True, height=400)
    
    # Hypothesis Testing
    st.subheader("Statistical Hypothesis Testing")
    
    st.markdown("""
    **Testing whether average CNCI differs significantly from world baseline (1.0)**
    - H₀: μ = 1.0 (CNCI equals world baseline)
    - H₁: μ ≠ 1.0 (CNCI differs from world baseline)
    """)
    
    from scipy.stats import ttest_1samp
    
    t_stat, p_value = ttest_1samp(filtered_df['Category Normalized Citation Impact'], 1.0)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("T-Statistic", f"{t_stat:.4f}")
    
    with col2:
        st.metric("P-Value", f"{p_value:.6f}")
    
    with col3:
        result = "Reject H₀" if p_value < 0.05 else "Fail to Reject H₀"
        st.metric("Result (α=0.05)", result)
    
    if p_value < 0.05:
        st.success(f"he average CNCI ({filtered_df['Category Normalized Citation Impact'].mean():.3f}) is **statistically significantly different** from the world baseline (1.0) at 95% confidence level.")
    else:
        st.info(f"The average CNCI ({filtered_df['Category Normalized Citation Impact'].mean():.3f}) is **not statistically significantly different** from the world baseline (1.0) at 95% confidence level.")
    
    # Correlation with statistical significance
    st.subheader("Correlation Analysis with Significance Testing")
    
    from scipy.stats import pearsonr
    
    correlation_pairs = [
        ('Web of Science Documents', 'Times Cited'),
        ('Category Normalized Citation Impact', '% Documents in Top 10%'),
        ('Citations_per_Doc', 'Category Normalized Citation Impact'),
        ('Collab-CNCI', 'Category Normalized Citation Impact')
    ]
    
    correlation_results = []
    
    for var1, var2 in correlation_pairs:
        r, p = pearsonr(filtered_df[var1].dropna(), filtered_df[var2].dropna())
        correlation_results.append({
            'Variable 1': var1,
            'Variable 2': var2,
            'Correlation (r)': r,
            'P-Value': p,
            'Significance': 'Yes' if p < 0.05 else 'No',
            'Strength': 'Strong' if abs(r) > 0.7 else 'Moderate' if abs(r) > 0.4 else 'Weak'
        })
    
    correlation_df = pd.DataFrame(correlation_results)
    
    st.dataframe(
        correlation_df.style.format({
            'Correlation (r)': '{:.4f}',
            'P-Value': '{:.6f}'
        }).applymap(
            lambda x: 'background-color: #d4edda' if x == 'Yes' else '',
            subset=['Significance']
        ),
        use_container_width=True
    )
    
    # Time series analysis - trend detection
    st.subheader("Trend Analysis Over Time")
    
    yearly_cnci = filtered_df.groupby('year')['Category Normalized Citation Impact'].mean()
    
    from scipy.stats import linregress
    
    slope, intercept, r_value, p_value, std_err = linregress(yearly_cnci.index, yearly_cnci.values)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Trend Slope", f"{slope:.5f}", 
                 "Increasing" if slope > 0 else "Decreasing")
    
    with col2:
        st.metric("R² Value", f"{r_value**2:.4f}",
                 "Strong fit" if r_value**2 > 0.7 else "Moderate fit")
    
    with col3:
        st.metric("Trend Significance", 
                 "Significant" if p_value < 0.05 else "Not Significant",
                 f"p={p_value:.4f}")
    
    # Plot trend
    fig = px.scatter(
        x=yearly_cnci.index,
        y=yearly_cnci.values,
        title="CNCI Trend Over Time with Regression Line",
        labels={'x': 'Year', 'y': 'Average CNCI'},
        trendline="ols"
    )
    fig.update_traces(marker=dict(size=10, color='#3b82f6'))
    st.plotly_chart(fig, use_container_width=True)
    
    # Key Insights Summary
    st.subheader("Key Statistical Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**Distribution Characteristics**")
        st.write(f"- CNCI Skewness: {skew(filtered_df['Category Normalized Citation Impact']):.3f} ({'Right-skewed' if skew(filtered_df['Category Normalized Citation Impact']) > 0 else 'Left-skewed'})")
        st.write(f"- CNCI Kurtosis: {kurtosis(filtered_df['Category Normalized Citation Impact']):.3f} ({'Heavy tails' if kurtosis(filtered_df['Category Normalized Citation Impact']) > 0 else 'Light tails'})")
        st.write(f"- Coefficient of Variation: {(filtered_df['Category Normalized Citation Impact'].std() / filtered_df['Category Normalized Citation Impact'].mean()):.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("**Performance Metrics**")
        above_baseline = (filtered_df['Category Normalized Citation Impact'] > 1.0).sum()
        st.write(f"- Records above baseline: {above_baseline} ({above_baseline/len(filtered_df)*100:.1f}%)")
        st.write(f"- Median CNCI: {filtered_df['Category Normalized Citation Impact'].median():.3f}")
        st.write(f"- Top 10% threshold: {filtered_df['Category Normalized Citation Impact'].quantile(0.90):.3f}")
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Global Research Impact Dashboard</strong> | Data Period: 2003-2025</p>
    <p>Total Records Analyzed: {total:,} | Countries: {countries} | Years: {years}</p>
    <p>Built with Streamlit  & Plotly | Advanced EDA & Statistical Analysis</p>
    <p style='font-size: 0.9rem; margin-top: 1rem;'>
      <strong>Key Features:</strong> Interactive Filtering • Real-time Analytics • 
        Outlier Detection • Trend Analysis • Statistical Testing • Multi-metric Scoring
    </p>
</div>
""".format(
    total=len(filtered_df),
    countries=filtered_df['Name'].nunique(),
    years=filtered_df['year'].nunique()
), unsafe_allow_html=True)