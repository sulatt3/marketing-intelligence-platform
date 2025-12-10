"""
Marketing Intelligence Platform - Main UI
Integrated platform with Competitive Intelligence + Customer Intelligence
"""

import streamlit as st
import os
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from difflib import SequenceMatcher

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import google.generativeai as genai
from newsapi import NewsApiClient
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import requests

# Page config
st.set_page_config(
    page_title="Marketing Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.5-pro")

# Wikipedia API headers
WIKI_HEADERS = {
    'User-Agent': 'Marketing Intelligence Platform/1.0 (https://github.com/sulatt3/marketing-intelligence-platform; su.h.latt3@gmail.com)'
}

# ==================== COMPETITIVE INTELLIGENCE FUNCTIONS ====================

@st.cache_data(ttl=3600)
def fetch_news(company: str) -> List[Dict[str, Any]]:
    try:
        newsapi = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))
        from_date = (datetime.now() - timedelta(days=29)).strftime("%Y-%m-%d")
        articles = newsapi.get_everything(q=company, language="en", sort_by="publishedAt", from_param=from_date, page_size=100)
        return [{"source": "News", "title": a["title"], "url": a["url"],
                 "date": a["publishedAt"][:10] if a["publishedAt"] else "Unknown",
                 "snippet": a["description"] or a["content"] or "",
                 "full_content": a["content"] or ""}
                for a in articles.get("articles", [])]
    except Exception as e:
        st.warning(f"NewsAPI error: {e}")
        return []

@st.cache_data(ttl=3600)
def fetch_wikipedia_pageviews(company: str) -> List[Dict[str, Any]]:
    try:
        search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={company}&limit=1&format=json"
        search_resp = requests.get(search_url, headers=WIKI_HEADERS, timeout=10)
        
        if search_resp.status_code != 200:
            return [{"source": "Wikipedia", "total_pageviews": None}]
        
        titles = search_resp.json()[1]
        if not titles:
            return [{"source": "Wikipedia", "total_pageviews": None}]
        
        title = titles[0].replace(" ", "_")
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        
        pageviews_url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{title}/daily/{start_date}/{end_date}"
        pageviews_resp = requests.get(pageviews_url, headers=WIKI_HEADERS, timeout=10)
        
        if pageviews_resp.status_code != 200:
            return [{"source": "Wikipedia", "total_pageviews": None}]
        
        data = pageviews_resp.json()
        views = [item['views'] for item in data.get('items', [])]
        
        if not views:
            return [{"source": "Wikipedia", "total_pageviews": None}]
        
        total_views = sum(views)
        avg_views = int(total_views / len(views))
        peak_views = max(views)
        recent_avg = sum(views[-7:]) / 7 if len(views) >= 7 else avg_views
        older_avg = sum(views[:-7]) / len(views[:-7]) if len(views) > 7 else avg_views
        trend = "rising" if recent_avg > older_avg * 1.1 else ("declining" if recent_avg < older_avg * 0.9 else "stable")
        
        return [{
            "source": "Wikipedia",
            "total_pageviews": total_views,
            "avg_daily_pageviews": avg_views,
            "peak_daily_pageviews": peak_views,
            "trend_direction": trend,
            "article_title": titles[0]
        }]
    except Exception:
        return [{"source": "Wikipedia", "total_pageviews": None}]

def collect_all_data(company: str, use_news: bool, use_wikipedia: bool) -> List[Dict[str, Any]]:
    fetchers = []
    if use_news:
        fetchers.append(fetch_news)
    if use_wikipedia:
        fetchers.append(fetch_wikipedia_pageviews)
    all_data = []
    if fetchers:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(f, company): f for f in fetchers}
            for future in as_completed(futures):
                data = future.result()
                all_data.extend(data if isinstance(data, list) else [data])
    return all_data

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def deduplicate_by_title(items: List[Dict[str, Any]], threshold: float = 0.85) -> List[Dict[str, Any]]:
    if not items:
        return []
    unique, seen = [], []
    for item in items:
        title = item.get("title", "")
        if title and not any(similarity(title, s) > threshold for s in seen):
            unique.append(item)
            seen.append(title)
    return unique

def score_relevance(item: Dict[str, Any], company: str) -> float:
    score, company_lower = 0.0, company.lower()
    title = item.get("title", "").lower()
    if company_lower in title:
        score += 40
    combined = f"{item.get('snippet', '').lower()} {item.get('full_content', '').lower()}"
    if company_lower in combined:
        score += 20
    score += min(combined.count(company_lower) * 5, 20)
    date_str = item.get("date", "")
    if date_str != "Unknown":
        try:
            days_old = (datetime.now() - datetime.strptime(date_str, "%Y-%m-%d")).days
            score += 15 if days_old <= 7 else (10 if days_old <= 14 else (5 if days_old <= 30 else 0))
        except:
            pass
    return min(score, 100)

def rank_and_filter(items: List[Dict[str, Any]], company: str, top_n: int = 60) -> List[Dict[str, Any]]:
    for item in items:
        item["relevance_score"] = score_relevance(item, company)
    return sorted(items, key=lambda x: x.get("relevance_score", 0), reverse=True)[:top_n]

def process_data(raw_data: List[Dict[str, Any]], company: str, top_n: int = 60) -> List[Dict[str, Any]]:
    return rank_and_filter(deduplicate_by_title(raw_data), company, top_n=top_n)

def analyze_sentiment(text: str) -> str:
    pos = ["launch", "growth", "success", "innovation", "partnership", "funding", "expansion", "breakthrough"]
    neg = ["loss", "decline", "controversy", "lawsuit", "criticism", "failure", "layoff", "scandal"]
    t = text.lower()
    p, n = sum(1 for w in pos if w in t), sum(1 for w in neg if w in t)
    return "positive" if p > n else ("negative" if n > p else "neutral")

def build_synthesis_prompt(processed_data: List[Dict[str, Any]], company: str) -> str:
    news = [i for i in processed_data if i.get("source") == "News"]
    wiki = [i for i in processed_data if i.get("source") == "Wikipedia"]
    news_text = "\n".join([f"[{i}] {item['title']} ({item['date']})" for i, item in enumerate(news, 1)])
    
    wiki_text = ""
    if wiki and wiki[0].get("total_pageviews"):
        w = wiki[0]
        wiki_text = f"\nWikipedia Interest (Last 30 Days):\n- Article: {w.get('article_title', 'N/A')}\n- Total Pageviews: {w['total_pageviews']:,}\n- Avg Daily Pageviews: {w['avg_daily_pageviews']:,}\n- Peak Daily: {w['peak_daily_pageviews']:,}\n- Trend: {w['trend_direction']}"
    
    return f"""You are an expert competitive intelligence analyst. Analyze {company} and create a comprehensive brief.

# Competitive Intelligence Brief: {company}
**Generated:** {datetime.now().strftime("%Y-%m-%d")}

## Executive Summary
High-level overview of competitive position and recent activities

## Recent Key Moves
Strategic moves and developments in last 30 days with dates

## Product Launches & Features
New products, features, updates

## Funding, Partnerships & Hiring
Funding rounds, partnerships, acquisitions, key hires

## Market Interest Analysis
Analyze Wikipedia pageviews data. What does the trend direction indicate?

## Competitive Threats & Opportunities
What threats does {company} pose? What opportunities/weaknesses exist?

## Strategic Recommendations
3-5 actionable recommendations for competitors

## Timeline of Key Events
Chronological timeline of important events

## Sources
Articles analyzed and date range

Be specific, cite dates, focus on facts. 1000-1500 words.

DATA:
{news_text}
{wiki_text}"""

@st.cache_data(ttl=3600)
def generate_competitive_brief(company: str, use_news: bool, use_wikipedia: bool, num_articles: int) -> tuple:
    raw = collect_all_data(company, use_news, use_wikipedia)
    if not raw:
        return "# Error\n\nNo data sources enabled.", None
    processed = process_data(raw, company, top_n=num_articles)
    news_items = [i for i in processed if i.get("source") == "News"]
    if len(news_items) < 5:
        return f"# Data Quality Warning: {company}\n\nOnly {len(news_items)} news articles found. Try full legal name or enable more sources.", None
    response = model.generate_content(build_synthesis_prompt(processed, company))
    return response.text, processed

def create_competitive_timeline_viz(processed_data):
    news = [i for i in processed_data if i.get("source") == "News" and i.get("date") != "Unknown"]
    if not news:
        return None
    for i in news:
        i["sentiment"] = analyze_sentiment(i.get("title", "") + " " + i.get("snippet", ""))
    df = pd.DataFrame(news)
    df["date"] = pd.to_datetime(df["date"])
    fig = px.scatter(df, x="date", y="relevance_score", color="sentiment", size="relevance_score",
                     hover_data=["title"], title="Article Timeline",
                     color_discrete_map={"positive": "#28a745", "negative": "#dc3545", "neutral": "#6c757d"})
    fig.update_layout(height=450, hovermode="closest")
    return fig

def create_competitive_sentiment_viz(processed_data):
    news = [i for i in processed_data if i.get("source") == "News"]
    if not news:
        return None
    for i in news:
        i["sentiment"] = analyze_sentiment(i.get("title", "") + " " + i.get("snippet", ""))
    df = pd.DataFrame(news)
    counts = df["sentiment"].value_counts()
    fig = go.Figure(data=[go.Bar(x=counts.index, y=counts.values,
                                  marker_color=["#28a745" if s == "positive" else "#dc3545" if s == "negative" else "#6c757d" for s in counts.index],
                                  text=counts.values, textposition="auto")])
    fig.update_layout(title="Sentiment Distribution", xaxis_title="Sentiment", yaxis_title="Number of Articles", height=400, showlegend=False)
    return fig

# ==================== CUSTOMER INTELLIGENCE FUNCTIONS ====================

@st.cache_data
def generate_customer_data(n_customers=1000):
    """Generate synthetic customer data - Fixed at 1000 customers for optimal segmentation"""
    np.random.seed(42)
    customer_data = {
        'customer_id': [f'C{str(i).zfill(4)}' for i in range(1, n_customers + 1)],
        'total_spend': np.random.gamma(2, 500, n_customers),
        'purchase_count': np.random.poisson(3, n_customers),
        'click_count': np.random.poisson(20, n_customers),
        'days_since_last_purchase': np.random.exponential(30, n_customers),
        'avg_order_value': [],
    }
    
    for i in range(n_customers):
        if customer_data['purchase_count'][i] > 0:
            customer_data['avg_order_value'].append(
                customer_data['total_spend'][i] / customer_data['purchase_count'][i]
            )
        else:
            customer_data['avg_order_value'].append(0)
    
    customers_df = pd.DataFrame(customer_data)
    customers_df['sentiment_score'] = np.random.uniform(-1, 1, n_customers)
    return customers_df

@st.cache_data
def perform_segmentation(customers_df, n_clusters=5):
    """Perform K-means clustering on customer behavioral features"""
    features = customers_df[[
        'total_spend', 'purchase_count', 'click_count',
        'days_since_last_purchase', 'avg_order_value'
    ]].fillna(0)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    customers_df['segment'] = kmeans.fit_predict(features_scaled)
    return customers_df

@st.cache_data
def assign_segment_labels(customers_df):
    """Assign business-meaningful labels and calculate conversion rates"""
    segment_stats = customers_df.groupby('segment').agg({
        'customer_id': 'count',
        'total_spend': 'mean',
        'purchase_count': 'mean',
        'click_count': 'mean',
        'days_since_last_purchase': 'mean',
        'avg_order_value': 'mean'
    }).round(2)
    
    segment_stats.columns = ['customer_count', 'avg_spend', 'avg_purchases', 
                             'avg_clicks', 'avg_recency', 'avg_order_value']
    
    def get_label(segment_num, stats):
        row = stats.loc[segment_num]
        if row['avg_spend'] > 2000 and row['avg_purchases'] > 4:
            return "VIP High-Value"
        elif row['avg_spend'] > 2000 and row['avg_purchases'] < 3:
            return "Premium Occasional"
        elif row['avg_purchases'] > 3 and row['avg_clicks'] > 20:
            return "Engaged Loyalists"
        elif row['avg_recency'] > 80:
            return "At-Risk Dormant"
        else:
            return "Casual Browsers"
    
    segment_labels = {seg: get_label(seg, segment_stats) for seg in range(5)}
    customers_df['segment_label'] = customers_df['segment'].map(segment_labels)
    
    def estimate_conversion(segment_num, stats):
        row = stats.loc[segment_num]
        if row['avg_clicks'] > 0:
            base_rate = (row['avg_purchases'] / row['avg_clicks']) * 100
            recency_factor = 1 / (1 + (row['avg_recency'] / 30))
            return round(base_rate * recency_factor, 2)
        return 0
    
    segment_stats['conversion_rate'] = [estimate_conversion(seg, segment_stats) for seg in range(5)]
    
    return customers_df, segment_stats, segment_labels

# ==================== SESSION STATE ====================

# Competitive Intelligence state
if "comp_report" not in st.session_state:
    st.session_state.comp_report = None
if "comp_company" not in st.session_state:
    st.session_state.comp_company = None
if "comp_data" not in st.session_state:
    st.session_state.comp_data = None

# Customer Intelligence state
if 'customers_df' not in st.session_state:
    st.session_state.customers_df = None

# ==================== MAIN UI ====================

st.title("Marketing Intelligence Platform")
st.markdown("AI-powered competitive analysis and customer segmentation")
st.markdown("---")

# Create tabs
tab1, tab2 = st.tabs(["Competitive Intelligence", "Customer Intelligence"])

# ==================== TAB 1: COMPETITIVE INTELLIGENCE ====================

with tab1:
    # Sidebar controls
    with st.sidebar:
        st.header("Competitive Analysis")
        company_name = st.text_input("Company Name", placeholder="e.g., Perplexity, Anthropic, OpenAI")
        st.markdown("---")
        st.subheader("Data Sources")
        use_news = st.checkbox("News API", value=True, key="comp_news")
        use_wikipedia = st.checkbox("Wikipedia Pageviews", value=True, key="comp_wiki")
        st.markdown("---")
        st.subheader("Analysis Options")
        num_articles = st.slider("Articles to analyze", 20, 100, 40, 10)
        st.markdown("---")
        generate_btn = st.button("Generate Report", type="primary", key="comp_generate")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Competitive Intelligence Report")
        
        if generate_btn:
            if company_name:
                if not (use_news or use_wikipedia):
                    st.error("Enable at least one data source")
                else:
                    with st.spinner(f"Analyzing {company_name}..."):
                        prog = st.progress(0)
                        status = st.empty()
                        
                        status.text("Step 1/4: Collecting data...")
                        prog.progress(25)
                        
                        status.text("Step 2/4: Processing data...")
                        prog.progress(50)
                        
                        status.text("Step 3/4: Generating insights...")
                        prog.progress(75)
                        
                        report, data = generate_competitive_brief(company_name, use_news, use_wikipedia, num_articles)
                        
                        prog.progress(100)
                        st.session_state.comp_report = report
                        st.session_state.comp_company = company_name
                        st.session_state.comp_data = data
                        
                        prog.empty()
                        status.empty()
                    
                    st.success("Analysis complete")
            else:
                st.error("Enter a company name")
        
        if st.session_state.comp_report and st.session_state.comp_data:
            st.markdown("---")
            
            # Metrics
            data = st.session_state.comp_data
            news_ct = len([d for d in data if d.get("source") == "News"])
            scores = [d.get("relevance_score", 0) for d in data if "relevance_score" in d]
            avg_rel = int(sum(scores) / len(scores)) if scores else 0
            wiki = [d for d in data if d.get("source") == "Wikipedia"]
            pageviews = wiki[0].get("avg_daily_pageviews") if wiki and wiki[0].get("avg_daily_pageviews") else None
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Articles Analyzed", news_ct)
            with col_b:
                delta_text = "High Quality" if avg_rel >= 70 else ("Good Quality" if avg_rel >= 50 else "Low Quality")
                st.metric("Avg Relevance Score", f"{avg_rel}/100", delta=delta_text)
            with col_c:
                if pageviews:
                    st.metric("Avg Daily Wikipedia Views", f"{pageviews:,}")
                else:
                    st.metric("Wikipedia Views", "N/A")
            
            st.markdown("---")
            st.subheader("Visual Insights")
            
            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                fig = create_competitive_sentiment_viz(st.session_state.comp_data)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with viz_col2:
                fig = create_competitive_timeline_viz(st.session_state.comp_data)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.download_button("Download Report", st.session_state.comp_report,
                              f"{st.session_state.comp_company}_{datetime.now().strftime('%Y%m%d')}.md",
                              mime="text/markdown")
            st.markdown(st.session_state.comp_report)
        
        else:
            st.info("Enter a company name in the sidebar and click Generate Report")

# ==================== TAB 2: CUSTOMER INTELLIGENCE ====================

with tab2:
    # Sidebar controls
    with st.sidebar:
        st.header("Customer Segmentation")
        st.markdown("**Dataset:** 1,000 synthetic customers")
        n_clusters = st.slider("Number of Segments", 3, 7, 5, key="cust_segments")
        st.markdown("---")
        if st.button("Generate New Segmentation", type="primary", key="gen_segments"):
            st.session_state.customers_df = None
        st.markdown("---")
        st.caption("Based on Segmint methodology (20M+ events processed)")
    
    st.header("Customer Intelligence")
    
    # Methodology explanation
    with st.expander("How Segmentation Works", expanded=False):
        st.markdown("""
        ### Segmentation Methodology
        
        **Based on Production Segmint System:**
        - Original system processed 20+ million customer events
        - Achieved conversion rates up to 28.95%
        - This demo uses the same core methodology on synthetic data
        
        **Features Used for Clustering:**
        1. **Total Spend** - Lifetime customer value
        2. **Purchase Count** - Transaction frequency
        3. **Click Count** - Engagement level
        4. **Days Since Last Purchase** - Recency (churn risk indicator)
        5. **Average Order Value** - Purchase size patterns
        
        **Clustering Process:**
        1. Features are standardized (zero mean, unit variance)
        2. K-means algorithm groups similar customers into segments
        3. Segments are labeled based on behavioral characteristics:
           - **VIP High-Value:** High spend + frequent purchases
           - **Premium Occasional:** High spend + infrequent purchases
           - **Engaged Loyalists:** Moderate spend + high engagement
           - **At-Risk Dormant:** High recency (inactive customers)
           - **Casual Browsers:** Moderate behavior across all metrics
        
        **Conversion Rate Calculation:**
```
        Base Rate = (Avg Purchases / Avg Clicks) × 100
        Recency Penalty = 1 / (1 + Days Since Purchase / 30)
        Final Rate = Base Rate × Recency Penalty
```
        
        **Practical Use:**
        - Target high-value segments with premium offers
        - Re-engage at-risk customers with win-back campaigns
        - Nurture engaged loyalists for lifetime value
        """)
    
    # Generate or load data
    if st.session_state.customers_df is None:
        with st.spinner("Generating customer segments..."):
            customers_df = generate_customer_data(1000)  # Fixed at 1000
            customers_df = perform_segmentation(customers_df, n_clusters)
            customers_df, segment_stats, segment_labels = assign_segment_labels(customers_df)
            st.session_state.customers_df = customers_df
            st.session_state.segment_stats = segment_stats
            st.session_state.segment_labels = segment_labels
    
    customers_df = st.session_state.customers_df
    segment_stats = st.session_state.segment_stats
    segment_labels = st.session_state.segment_labels
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(customers_df):,}")
    with col2:
        st.metric("Avg Customer Value", f"${customers_df['total_spend'].mean():.2f}")
    with col3:
        st.metric("Active Segments", n_clusters)
    with col4:
        st.metric("Avg Conversion", f"{segment_stats['conversion_rate'].mean():.1f}%")
    
    st.markdown("---")
    
    # Sub-tabs for different views
    subtab1, subtab2, subtab3, subtab4 = st.tabs(["Overview", "Segments", "Analysis", "Export"])
    
    with subtab1:
        st.subheader("Segment Distribution")
        
        # Pie chart
        segment_counts = customers_df['segment_label'].value_counts()
        fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                     title="Customer Distribution by Segment")
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot
        st.subheader("Purchase Behavior by Segment")
        fig = px.scatter(customers_df, x='purchase_count', y='total_spend',
                        color='segment_label', size='click_count',
                        hover_data=['customer_id', 'days_since_last_purchase'],
                        title="Customer Segments: Purchases vs Spend")
        st.plotly_chart(fig, use_container_width=True)
    
    with subtab2:
        st.subheader("Segment Performance")
        
        display_stats = segment_stats.copy()
        display_stats.index = [segment_labels[i] for i in display_stats.index]
        
        st.dataframe(
            display_stats.style.format({
                'avg_spend': '${:.2f}',
                'avg_purchases': '{:.2f}',
                'avg_clicks': '{:.2f}',
                'avg_recency': '{:.1f} days',
                'avg_order_value': '${:.2f}',
                'conversion_rate': '{:.2f}%'
            }),
            use_container_width=True
        )
        
        # Conversion rate bar chart
        st.subheader("Conversion Rates by Segment")
        fig = px.bar(
            x=[segment_labels[i] for i in segment_stats.index],
            y=segment_stats['conversion_rate'],
            labels={'x': 'Segment', 'y': 'Conversion Rate (%)'},
            title="Segment Conversion Rates"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with subtab3:
        st.subheader("Behavioral Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(customers_df, x='segment_label', y='total_spend',
                        title="Spend Distribution by Segment")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(customers_df, x='segment_label', y='days_since_last_purchase',
                        title="Recency by Segment")
            st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap
        st.subheader("Segment Characteristics Heatmap")
        heatmap_data = segment_stats[['avg_spend', 'avg_purchases', 'avg_clicks', 
                                       'avg_recency', 'conversion_rate']].T
        heatmap_data.columns = [segment_labels[i] for i in heatmap_data.columns]
        heatmap_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
        
        fig = px.imshow(heatmap_normalized,
                       labels=dict(x="Segment", y="Metric", color="Normalized Value"),
                       x=heatmap_normalized.columns, y=heatmap_normalized.index,
                       title="Normalized Segment Characteristics")
        st.plotly_chart(fig, use_container_width=True)
    
    with subtab4:
        st.subheader("Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = customers_df.to_csv(index=False)
            st.download_button(
                label="Download Customer Data",
                data=csv,
                file_name="customer_segments.csv",
                mime="text/csv",
                key="download_customers"
            )
        
        with col2:
            summary_csv = segment_stats.to_csv()
            st.download_button(
                label="Download Segment Summary",
                data=summary_csv,
                file_name="segment_summary.csv",
                mime="text/csv",
                key="download_summary"
            )
        
        st.info("Use these exports for further analysis or integration with your CRM system")

# Footer
st.markdown("---")
st.markdown("**Marketing Intelligence Platform** | Modules: Competitive Intelligence + Customer Intelligence")
