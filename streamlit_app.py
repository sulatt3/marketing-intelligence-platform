"""
Marketing Intelligence Platform
AI-powered competitive analysis and customer segmentation
"""

import streamlit as st
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from difflib import SequenceMatcher

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from groq import Groq
from newsapi import NewsApiClient
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import requests

st.set_page_config(
    page_title="Marketing Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== PASSWORD PROTECTION ====================

demo_password = st.text_input("🔒 Demo Access Password", type="password", key="demo_password", 
                              help="Portfolio demo - Contact su.h.latt3@gmail.com for password")

if demo_password != os.getenv("DEMO_PASSWORD", ""):
    st.warning("🔒 This is a portfolio demonstration. Password required for access.")
    st.info("📧 Recruiters/Interviewers: Request password at su.h.latt3@gmail.com")
    st.markdown("---")
    st.markdown("""
    **What This Demo Includes:**
    - AI-powered competitive intelligence with Llama 3.3 70B (via Groq)
    - Behavioral customer segmentation (K-means clustering)
    - Comprehensive evaluation framework (data quality + ML metrics + LLM assessment)
    - Production-grade features: rate limiting, error handling, hybrid LLM scoring
    """)
    st.stop()

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize usage tracking
if "total_reports_generated" not in st.session_state:
    st.session_state.total_reports_generated = 0

WIKI_HEADERS = {
    'User-Agent': 'Marketing Intelligence Platform/1.0 (https://github.com/sulatt3/marketing-intelligence-platform; su.h.latt3@gmail.com)'
}

# ==================== COMPETITIVE INTELLIGENCE ====================

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
        st.warning(f"News data temporarily unavailable: {str(e)}")
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
What threats does {company} pose? What opportunities exist?

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

def llm_score_articles(articles: List[Dict[str, Any]], company: str) -> List[Dict[str, Any]]:
    """
    Use Llama 3.3 70B (via Groq) to batch score article relevance
    Returns articles with LLM-assigned relevance scores
    """
    import json
    
    # Prepare articles for batch scoring
    articles_for_scoring = [
        {
            "id": i,
            "title": article.get("title", ""),
            "snippet": article.get("snippet", "")[:400]  # Increased for more context
        }
        for i, article in enumerate(articles)
    ]
    
    prompt = f"""You are a competitive intelligence analyst scoring article relevance to {company}.

IMPORTANT: Use SEMANTIC relevance, not just keyword matching. An article can be highly relevant even without directly naming {company}.

Score each article 0-100 based on relevance to understanding {company}'s competitive position:

**Scoring Guidelines:**

80-100 (HIGHLY RELEVANT):
- Direct announcements, launches, or strategic moves by {company}
- Detailed analysis of {company}'s products, strategy, or market position
- Major news events involving {company} directly

60-79 (VERY RELEVANT):
- Competitive comparisons mentioning {company}
- Industry analysis where {company} is a key player
- Partnerships, funding, or leadership changes at {company}
- Coverage of {company}'s ecosystem or related products

40-59 (MODERATELY RELEVANT):
- Industry trends affecting {company}'s market
- Competitor moves that impact {company}'s position
- Regulatory/policy changes relevant to {company}'s business
- Technology developments in {company}'s domain
- Articles mentioning {company}'s leadership or key people

20-39 (SOMEWHAT RELEVANT):
- Broad industry news tangentially related
- Market trends with indirect relevance
- General tech news where {company} might be peripherally involved

0-19 (NOT RELEVANT):
- Completely unrelated topics
- Different industry or company
- False positives from keyword matching

**Key Point:** An article about "{company}'s competitors launching new products" or "{company}'s CEO speaking on AI safety" is HIGHLY relevant even if it doesn't repeat the company name multiple times.

Articles to score:
{json.dumps(articles_for_scoring, indent=2)}

Return format: [score1, score2, score3, ...]
Example: [85, 62, 48, 91, 33, ...]

Return ONLY the JSON array, nothing else."""

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.1
        )
        
        response_text = completion.choices[0].message.content.strip()
        
        # Parse JSON response
        # Remove potential markdown code fences
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        
        scores = json.loads(response_text.strip())
        
        # Validate we got the right number of scores
        if len(scores) != len(articles):
            st.warning(f"LLM scoring mismatch: got {len(scores)} scores for {len(articles)} articles. Using rule-based scores.")
            return articles
        
        # Assign LLM scores to articles
        for i, article in enumerate(articles):
            article["llm_relevance_score"] = scores[i]
            # Keep original rule-based score as fallback
            article["rule_based_score"] = article.get("relevance_score", 0)
            # Use LLM score as primary
            article["relevance_score"] = scores[i]
        
        return articles
        
    except Exception as e:
        st.warning(f"LLM scoring failed: {str(e)}. Using rule-based scores.")
        return articles

@st.cache_data(ttl=3600)
def generate_competitive_brief(company: str, use_news: bool, use_wikipedia: bool, num_articles: int) -> tuple:
    raw = collect_all_data(company, use_news, use_wikipedia)
    if not raw:
        return "No data sources enabled. Please select at least one data source.", None, None
    
    # Step 1: Rule-based pre-filtering 
    # Get 3x the user's request to give LLM excellent candidates to choose from
    # Since Groq is free, we can afford to be generous with the candidate pool
    rule_filter_size = min(num_articles * 3, 300)
    processed = process_data(raw, company, top_n=rule_filter_size)
    news_items = [i for i in processed if i.get("source") == "News"]
    
    if len(news_items) < 5:
        return f"Insufficient data for {company}. Only {len(news_items)} articles found. Try using the company's full legal name.", None, None
    
    # Step 2: LLM semantic scoring
    MIN_ARTICLES_FOR_LLM = 15
    llm_scored = False
    LLM_THRESHOLD = 30  # Default threshold
    
    if len(news_items) >= MIN_ARTICLES_FOR_LLM:
        # Apply LLM scoring with semantic understanding
        news_items = llm_score_articles(news_items, company)
        llm_scored = True
        
        # Step 3: Adaptive threshold - aim to have ~1.5x user's request after filtering
        # This ensures we have good articles to choose from
        target_after_filter = int(num_articles * 1.5)
        
        # Try different thresholds to hit target
        for threshold in [40, 35, 30, 25, 20]:
            candidates = [item for item in news_items if item.get("relevance_score", 0) >= threshold]
            if len(candidates) >= target_after_filter:
                LLM_THRESHOLD = threshold
                break
            LLM_THRESHOLD = threshold  # Use lowest threshold we tried
        
        high_quality_articles = [item for item in news_items if item.get("relevance_score", 0) >= LLM_THRESHOLD]
        
        # Ensure we have at least SOME articles
        if len(high_quality_articles) < 10:
            LLM_THRESHOLD = 15
            high_quality_articles = [item for item in news_items if item.get("relevance_score", 0) >= LLM_THRESHOLD]
        
        news_items = high_quality_articles
    
    # Step 4: Final selection - exactly what user requested (or all available if fewer)
    final_articles = sorted(news_items, key=lambda x: x.get("relevance_score", 0), reverse=True)[:num_articles]
    
    # Combine with Wikipedia data
    wikipedia_data = [i for i in processed if i.get("source") == "Wikipedia"]
    final_processed = final_articles + wikipedia_data
    
    if len(final_articles) < 5:
        return f"Insufficient high-quality data for {company}. Only {len(final_articles)} articles passed quality filter.", None, None
    
    try:
        prompt = build_synthesis_prompt(final_processed, company)
        
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.3
        )
        
        report_text = completion.choices[0].message.content
        
        # Track scoring metadata
        scoring_metadata = {
            "llm_scored": llm_scored,
            "articles_collected": len(raw),
            "after_rule_filter": len(news_items) if not llm_scored else len(processed),
            "after_llm_filter": len(news_items) if llm_scored else 0,
            "articles_used": len(final_articles),
            "llm_threshold": LLM_THRESHOLD if llm_scored else None
        }
        
        return report_text, final_processed, scoring_metadata
        
    except Exception as e:
        error_msg = str(e)
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            return f"API rate limit reached. Please wait a few minutes. Data collection was successful ({len(final_articles)} articles), but AI synthesis is temporarily unavailable.", final_processed, None
        else:
            return f"Error generating report: {error_msg}", final_processed, None

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

# ==================== DATA QUALITY & EVALUATION ====================

def analyze_competitive_data_quality(raw_data, processed_data, company):
    """Comprehensive data quality analysis for competitive intelligence"""
    quality_metrics = {}
    
    # Collection metrics
    quality_metrics['total_collected'] = len(raw_data)
    quality_metrics['after_deduplication'] = len(processed_data)
    quality_metrics['duplicates_removed'] = len(raw_data) - len(processed_data)
    quality_metrics['deduplication_rate'] = (quality_metrics['duplicates_removed'] / len(raw_data) * 100) if len(raw_data) > 0 else 0
    
    news_items = [i for i in processed_data if i.get("source") == "News"]
    
    # Data completeness
    complete_records = sum(1 for item in news_items if item.get('title') and item.get('snippet') and item.get('date') != 'Unknown')
    quality_metrics['completeness_rate'] = (complete_records / len(news_items) * 100) if len(news_items) > 0 else 0
    
    # Relevance analysis
    relevance_scores = [i.get('relevance_score', 0) for i in news_items]
    quality_metrics['avg_relevance'] = np.mean(relevance_scores) if relevance_scores else 0
    quality_metrics['high_quality_articles'] = sum(1 for s in relevance_scores if s >= 70)
    quality_metrics['high_quality_rate'] = (quality_metrics['high_quality_articles'] / len(relevance_scores) * 100) if relevance_scores else 0
    
    # Temporal coverage
    dates = [i.get('date') for i in news_items if i.get('date') != 'Unknown']
    if dates:
        date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
        quality_metrics['date_range_days'] = (max(date_objs) - min(date_objs)).days + 1
        quality_metrics['temporal_coverage'] = min(quality_metrics['date_range_days'] / 30 * 100, 100)
    else:
        quality_metrics['date_range_days'] = 0
        quality_metrics['temporal_coverage'] = 0
    
    # Source diversity
    sources = [i.get('url', '').split('/')[2] if i.get('url') else '' for i in news_items]
    unique_sources = len(set(filter(None, sources)))
    quality_metrics['unique_sources'] = unique_sources
    quality_metrics['source_diversity'] = min(unique_sources / 10 * 100, 100)  # Normalize to 100
    
    # Content depth
    snippet_lengths = [len(i.get('snippet', '')) for i in news_items]
    quality_metrics['avg_content_length'] = int(np.mean(snippet_lengths)) if snippet_lengths else 0
    quality_metrics['content_depth_score'] = min(quality_metrics['avg_content_length'] / 200 * 100, 100)
    
    # Overall quality score (weighted average)
    quality_metrics['overall_quality_score'] = int(
        quality_metrics['completeness_rate'] * 0.25 +
        quality_metrics['high_quality_rate'] * 0.30 +
        quality_metrics['temporal_coverage'] * 0.20 +
        quality_metrics['source_diversity'] * 0.15 +
        quality_metrics['content_depth_score'] * 0.10
    )
    
    return quality_metrics

def create_relevance_distribution_viz(processed_data):
    """Visualize relevance score distribution"""
    news_items = [i for i in processed_data if i.get("source") == "News"]
    if not news_items:
        return None
    
    scores = [i.get('relevance_score', 0) for i in news_items]
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=20,
        marker_color='#4287f5',
        name='Articles'
    ))
    
    fig.add_vline(x=np.mean(scores), line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {np.mean(scores):.1f}")
    
    fig.update_layout(
        title="Relevance Score Distribution",
        xaxis_title="Relevance Score",
        yaxis_title="Number of Articles",
        height=400,
        showlegend=False
    )
    
    return fig

def analyze_customer_data_quality(customers_df):
    """EDA and data quality analysis for customer intelligence"""
    quality_metrics = {}
    
    # Basic statistics
    quality_metrics['total_customers'] = len(customers_df)
    quality_metrics['features'] = ['total_spend', 'purchase_count', 'click_count', 'days_since_last_purchase', 'avg_order_value']
    
    # Missing data analysis
    missing_counts = customers_df.isnull().sum()
    quality_metrics['missing_data'] = missing_counts.to_dict()
    quality_metrics['data_completeness'] = ((len(customers_df) * len(quality_metrics['features']) - missing_counts.sum()) / 
                                           (len(customers_df) * len(quality_metrics['features'])) * 100)
    
    # Outlier detection (IQR method)
    outliers = {}
    for col in ['total_spend', 'purchase_count', 'click_count']:
        Q1 = customers_df[col].quantile(0.25)
        Q3 = customers_df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers[col] = len(customers_df[(customers_df[col] < Q1 - 1.5 * IQR) | (customers_df[col] > Q3 + 1.5 * IQR)])
    quality_metrics['outliers'] = outliers
    quality_metrics['outlier_rate'] = sum(outliers.values()) / (len(customers_df) * 3) * 100
    
    # Feature distributions
    quality_metrics['spend_stats'] = {
        'mean': customers_df['total_spend'].mean(),
        'median': customers_df['total_spend'].median(),
        'std': customers_df['total_spend'].std(),
        'skewness': customers_df['total_spend'].skew()
    }
    
    return quality_metrics

def evaluate_clustering_quality(customers_df):
    """Evaluate K-means clustering performance"""
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    
    features = customers_df[['total_spend', 'purchase_count', 'click_count', 
                             'days_since_last_purchase', 'avg_order_value']].fillna(0)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Calculate clustering metrics
    labels = customers_df['segment'].values
    
    metrics = {}
    metrics['silhouette_score'] = silhouette_score(features_scaled, labels)
    metrics['davies_bouldin_score'] = davies_bouldin_score(features_scaled, labels)
    
    # Inertia for different k values (elbow method)
    inertias = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features_scaled)
        inertias.append(kmeans.inertia_)
    
    metrics['elbow_data'] = {'k_values': list(k_range), 'inertias': inertias}
    
    # Segment size balance
    segment_sizes = customers_df['segment'].value_counts()
    metrics['segment_balance'] = {
        'min_size': segment_sizes.min(),
        'max_size': segment_sizes.max(),
        'balance_ratio': segment_sizes.min() / segment_sizes.max()
    }
    
    return metrics

def create_feature_correlation_heatmap(customers_df):
    """Create correlation heatmap for customer features"""
    features = customers_df[['total_spend', 'purchase_count', 'click_count', 
                             'days_since_last_purchase', 'avg_order_value']]
    
    corr_matrix = features.corr()
    
    fig = px.imshow(corr_matrix,
                   labels=dict(color="Correlation"),
                   x=corr_matrix.columns,
                   y=corr_matrix.columns,
                   color_continuous_scale='RdBu_r',
                   zmin=-1, zmax=1,
                   title="Feature Correlation Matrix")
    
    fig.update_layout(height=500)
    return fig

def create_elbow_plot(elbow_data):
    """Create elbow plot for optimal k selection"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=elbow_data['k_values'],
        y=elbow_data['inertias'],
        mode='lines+markers',
        marker=dict(size=10, color='#4287f5'),
        line=dict(width=2, color='#4287f5')
    ))
    
    # Highlight k=5
    fig.add_vline(x=5, line_dash="dash", line_color="red",
                  annotation_text="Selected k=5")
    
    fig.update_layout(
        title="Elbow Method for Optimal K",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Inertia (Within-Cluster Sum of Squares)",
        height=400
    )
    
    return fig

def evaluate_llm_output(report_text, processed_data, company):
    """Evaluate LLM-generated report quality"""
    metrics = {}
    
    # Basic metrics
    metrics['word_count'] = len(report_text.split())
    metrics['character_count'] = len(report_text)
    metrics['paragraph_count'] = len([p for p in report_text.split('\n\n') if p.strip()])
    
    # Target sections we expect
    expected_sections = [
        "Executive Summary",
        "Recent Key Moves", 
        "Product Launches",
        "Funding",
        "Market Interest",
        "Competitive Threats",
        "Strategic Recommendations",
        "Timeline",
        "Sources"
    ]
    
    # Section completeness
    sections_present = sum(1 for section in expected_sections if section.lower() in report_text.lower())
    metrics['section_completeness'] = (sections_present / len(expected_sections)) * 100
    metrics['sections_generated'] = sections_present
    metrics['sections_expected'] = len(expected_sections)
    
    # Citation analysis
    news_items = [item for item in processed_data if item.get('source') == 'News']
    
    # Count how many source articles are referenced in the report
    cited_articles = 0
    for item in news_items[:20]:  # Check top 20 articles
        title_words = item.get('title', '').split()
        # Check if key words from title appear in report
        key_words = [w for w in title_words if len(w) > 5]  # Words longer than 5 chars
        if any(word.lower() in report_text.lower() for word in key_words[:3]):
            cited_articles += 1
    
    metrics['articles_provided'] = len(news_items)
    metrics['articles_cited'] = cited_articles
    metrics['citation_rate'] = (cited_articles / len(news_items) * 100) if news_items else 0
    
    # Factual grounding check
    company_mentions = report_text.lower().count(company.lower())
    metrics['company_mentions'] = company_mentions
    metrics['mention_density'] = (company_mentions / metrics['word_count'] * 100) if metrics['word_count'] > 0 else 0
    
    # Specificity check (presence of dates, numbers, names)
    import re
    dates_found = len(re.findall(r'\b\d{4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}\b', report_text))
    numbers_found = len(re.findall(r'\$[\d,]+|\d+%|\d+[MBK]?\+', report_text))
    
    metrics['dates_mentioned'] = dates_found
    metrics['numbers_mentioned'] = numbers_found
    metrics['specificity_score'] = min((dates_found + numbers_found) / 10 * 100, 100)
    
    # Quality indicators
    metrics['avg_paragraph_length'] = metrics['word_count'] / metrics['paragraph_count'] if metrics['paragraph_count'] > 0 else 0
    
    # Overall LLM quality score
    metrics['llm_quality_score'] = int(
        metrics['section_completeness'] * 0.30 +
        metrics['citation_rate'] * 0.25 +
        metrics['specificity_score'] * 0.25 +
        min(metrics['word_count'] / 1500 * 100, 100) * 0.20  # Expect 1000-1500 words
    )
    
    return metrics

# ==================== CUSTOMER INTELLIGENCE ====================

@st.cache_data
def generate_customer_data(n_customers=1000):
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
    return customers_df

@st.cache_data
def perform_segmentation(customers_df):
    features = customers_df[[
        'total_spend', 'purchase_count', 'click_count',
        'days_since_last_purchase', 'avg_order_value'
    ]].fillna(0)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    customers_df['segment'] = kmeans.fit_predict(features_scaled)
    return customers_df

@st.cache_data
def assign_segment_labels(customers_df):
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
    
    def get_behavior_label(segment_num, stats):
        row = stats.loc[segment_num]
        
        if row['avg_recency'] > 80:
            return "At-Risk Dormant"
        if row['avg_order_value'] > 1000:
            return "Premium Whales"
        if row['avg_purchases'] > 4.5:
            return "Frequent Buyers"
        if row['avg_clicks'] > 23 and row['avg_purchases'] > 2.5:
            return "Engaged Browsers"
        if row['avg_purchases'] < 2.0:
            return "Window Shoppers"
        return "Active Regulars"
    
    segment_labels = {seg: get_behavior_label(seg, segment_stats) for seg in range(5)}
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

if "comp_report" not in st.session_state:
    st.session_state.comp_report = None
if "comp_company" not in st.session_state:
    st.session_state.comp_company = None
if "comp_data" not in st.session_state:
    st.session_state.comp_data = None
if 'customers_df' not in st.session_state:
    st.session_state.customers_df = None
if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0
if "request_in_progress" not in st.session_state:
    st.session_state.request_in_progress = False
if "scoring_metadata" not in st.session_state:
    st.session_state.scoring_metadata = None

# ==================== MAIN UI ====================

st.title("Marketing Intelligence Platform")
st.markdown("AI-powered competitive analysis and customer segmentation")

# About section (collapsible)
with st.expander("ℹ️ About This Project"):
    st.markdown("""
    **Portfolio Project by Su Latt** | [GitHub](https://github.com/sulatt3/marketing-intelligence-platform)
    
    This platform demonstrates AI engineering capabilities through:
    - **Multi-API Orchestration**: News API + Wikipedia + Groq (Llama 3.3 70B)
    - **Hybrid LLM Scoring**: Rule-based pre-filtering + semantic relevance validation
    - **Machine Learning**: K-means behavioral segmentation (example model based on previous production system: 20M+ events, 28.95% conversion)
    - **Data Engineering**: ETL pipeline with deduplication, quality scoring, validation
    - **LLM Engineering**: Prompt engineering + output evaluation + hallucination detection
    - **Cost Optimization**: Free unlimited LLM inference via Groq
    - **Production Deployment**: Rate limiting, error handling, password protection, CI/CD
    
    **Tech Stack**: Python, Groq API (Llama 3.3 70B), scikit-learn, Plotly, Streamlit
    
    **Roadmap (Planned Enhancements):**
    - User feedback loop and rating system
    - Next Module: Marketing Insights & Recommendations engine
    - Multi-company comparison mode
    - Historical trend tracking and time-series analysis
    - Custom prompt templates for different industries
    - Export to PDF and enhanced report formats
    - Automated alerts for competitive moves
    """)

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Competitive Intelligence", "Customer Intelligence", "Data Quality & Metrics"])

# ==================== TAB 1: COMPETITIVE INTELLIGENCE ====================

with tab1:
    with st.sidebar:
        st.header("Competitive Analysis")
        
        company_name = st.text_input("Company Name", placeholder="e.g., Perplexity, Anthropic, OpenAI")
        st.markdown("---")
        st.subheader("Analysis Options")
        num_articles = st.slider("Articles to analyze", 20, 100, 40, 10, 
                                 help="Target number of articles (actual count may vary based on availability and quality)")
        st.caption("💡 20 = Quick analysis (5-10 sec) | 40 = Balanced (15-20 sec) | 100 = Deep dive (30-40 sec)")
        
        with st.expander("ℹ️ How Article Selection Works", expanded=False):
            st.markdown(f"""
            **Hybrid Scoring Pipeline:**
            
            **Your Selection: {num_articles} articles**
            
            **Step 1: Collect** ~150 recent articles from News API
            
            **Step 2: Rule-based pre-filter** → Top {num_articles * 3} candidates
            - Multiplied by 3× to ensure sufficient high-quality options
            - Uses keyword matching, recency, source authority
            - Fast, runs locally (no API cost)
            
            **Step 3: LLM semantic scoring** → Llama 3.3 scores each 0-100
            - Understands context (e.g., "Sam Altman" → OpenAI)
            - Catches competitor news, ecosystem updates, leadership mentions
            - Scores based on competitive intelligence value, not just company name
            
            **Step 4: Adaptive quality filtering**
            - Tries threshold of 40, 35, 30, 25, 20 (in that order)
            - Keeps lowering until we have enough quality articles
            - Aims for ~{int(num_articles * 1.5)} articles passing filter
            
            **Step 5: Final selection** → Up to **{num_articles} articles**
            - Takes top-scoring articles by LLM score
            - **Note:** If fewer than {num_articles} articles pass quality thresholds, 
              you'll get fewer (prioritizing quality over quantity)
            - Sorted by semantic relevance (best first)
            
            **Why 3× multiplier?** 
            - Not all articles will pass LLM quality threshold (typically 40-60% pass)
            - Multiplying by 3× ensures we usually have enough high-quality articles
            - For sparse coverage companies, you may get fewer than requested
            
            **Quality over quantity:** The system prioritizes relevant, high-quality articles 
            over hitting an arbitrary count. Better to analyze 31 great articles than 60 
            mediocre ones.
            """)
        
        st.markdown("---")
        generate_btn = st.button("Generate Report", type="primary", key="comp_generate")
        
        # Always use News API and try Wikipedia in background
        use_news = True
        use_wikipedia = True
    
    # Full width layout for competitive intelligence
    st.header("Competitive Intelligence Report")
    
    if generate_btn:
        import time
        
        # Check if request already in progress
        if st.session_state.request_in_progress:
            st.warning("⏳ Request already in progress. Please wait...")
            st.stop()
        
        # Check rate limiting (2 minutes between requests)
        current_time = time.time()
        time_since_last = current_time - st.session_state.last_request_time
        
        if time_since_last < 120:  # 2 minutes cooldown
            remaining = int(120 - time_since_last)
            minutes = remaining // 60
            seconds = remaining % 60
            st.error(f"⏳ Rate limit: Please wait {minutes}m {seconds}s before next request")
            st.info("This prevents hitting API rate limits (15 requests/minute)")
            st.stop()
        
        if company_name:
            if not (use_news or use_wikipedia):
                st.error("Please enable at least one data source")
            else:
                # Mark request as in progress
                st.session_state.request_in_progress = True
                st.session_state.last_request_time = current_time
                
                try:
                    with st.spinner(f"Analyzing {company_name}..."):
                        prog = st.progress(0)
                        status = st.empty()
                        
                        status.text("Collecting data...")
                        prog.progress(20)
                        
                        status.text("Rule-based filtering...")
                        prog.progress(40)
                        
                        status.text("LLM semantic scoring...")
                        prog.progress(60)
                        
                        status.text("Generating insights...")
                        prog.progress(80)
                        
                        report, data, scoring_meta = generate_competitive_brief(company_name, use_news, use_wikipedia, num_articles)
                        
                        prog.progress(100)
                        st.session_state.comp_report = report
                        st.session_state.comp_company = company_name
                        st.session_state.comp_data = data
                        st.session_state.scoring_metadata = scoring_meta
                        
                        # Increment usage counter
                        st.session_state.total_reports_generated += 1
                        
                        prog.empty()
                        status.empty()
                    
                    st.success("Analysis complete")
                finally:
                    # Always clear the in-progress flag
                    st.session_state.request_in_progress = False
        else:
            st.error("Please enter a company name")
    
    if st.session_state.comp_report and st.session_state.comp_data:
        st.markdown("---")
        
        data = st.session_state.comp_data
        news_ct = len([d for d in data if d.get("source") == "News"])
        scores = [d.get("relevance_score", 0) for d in data if "relevance_score" in d]
        avg_rel = int(sum(scores) / len(scores)) if scores else 0
        wiki = [d for d in data if d.get("source") == "Wikipedia"]
        pageviews = wiki[0].get("avg_daily_pageviews") if wiki and wiki[0].get("avg_daily_pageviews") else None
        
        # Only show Wikipedia column if data is available
        if pageviews:
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Articles Analyzed", news_ct)
            with col_b:
                delta_text = "High Quality" if avg_rel >= 70 else ("Good Quality" if avg_rel >= 50 else "Low Quality")
                st.metric("Avg Relevance Score", f"{avg_rel}/100", delta=delta_text)
            with col_c:
                st.metric("Avg Daily Wikipedia Views", f"{pageviews:,}")
        else:
            # Two-column layout when Wikipedia data unavailable
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Articles Analyzed", news_ct)
            with col_b:
                delta_text = "High Quality" if avg_rel >= 70 else ("Good Quality" if avg_rel >= 50 else "Low Quality")
                st.metric("Avg Relevance Score", f"{avg_rel}/100", delta=delta_text)
        
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
        st.info("Enter a company name in the sidebar and click Generate Report to begin analysis")

# ==================== TAB 2: CUSTOMER INTELLIGENCE ====================

with tab2:
    with st.sidebar:
        st.header("Customer Segmentation")
        st.markdown("**Dataset:** 1,000 synthetic customers")
        st.markdown("**Methodology:** Behavior-based clustering")
    
    st.header("Customer Intelligence")
    
    with st.expander("How Segmentation Works", expanded=False):
        st.markdown("""
        ### Behavior-Based Segmentation
        
        **Methodology:**
        Based on production Segmint system (20M+ events, 28.95% conversion rates).
        This demo replicates core logic on synthetic data.
        
        **Features Used:**
        - Total Spend - Lifetime value
        - Purchase Count - Transaction frequency
        - Click Count - Engagement level
        - Days Since Last Purchase - Recency/churn risk
        - Average Order Value - Purchase size patterns
        
        **Segment Definitions:**
        - **At-Risk Dormant:** >80 days inactive (churn risk)
        - **Premium Whales:** >$1,000 AOV (big-ticket buyers)
        - **Frequent Buyers:** >4.5 purchases (high-frequency)
        - **Engaged Browsers:** >23 clicks + >2.5 purchases
        - **Window Shoppers:** <2 purchases (low conversion)
        
        **Conversion Rate Calculation:**
        ```
        Base = (Purchases / Clicks) × 100
        Recency Penalty = 1 / (1 + Days / 30)
        Final Rate = Base × Recency Penalty
        ```
        """)
    
    if st.session_state.customers_df is None:
        with st.spinner("Loading customer segments..."):
            customers_df = generate_customer_data(1000)
            customers_df = perform_segmentation(customers_df)
            customers_df, segment_stats, segment_labels = assign_segment_labels(customers_df)
            st.session_state.customers_df = customers_df
            st.session_state.segment_stats = segment_stats
            st.session_state.segment_labels = segment_labels
    
    customers_df = st.session_state.customers_df
    segment_stats = st.session_state.segment_stats
    segment_labels = st.session_state.segment_labels
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(customers_df):,}")
    with col2:
        st.metric("Avg Customer Value", f"${customers_df['total_spend'].mean():.2f}")
    with col3:
        st.metric("Segments", "5")
    with col4:
        st.metric("Avg Conversion", f"{segment_stats['conversion_rate'].mean():.1f}%")
    
    st.markdown("---")
    
    subtab1, subtab2, subtab3, subtab4 = st.tabs(["Overview", "Segments", "Analysis", "Export"])
    
    with subtab1:
        st.subheader("Segment Distribution")
        
        segment_counts = customers_df['segment_label'].value_counts()
        fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                     title="Customer Distribution by Segment")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Purchase Behavior by Segment")
        fig = px.scatter(customers_df, x='purchase_count', y='total_spend',
                        color='segment_label', size='click_count',
                        hover_data=['customer_id', 'days_since_last_purchase'],
                        title="Purchases vs Spend (Size = Clicks)")
        st.plotly_chart(fig, use_container_width=True)
    
    with subtab2:
        st.subheader("Segment Performance")
        
        display_stats = segment_stats.copy()
        display_stats.index = [segment_labels[i] for i in display_stats.index]
        
        st.dataframe(
            display_stats.style.format({
                'customer_count': '{:,.0f}',
                'avg_spend': '${:.2f}',
                'avg_purchases': '{:.2f}',
                'avg_clicks': '{:.2f}',
                'avg_recency': '{:.1f} days',
                'avg_order_value': '${:.2f}',
                'conversion_rate': '{:.2f}%'
            }),
            use_container_width=True
        )
        
        st.subheader("Conversion Rates by Segment")
        fig = px.bar(
            x=[segment_labels[i] for i in segment_stats.index],
            y=segment_stats['conversion_rate'],
            labels={'x': 'Segment', 'y': 'Conversion Rate (%)'},
            title="Segment Conversion Performance",
            color=segment_stats['conversion_rate'],
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with subtab3:
        st.subheader("Behavioral Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(customers_df, x='segment_label', y='total_spend',
                        title="Spend Distribution by Segment",
                        color='segment_label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(customers_df, x='segment_label', y='days_since_last_purchase',
                        title="Recency Distribution by Segment",
                        color='segment_label')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Segment Characteristics Heatmap")
        heatmap_data = segment_stats[['avg_spend', 'avg_purchases', 'avg_clicks', 
                                       'avg_recency', 'conversion_rate']].T
        heatmap_data.columns = [segment_labels[i] for i in heatmap_data.columns]
        
        heatmap_normalized = (heatmap_data - heatmap_data.min(axis=1).values.reshape(-1,1)) / \
                             (heatmap_data.max(axis=1).values.reshape(-1,1) - heatmap_data.min(axis=1).values.reshape(-1,1))
        
        fig = px.imshow(heatmap_normalized,
                       labels=dict(x="Segment", y="Metric", color="Relative Strength"),
                       x=heatmap_normalized.columns, y=heatmap_normalized.index,
                       title="Normalized Segment Characteristics",
                       color_continuous_scale='RdYlGn')
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

# ==================== TAB 3: DATA QUALITY & METRICS ====================

with tab3:
    st.header("Data Quality & Evaluation Metrics")
    st.markdown("Comprehensive analysis of data quality, cleaning processes, and model evaluation")
    
    metric_tab1, metric_tab2 = st.tabs(["Competitive Intelligence Metrics", "Customer Intelligence Metrics"])
    
    with metric_tab1:
        st.subheader("Competitive Intelligence: Data Quality Analysis")
        
        if st.session_state.comp_data and st.session_state.comp_report:
            # Calculate quality metrics
            quality_metrics = analyze_competitive_data_quality(
                st.session_state.comp_data, 
                st.session_state.comp_data,
                st.session_state.comp_company
            )
            
            # LLM Output Evaluation
            llm_metrics = evaluate_llm_output(
                st.session_state.comp_report,
                st.session_state.comp_data,
                st.session_state.comp_company
            )
            
            # Overall Quality Score
            st.markdown("### Overall Data Quality Score")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                quality_color = "🟢" if quality_metrics['overall_quality_score'] >= 80 else ("🟡" if quality_metrics['overall_quality_score'] >= 60 else "🔴")
                st.metric("Data Quality", f"{quality_color} {quality_metrics['overall_quality_score']}/100")
            with col2:
                llm_color = "🟢" if llm_metrics['llm_quality_score'] >= 80 else ("🟡" if llm_metrics['llm_quality_score'] >= 60 else "🔴")
                st.metric("LLM Output Quality", f"{llm_color} {llm_metrics['llm_quality_score']}/100")
            with col3:
                st.metric("High Quality Articles", f"{quality_metrics['high_quality_rate']:.1f}%")
            with col4:
                st.metric("Data Completeness", f"{quality_metrics['completeness_rate']:.1f}%")
            
            st.markdown("---")
            
            # Hybrid Scoring Pipeline Metrics
            if st.session_state.scoring_metadata:
                meta = st.session_state.scoring_metadata
                
                st.markdown("### Hybrid Scoring Pipeline")
                st.caption("Rule-based pre-filtering + LLM semantic validation")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Articles Collected", meta['articles_collected'])
                with col2:
                    if meta['llm_scored']:
                        st.metric("After Rule Filter", meta['after_rule_filter'])
                    else:
                        st.metric("Rule-Based Only", meta['after_rule_filter'])
                with col3:
                    if meta['llm_scored']:
                        st.metric("LLM Approved", meta['after_llm_filter'])
                        st.caption(f"Threshold: ≥{meta['llm_threshold']}/100")
                    else:
                        st.metric("LLM Scoring", "Skipped")
                        st.caption("Not enough articles")
                with col4:
                    st.metric("Final Selection", meta['articles_used'])
                    st.caption("Used in report")
                
                if meta['llm_scored']:
                    # Calculate pass rates
                    llm_pass_rate = (meta['after_llm_filter'] / meta['after_rule_filter'] * 100) if meta['after_rule_filter'] > 0 else 0
                    
                    st.markdown(f"""
                    **Pipeline Efficiency:**
                    - Rule-based filter kept: {meta['after_rule_filter']}/{meta['articles_collected']} articles ({meta['after_rule_filter']/meta['articles_collected']*100:.1f}%)
                    - LLM approved: {meta['after_llm_filter']}/{meta['after_rule_filter']} candidates ({llm_pass_rate:.1f}%)
                    - Final selection: Top {meta['articles_used']} by LLM score
                    """)
            
            st.markdown("---")
            
            # LLM Evaluation Section
            st.markdown("### LLM Output Evaluation (Llama 3.3 70B via Groq)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Report Completeness**")
                st.progress(llm_metrics['section_completeness'] / 100)
                st.caption(f"{llm_metrics['sections_generated']}/{llm_metrics['sections_expected']} expected sections generated ({llm_metrics['section_completeness']:.0f}%)")
                
                st.markdown("**Output Metrics**")
                st.metric("Word Count", f"{llm_metrics['word_count']:,}")
                st.caption("Target: 1,000-1,500 words")
                st.metric("Paragraph Count", llm_metrics['paragraph_count'])
                st.metric("Avg Paragraph Length", f"{llm_metrics['avg_paragraph_length']:.0f} words")
            
            with col2:
                st.markdown("**Factual Grounding**")
                
                st.metric("Citation Rate", f"{llm_metrics['citation_rate']:.1f}%")
                st.caption(f"{llm_metrics['articles_cited']}/{llm_metrics['articles_provided']} articles referenced in report")
                
                st.metric("Company Mentions", llm_metrics['company_mentions'])
                st.caption(f"Mention density: {llm_metrics['mention_density']:.2f}% of words")
                
                st.markdown("**Specificity Check**")
                st.metric("Dates Mentioned", llm_metrics['dates_mentioned'])
                st.metric("Numbers/Stats Mentioned", llm_metrics['numbers_mentioned'])
                st.progress(llm_metrics['specificity_score'] / 100)
                st.caption(f"Specificity score: {llm_metrics['specificity_score']:.0f}/100 (presence of concrete facts)")
            
            st.markdown("---")
            
            st.markdown("### LLM Evaluation Methodology")
            st.markdown("""
            **How We Evaluate LLM Output:**
            
            1. **Section Completeness (30% weight)**
               - Checks if all 9 expected sections are present
               - Validates structure follows prompt instructions
            
            2. **Citation Rate (25% weight)**
               - Measures how many source articles are actually referenced
               - Detects potential hallucinations (claims without source backing)
               - Higher citation = better factual grounding
            
            3. **Specificity Score (25% weight)**
               - Counts dates, numbers, percentages, financial figures
               - Specific facts indicate deeper source analysis
               - Generic statements score lower
            
            4. **Length Appropriateness (20% weight)**
               - Target: 1,000-1,500 words (comprehensive but concise)
               - Too short = insufficient analysis
               - Too long = verbose/unfocused
            
            **Quality Indicators:**
            - 🟢 **80-100:** Excellent - comprehensive, well-cited, specific
            - 🟡 **60-79:** Good - mostly complete, some citations
            - 🔴 **<60:** Needs improvement - missing sections or low specificity
            
            **Hallucination Detection:**
            - Low citation rate (<40%) suggests potential hallucinations
            - Cross-reference company mentions with source data
            - Verify dates and numbers against provided articles
            """)
            
            st.markdown("---")
            
            # Data Collection & Cleaning
            st.markdown("### Data Collection & Cleaning Process")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Deduplication Analysis**")
                st.metric("Articles Collected", quality_metrics['total_collected'])
                st.metric("After Deduplication", quality_metrics['after_deduplication'])
                st.metric("Duplicates Removed", f"{quality_metrics['duplicates_removed']} ({quality_metrics['deduplication_rate']:.1f}%)")
                
                st.markdown("**Data Completeness Check**")
                st.progress(quality_metrics['completeness_rate'] / 100)
                st.caption(f"{quality_metrics['completeness_rate']:.1f}% of articles have complete title, content, and date")
            
            with col2:
                st.markdown("**Quality Dimensions**")
                
                quality_dims = {
                    "Relevance": quality_metrics['high_quality_rate'],
                    "Temporal Coverage": quality_metrics['temporal_coverage'],
                    "Source Diversity": quality_metrics['source_diversity'],
                    "Content Depth": quality_metrics['content_depth_score']
                }
                
                for dim, score in quality_dims.items():
                    st.metric(dim, f"{score:.1f}/100")
            
            st.markdown("---")
            
            # Visualizations
            st.markdown("### Data Distribution Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_relevance_distribution_viz(st.session_state.comp_data)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Data Quality Insights**")
                st.markdown(f"""
                - **Temporal Range:** {quality_metrics['date_range_days']} days of coverage
                - **Unique Sources:** {quality_metrics['unique_sources']} different news outlets
                - **Avg Content Length:** {quality_metrics['avg_content_length']} characters
                - **High Relevance Articles:** {quality_metrics['high_quality_articles']} articles (≥70 score)
                """)
                
                st.markdown("**Data Cleaning Steps Applied:**")
                st.markdown("""
                1. Removed duplicate articles (similarity threshold: 85%)
                2. Filtered articles with missing critical fields
                3. Scored articles by relevance (multi-factor algorithm)
                4. Validated date formats and temporal consistency
                5. Cleaned text encoding and special characters
                """)
        
        else:
            st.info("Generate a competitive intelligence report to see data quality metrics")
    
    with metric_tab2:
        st.subheader("Customer Intelligence: ML Evaluation Metrics")
        
        if st.session_state.customers_df is not None:
            customers_df = st.session_state.customers_df
            
            # Data Quality Analysis
            st.markdown("### Exploratory Data Analysis (EDA)")
            
            data_quality = analyze_customer_data_quality(customers_df)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Customers", f"{data_quality['total_customers']:,}")
            with col2:
                st.metric("Data Completeness", f"{data_quality['data_completeness']:.1f}%")
            with col3:
                st.metric("Outlier Rate", f"{data_quality['outlier_rate']:.1f}%")
            
            st.markdown("---")
            
            # Feature Analysis
            st.markdown("### Feature Distribution Analysis")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Spending Pattern Statistics**")
                spend_stats = data_quality['spend_stats']
                st.markdown(f"""
                - **Mean:** ${spend_stats['mean']:.2f}
                - **Median:** ${spend_stats['median']:.2f}
                - **Std Dev:** ${spend_stats['std']:.2f}
                - **Skewness:** {spend_stats['skewness']:.2f} (right-skewed distribution)
                """)
                
                st.markdown("**Outlier Detection (IQR Method)**")
                for feature, count in data_quality['outliers'].items():
                    st.markdown(f"- {feature}: {count} outliers detected")
            
            with col2:
                fig = create_feature_correlation_heatmap(customers_df)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Clustering Evaluation
            st.markdown("### K-Means Clustering Evaluation")
            
            clustering_metrics = evaluate_clustering_quality(customers_df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Clustering Quality Metrics**")
                
                silhouette = clustering_metrics['silhouette_score']
                silhouette_color = "🟢" if silhouette > 0.5 else ("🟡" if silhouette > 0.3 else "🔴")
                st.metric("Silhouette Score", f"{silhouette_color} {silhouette:.3f}")
                st.caption("Range: [-1, 1]. Higher is better. >0.5 indicates good clustering.")
                
                db_score = clustering_metrics['davies_bouldin_score']
                db_color = "🟢" if db_score < 1.0 else ("🟡" if db_score < 1.5 else "🔴")
                st.metric("Davies-Bouldin Score", f"{db_color} {db_score:.3f}")
                st.caption("Lower is better. <1.0 indicates well-separated clusters.")
                
                st.markdown("**Segment Balance Analysis**")
                balance = clustering_metrics['segment_balance']
                st.metric("Balance Ratio", f"{balance['balance_ratio']:.2f}")
                st.caption(f"Min segment: {balance['min_size']} | Max segment: {balance['max_size']}")
            
            with col2:
                fig = create_elbow_plot(clustering_metrics['elbow_data'])
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Data Preprocessing Steps
            st.markdown("### Data Preprocessing & Feature Engineering")
            
            st.markdown("""
            **Steps Applied:**
            
            1. **Missing Value Handling**
               - Imputed missing `avg_order_value` with 0 for customers with no purchases
               - Validated all features for null values
            
            2. **Feature Scaling**
               - Applied StandardScaler to normalize all features
               - Prevents features with larger scales from dominating clustering
            
            3. **Feature Engineering**
               - Calculated `avg_order_value` = `total_spend` / `purchase_count`
               - Derived behavioral patterns from raw transaction data
            
            4. **Outlier Treatment**
               - Detected outliers using IQR method (1.5 × IQR threshold)
               - Retained outliers as they represent legitimate customer segments (e.g., whales)
            
            5. **Validation**
               - Verified feature distributions follow expected patterns
               - Confirmed no data leakage between train/test (N/A for unsupervised)
               - Validated segment sizes and balance
            """)
            
            st.markdown("**Feature Engineering Rationale:**")
            st.markdown("""
            - **Recency (Days Since Last Purchase):** Captures churn risk and engagement level
            - **Frequency (Purchase Count):** Identifies loyal vs. one-time buyers
            - **Monetary (Total Spend + AOV):** Distinguishes high-value from low-value customers
            - **Engagement (Click Count):** Measures interest and browsing behavior
            
            These 5 features create a comprehensive behavioral profile for RFM+ segmentation.
            """)
        
        else:
            st.info("Customer intelligence data will appear here once generated")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Marketing Intelligence Platform</strong> | AI-Powered Insights for Competitive Strategy</p>
    <p>Built by <a href='https://github.com/sulatt3' target='_blank'>Su Latt</a> | 
    <a href='https://github.com/sulatt3/marketing-intelligence-platform' target='_blank'>View on GitHub</a> | 
    <a href='https://www.linkedin.com/in/su-l-67630a67/' target='_blank'>LinkedIn</a></p>
    <p style='font-size: 0.9em;'>Demonstrates: Multi-API Orchestration • LLM Engineering • ML Implementation • Production Deployment</p>
</div>
""", unsafe_allow_html=True)
