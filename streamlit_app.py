"""
Marketing Intelligence Platform - Main UI
With Wikipedia Pageviews (replacing unreliable Google Trends)
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
import google.generativeai as genai
from newsapi import NewsApiClient
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

# Caching for performance
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
    """Fetch Wikipedia pageviews as interest metric"""
    try:
        # Define User-Agent header
        headers = {
            'User-Agent': 'Marketing Intelligence Platform/1.0 (https://github.com/sulatt3/marketing-intelligence-platform; su.h.latt3@gmail.com)'
        }
        
        # Step 1: Search for article - ADD headers HERE
        search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={company}&limit=1&format=json"
        search_resp = requests.get(search_url, headers=headers, timeout=10)  # ← Added headers
        titles = search_resp.json()[1]
        
        if not titles:
            return [{"source": "Wikipedia", "total_pageviews": None}]
        
        title = titles[0].replace(" ", "_")
        
        # Step 2: Get pageviews - ADD headers HERE TOO
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        
        pageviews_url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{title}/daily/{start_date}/{end_date}"
        pageviews_resp = requests.get(pageviews_url, headers=headers, timeout=10)  # ← Added headers
        
        if pageviews_resp.status_code == 200:
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
        else:
            return [{"source": "Wikipedia", "total_pageviews": None}]
            
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
Analyze Wikipedia pageviews data. What does the trend direction indicate? High pageviews = high public interest.

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

def create_timeline_viz(processed_data):
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

def create_sentiment_viz(processed_data):
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

# Session state
if "current_report" not in st.session_state:
    st.session_state.current_report = None
if "current_company" not in st.session_state:
    st.session_state.current_company = None
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None

# UI
st.title("Marketing Intelligence Platform")
st.markdown("**Competitive Intelligence Module**")
st.markdown("---")

with st.sidebar:
    st.header("Competitive Analysis")
    company_name = st.text_input("Company Name", placeholder="e.g., Perplexity, Anthropic, OpenAI")
    st.markdown("---")
    st.subheader("Data Sources")
    use_news = st.checkbox("News API", value=True, help="Recent articles from last 30 days")
    use_wikipedia = st.checkbox("Wikipedia Pageviews", value=True, help="Public interest metric")
    st.markdown("---")
    st.subheader("Analysis Options")
    num_articles = st.slider("Articles to analyze", min_value=20, max_value=100, value=50, step=5)
    st.markdown("---")
    generate_btn = st.button("Generate Report", type="primary")
    st.markdown("---")
    st.caption("Powered by Gemini 2.5 Pro")

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
                    st.session_state.current_report = report
                    st.session_state.current_company = company_name
                    st.session_state.processed_data = data
                    
                    prog.empty()
                    status.empty()
                
                st.success(f"Analysis complete")
        else:
            st.error("Enter a company name")
    
    if st.session_state.current_report and st.session_state.processed_data:
        st.markdown("---")
        
        # Metrics
        data = st.session_state.processed_data
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
            fig = create_sentiment_viz(st.session_state.processed_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sentiment data")
        
        with viz_col2:
            fig = create_timeline_viz(st.session_state.processed_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No timeline data")
        
        st.markdown("---")
        st.download_button("Download Report", st.session_state.current_report,
                          f"{st.session_state.current_company}_{datetime.now().strftime('%Y%m%d')}.md",
                          mime="text/markdown")
        st.markdown(st.session_state.current_report)
    
    else:
        st.info("Enter a company name and click Generate Report")
        st.markdown("""
        ### How it works:
        1. **Data Collection**: 100+ news articles + Wikipedia pageviews
        2. **Smart Filtering**: Deduplication and relevance scoring (0-100)
        3. **AI Synthesis**: Gemini 2.5 Pro generates insights
        4. **Visualizations**: Interactive sentiment and timeline charts
        5. **Results**: Professional report in 60 seconds
        """)

with col2:
    st.header("Past Reports")
    reports_dir = Path("reports")
    if reports_dir.exists():
        report_files = list(reports_dir.glob("*.md"))
        if report_files:
            st.success(f"Found {len(report_files)} saved reports")
            selected = st.selectbox("Select report:", report_files, format_func=lambda x: x.stem.replace('_', ' '))
            if selected:
                with open(selected, 'r') as f:
                    past = f.read()
                with st.expander("View Report", expanded=False):
                    st.markdown(past)
                st.download_button("Download", past, selected.name, mime="text/markdown", key=f"dl_{selected.name}")
        else:
            st.info("No reports yet")
    else:
        st.info("No reports yet")

st.markdown("---")
st.markdown("**Marketing Intelligence Platform** | Competitive Intelligence")
