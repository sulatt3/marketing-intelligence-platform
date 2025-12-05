"""
Marketing Intelligence Platform - Main UI
"""

import streamlit as st
import os
from datetime import datetime
from pathlib import Path

# Import our competitive intelligence modules
from modules.competitive.data_collector import CompetitiveDataCollector
from modules.competitive.analyzer import CompetitiveAnalyzer

# Page config
st.set_page_config(
    page_title="Marketing Intelligence Platform",
    page_icon="chart",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_report' not in st.session_state:
    st.session_state.current_report = None
if 'current_company' not in st.session_state:
    st.session_state.current_company = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# Title
st.title("Marketing Intelligence Platform")
st.markdown("**Status:** Week 1 - Competitive Intelligence Module")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Competitive Analysis")
    
    company_name = st.text_input(
        "Company Name",
        placeholder="e.g., Perplexity, Anthropic, OpenAI",
        help="Enter the company you want to analyze"
    )
    
    st.markdown("---")
    
    st.subheader("Data Sources")
    use_news = st.checkbox("News API", value=True, help="Recent articles from last 30 days")
    use_trends = st.checkbox("Google Trends", value=True, help="Search interest data")
    
    st.markdown("---")
    
    st.subheader("Analysis Options")
    num_articles = st.slider(
        "Articles to analyze",
        min_value=20,
        max_value=50,
        value=40,
        step=5,
        help="More articles = more comprehensive analysis"
    )
    
    st.markdown("---")
    
    generate_btn = st.button("Generate Report", type="primary")
    
    st.markdown("---")
    st.caption("Powered by Gemini 2.5 Pro")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Competitive Intelligence Report")
    
    if generate_btn:
        if company_name:
            if not (use_news or use_trends):
                st.error("Please enable at least one data source")
            else:
                try:
                    with st.spinner(f"Analyzing {company_name}... This takes 45-60 seconds"):
                        # Initialize modules
                        collector = CompetitiveDataCollector()
                        analyzer = CompetitiveAnalyzer()
                        
                        # Progress indicators
                        progress = st.progress(0)
                        status = st.empty()
                        
                        # Step 1: Collect data
                        status.text("Step 1/4: Collecting data from sources...")
                        progress.progress(25)
                        raw_data = collector.collect_all(company_name, use_news, use_trends)
                        
                        # Step 2: Process data
                        status.text("Step 2/4: Processing and ranking data...")
                        progress.progress(50)
                        processed_data = analyzer.process_data(raw_data, company_name, top_n=num_articles)
                        
                        # Step 3: Generate report
                        status.text("Step 3/4: Generating AI insights...")
                        progress.progress(75)
                        report = analyzer.generate_insights(processed_data, company_name)
                        
                        # Step 4: Save
                        status.text("Step 4/4: Saving report...")
                        progress.progress(100)
                        
                        # Save to file
                        reports_dir = Path("reports")
                        reports_dir.mkdir(exist_ok=True)
                        filename = f"{company_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md"
                        filepath = reports_dir / filename
                        
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(report)
                        
                        # Store in session
                        st.session_state.current_report = report
                        st.session_state.current_company = company_name
                        st.session_state.processed_data = processed_data
                        
                        progress.empty()
                        status.empty()
                    
                    st.success(f"Report generated for {company_name}!")
                    
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
                    st.exception(e)
        else:
            st.error("Please enter a company name")
    
    # Display report
    if st.session_state.current_report:
        st.markdown("---")
        
        # Metrics
        if st.session_state.processed_data:
            data = st.session_state.processed_data
            news_count = len([d for d in data if d.get('source') == 'News'])
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Articles Analyzed", news_count)
            with col_b:
                scores = [d.get('relevance_score', 0) for d in data if 'relevance_score' in d]
                avg_score = int(sum(scores) / len(scores)) if scores else 0
                st.metric("Avg Relevance Score", f"{avg_score}/100")
            with col_c:
                trends = [d for d in data if d.get('source') == 'Google Trends']
                if trends and trends[0].get('current_interest_level'):
                    st.metric("Search Interest", f"{trends[0]['current_interest_level']}/100")
                else:
                    st.metric("Search Interest", "N/A")
        
        st.markdown("---")
        
        # Download button
        st.download_button(
            label="Download Report",
            data=st.session_state.current_report,
            file_name=f"{st.session_state.current_company}_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )
        
        # Display
        st.markdown(st.session_state.current_report)
    
    else:
        st.info("Enter a company name in the sidebar and click Generate Report")
        
        st.markdown("### How it works:")
        st.markdown("""
        1. **Data Collection**: Fetches 100+ news articles and Google Trends data
        2. **Smart Filtering**: Deduplicates and ranks by relevance (0-100 score)
        3. **AI Synthesis**: Gemini 2.5 Pro generates strategic insights
        4. **Results**: Professional competitive intelligence report in 60 seconds
        """)
        
        st.markdown("### Report Sections:")
        st.markdown("""
        - Executive Summary
        - Recent Key Moves (with dates)
        - Product Launches & Features
        - Funding, Partnerships & Hiring
        - Google Trends Analysis
        - Competitive Threats & Opportunities
        - Strategic Recommendations (3-5 actionable items)
        - Timeline of Key Events
        """)

with col2:
    st.header("Past Reports")
    
    reports_dir = Path("reports")
    if reports_dir.exists():
        report_files = list(reports_dir.glob("*.md"))
        
        if report_files:
            st.success(f"Found {len(report_files)} saved reports")
            
            selected_report = st.selectbox(
                "Select a report to view:",
                report_files,
                format_func=lambda x: x.stem.replace('_', ' ')
            )
            
            if selected_report:
                with open(selected_report, 'r') as f:
                    past_report = f.read()
                
                with st.expander("View Report", expanded=False):
                    st.markdown(past_report)
                
                st.download_button(
                    label="Download This Report",
                    data=past_report,
                    file_name=selected_report.name,
                    mime="text/markdown",
                    key=f"download_{selected_report.name}"
                )
        else:
            st.info("No reports yet. Generate your first one!")
    else:
        st.info("No reports yet. Generate your first one!")

# Footer
st.markdown("---")
st.markdown("**Marketing Intelligence Platform** | Week 1: Competitive Intelligence")
