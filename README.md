# Marketing Intelligence Platform

AI-powered competitive analysis and customer segmentation for data-driven marketing strategy.

**Live Demo:** [marketing-intelligence-platform.streamlit.app](https://marketing-intelligence-platform.streamlit.app)

---

## Overview

The Marketing Intelligence Platform integrates real-time competitive intelligence with behavioral customer segmentation to provide actionable insights for marketing teams. Built with production-grade AI orchestration, the platform demonstrates enterprise-scale data processing patterns and strategic analytics capabilities.

### Key Features

**Competitive Intelligence Module**
- Multi-source data integration (News API, Wikipedia Pageviews)
- AI-powered synthesis using Gemini 2.5 Pro
- Smart deduplication and relevance scoring
- Sentiment analysis and trend detection
- Interactive visualizations with timeline and sentiment breakdowns

**Customer Intelligence Module**
- Behavior-based K-means clustering
- Five distinct customer segments with conversion rate analysis
- Based on production Segmint system (20M+ events, 28.95% conversion rates)
- Multi-dimensional behavioral analysis
- CSV export for further analysis

---

## Technical Architecture

### Data Pipeline
```
Data Sources → Collection → Processing → AI Synthesis → Visualization
     ↓             ↓            ↓             ↓              ↓
  News API    ThreadPool   Deduplication   Gemini      Plotly/Pandas
  Wikipedia   Concurrent   Relevance       2.5 Pro     Interactive
  Pageviews   Fetching     Scoring                     Charts
```

### Tech Stack

**Core Framework**
- Streamlit (UI/deployment)
- Python 3.9+

**AI & ML**
- Google Gemini 2.5 Pro (synthesis)
- scikit-learn (K-means clustering)
- Custom NLP (sentiment analysis)

**Data Processing**
- pandas (data manipulation)
- numpy (numerical operations)
- concurrent.futures (parallel API calls)

**Visualization**
- Plotly (interactive charts)
- Plotly Express (statistical plots)

**APIs**
- News API (news aggregation)
- Wikipedia Pageviews API (market interest)

---

## Installation

### Prerequisites
- Python 3.9+
- API Keys:
  - [News API](https://newsapi.org/) (free tier: 100 requests/day)
  - [Google AI Studio](https://aistudio.google.com/) (Gemini API - free tier available)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/sulatt3/marketing-intelligence-platform.git
cd marketing-intelligence-platform
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure API keys**

Create a `.streamlit/secrets.toml` file:
```toml
NEWSAPI_KEY = "your_newsapi_key_here"
GEMINI_API_KEY = "your_gemini_key_here"
```

Or set environment variables:
```bash
export NEWSAPI_KEY="your_newsapi_key_here"
export GEMINI_API_KEY="your_gemini_key_here"
```

4. **Run locally**
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

---

## Usage

### Competitive Intelligence

1. **Enter company name** (e.g., "Perplexity", "Anthropic", "OpenAI")
2. **Select data sources** (News API, Wikipedia Pageviews, or both)
3. **Adjust analysis depth** (20-100 articles)
4. **Generate report** - AI synthesizes insights in 15-30 seconds

**Output includes:**
- Executive summary
- Recent strategic moves
- Product launches
- Market interest trends
- Competitive threats and opportunities
- Strategic recommendations
- Downloadable markdown report

### Customer Intelligence

1. **Navigate to Customer Intelligence tab**
2. **Segments auto-generate** on first load (1,000 synthetic customers)
3. **Explore four sub-tabs:**
   - **Overview:** Distribution and purchase behavior
   - **Segments:** Performance metrics and conversion rates
   - **Analysis:** Box plots and characteristic heatmaps
   - **Export:** Download customer data and segment summaries

**Segment Types:**
- **At-Risk Dormant** (>80 days inactive)
- **Premium Whales** (>$1,000 AOV)
- **Frequent Buyers** (>4.5 purchases)
- **Engaged Browsers** (>23 clicks + >2.5 purchases)
- **Window Shoppers** (<2 purchases)

---

## Key Technical Implementations

### 1. Smart Deduplication
Uses SequenceMatcher for fuzzy string matching to eliminate near-duplicate articles (85% similarity threshold).

### 2. Multi-Dimensional Relevance Scoring
```python
Score Components:
- Title mentions: 40 points
- Content relevance: 20 points
- Mention frequency: up to 20 points
- Recency: 15 points (≤7 days) → 0 points (>30 days)
Max Score: 100
```

### 3. Behavior-Based Segmentation
Priority-based labeling system ensures segments reflect most critical behavioral patterns:
1. Churn risk (recency >80 days)
2. High value (AOV >$1,000)
3. High frequency (purchases >4.5)
4. High engagement (clicks >23, purchases >2.5)
5. Low conversion (purchases <2.0)

### 4. Conversion Rate Estimation
```python
Base Rate = (Avg Purchases / Avg Clicks) × 100
Recency Penalty = 1 / (1 + Days Since Purchase / 30)
Final Rate = Base Rate × Recency Penalty
```

---

## Project Structure

```
marketing-intelligence-platform/
├── streamlit_app.py          # Main application
├── requirements.txt           # Python dependencies
├── .streamlit/
│   └── secrets.toml          # API keys (not in git)
├── README.md                 # Documentation
└── .gitignore               # Git ignore rules
```

---

## Dependencies

```
streamlit>=1.28.0
google-generativeai>=0.3.0
newsapi-python>=0.2.7
plotly>=5.17.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
requests>=2.31.0
```

---

## Deployment

### Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Add secrets in Streamlit dashboard:
   - `NEWSAPI_KEY`
   - `GEMINI_API_KEY`
5. Deploy

### Alternative Platforms
- **Heroku:** Use `Procfile` with `web: streamlit run streamlit_app.py`
- **AWS EC2:** Run with systemd service
- **Docker:** Create `Dockerfile` with Streamlit base image

---

## Production Considerations

### Rate Limiting
- **News API:** 100 requests/day (free tier) - implement caching
- **Gemini API:** Subject to quota limits - handle ResourceExhausted errors
- **Wikipedia:** Generally unlimited, respectful delays recommended

### Scalability
- Implement Redis caching for multi-user deployments
- Use background workers for long-running analyses
- Consider batch processing for enterprise use cases

### Security
- Never commit API keys to version control
- Use Streamlit secrets management or environment variables
- Implement authentication for production deployments

---

## Roadmap

**Completed**
- ✅ Competitive Intelligence module with AI synthesis
- ✅ Customer Intelligence with behavioral segmentation
- ✅ Interactive visualizations
- ✅ CSV export functionality

**Planned Enhancements**
- [ ] Module 3: Marketing Insights & Recommendations
- [ ] Historical trend tracking
- [ ] Multi-company comparison mode
- [ ] Real-time data refresh
- [ ] API endpoint for programmatic access

---

## About

This project demonstrates full-stack AI engineering capabilities including:
- Multi-API orchestration and data integration
- Prompt engineering for business intelligence synthesis
- Production ML patterns (K-means clustering, feature engineering)
- Interactive data visualization
- Scalable architecture design

Built by [Su Latt](https://github.com/sulatt3) as part of an AI engineering portfolio.

**Background:** Transitioned from Senior Analytics Manager (managing 15-person team, $2B+ in incremental sales across Fortune 500 clients) to AI Engineering. This project showcases technical depth in AI systems, data processing, and production deployment.

---

## License

MIT License - See LICENSE file for details

---

## Contact

**Su Latt**
- GitHub: [@sulatt3](https://github.com/sulatt3)
- Email: su.h.latt3@gmail.com
- LinkedIn: [Su Latt](https://linkedin.com/in/su-latt)

---

## Acknowledgments

- **Segmint System:** Customer segmentation logic based on production system processing 20M+ customer events
- **News API:** Real-time news aggregation
- **Wikipedia Pageviews API:** Market interest data
- **Google Gemini:** AI synthesis and analysis
