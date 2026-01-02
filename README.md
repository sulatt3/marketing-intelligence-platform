# Marketing Intelligence Platform

AI-powered competitive analysis and customer segmentation with comprehensive evaluation framework.

**Live Demo:** [marketing-intelligence-platform-su.streamlit.app](https://marketing-intelligence-platform-su.streamlit.app)

---

## Overview

The Marketing Intelligence Platform integrates real-time competitive intelligence with behavioral customer segmentation to provide actionable insights for marketing teams. Built with production-grade AI orchestration and comprehensive evaluation frameworks, the platform demonstrates enterprise-scale data processing patterns and responsible AI deployment.

**Key Innovation:** Comprehensive evaluation system including data quality scoring, ML model validation, and LLM output assessment with hallucination detection.

### Core Features

**Module 1: Competitive Intelligence**
- Multi-source data integration (News API, Wikipedia Pageviews)
- AI-powered synthesis using Claude Sonnet 4
- Smart deduplication (85% similarity threshold)
- Multi-dimensional relevance scoring
- Sentiment analysis and trend detection
- Interactive visualizations with timeline and sentiment breakdowns
- LLM output evaluation with citation tracking

**Module 2: Customer Intelligence**
- Behavior-based K-means clustering (5 segments)
- Based on production Segmint system (20M+ events, 28.95% conversion rates)
- Multi-dimensional behavioral analysis
- Conversion rate modeling with recency penalties
- Feature correlation analysis
- CSV export for CRM integration

**Module 3: Data Quality & Evaluation Metrics**
- Data quality scoring (0-100 composite score)
- LLM output evaluation (hallucination detection via citation rate)
- ML clustering metrics (silhouette score, Davies-Bouldin, elbow method)
- EDA and preprocessing documentation
- Comprehensive evaluation methodology

---

## Technical Architecture

### Data Pipeline
```
Multi-Source Collection → Quality Scoring → AI Synthesis → Evaluation
         ↓                      ↓               ↓              ↓
    News API +          Deduplication      Claude API    Citation Rate
    Wikipedia           Relevance Score    Synthesis     Specificity Check
    Parallel Fetch      Validation         Prompting     Quality Metrics
```

### Tech Stack

**AI & ML**
- Anthropic Claude Sonnet 4 (competitive intelligence synthesis)
- scikit-learn (K-means clustering, StandardScaler)
- Custom NLP (sentiment analysis, relevance scoring)

**Data Processing**
- pandas, numpy (data manipulation)
- concurrent.futures (parallel API calls)

**Framework & Visualization**
- Streamlit (UI/deployment)
- Plotly Express & Graph Objects (interactive charts)

**APIs**
- Anthropic Claude API, News API, Wikipedia Pageviews API

---

## Installation

### Prerequisites
- Python 3.9+
- [Anthropic Claude API key](https://console.anthropic.com/) (free tier: $5 credit)
- [News API key](https://newsapi.org/) (free tier: 100 requests/day)

### Setup

1. **Clone and install**
```bash
git clone https://github.com/sulatt3/marketing-intelligence-platform.git
cd marketing-intelligence-platform
pip install -r requirements.txt
```

2. **Configure API keys** (`.streamlit/secrets.toml`):
```toml
NEWSAPI_KEY = "your_newsapi_key_here"
ANTHROPIC_API_KEY = "your_anthropic_key_here"
```

3. **Run locally**
```bash
streamlit run streamlit_app.py
```

---

## Usage

### Competitive Intelligence
1. Enter company name → Generate report
2. View AI-synthesized strategic analysis
3. Explore visualizations (sentiment, timeline)
4. Download markdown report

### Customer Intelligence
1. Auto-generated segments (1,000 synthetic customers)
2. Explore 5 behavioral segments with conversion rates
3. View distributions, correlations, performance metrics
4. Export CSV for CRM integration

### Data Quality & Metrics
1. Review data quality scores and cleaning metrics
2. Evaluate LLM output quality (completeness, citations, specificity)
3. Validate ML clustering performance (silhouette, elbow plot)
4. Inspect feature correlations and preprocessing steps

---

## Key Technical Implementations

### 1. Smart Deduplication
Fuzzy string matching (85% similarity) eliminates near-duplicate articles

### 2. Multi-Dimensional Relevance Scoring
- Title mentions: 40pts | Content: 20pts | Frequency: 20pts | Recency: 15pts

### 3. LLM Output Evaluation
- **Section Completeness:** 9 expected sections validation
- **Citation Rate:** Hallucination detection (% source articles referenced)
- **Specificity Score:** Concrete facts vs. generic statements
- **Overall Quality:** 0-100 composite metric

### 4. ML Clustering Metrics
- Silhouette score (cluster separation)
- Davies-Bouldin score (cluster cohesion)
- Elbow method (optimal k validation)

### 5. Conversion Rate Modeling
```
Base = (Purchases / Clicks) × 100
Penalty = 1 / (1 + Days/30)
Final = Base × Penalty
```

---

## Evaluation Framework

### Data Quality
Completeness, deduplication rate, source diversity, temporal coverage, relevance distribution

### ML Model
Silhouette score, Davies-Bouldin score, elbow method, segment balance, feature correlation

### LLM Output
Section completeness, citation rate (hallucination detection), specificity score, length appropriateness

---

## Deployment

**Streamlit Cloud** (recommended): Connect GitHub → Add secrets → Deploy

**Production considerations:**
- Rate limiting (2-min cooldown)
- Parallel API orchestration
- Session state management
- Graceful error handling

---

## Roadmap

**Completed:**
- ✅ Competitive Intelligence with AI synthesis
- ✅ Customer segmentation with evaluation
- ✅ Comprehensive metrics (data + ML + LLM)

**Planned:**
- [ ] User feedback loop
- [ ] Module 3: Marketing Insights
- [ ] Multi-company comparison
- [ ] Historical tracking
- [ ] Custom prompt templates
- [ ] Enhanced exports (PDF)

---

## About

**Portfolio Project by Su Latt** | [GitHub](https://github.com/sulatt3) | [LinkedIn](https://linkedin.com/in/su-latt)

Demonstrates AI engineering capabilities:
- Multi-API orchestration
- LLM prompt engineering & evaluation
- Production ML implementation
- Data quality frameworks
- Responsible AI deployment

**Background:** Senior Analytics Manager → AI Engineering transition. 6+ years experience managing 15-person team, $2B+ impact across Fortune 500 clients. Customer segmentation based on production Segmint system (20M+ events, 28.95% conversion).

---

## Contact

**Su Latt**
- GitHub: [@sulatt3](https://github.com/sulatt3)
- Email: su.h.latt3@gmail.com
- LinkedIn: [su-latt](https://linkedin.com/in/su-latt)

---

## License

MIT License
