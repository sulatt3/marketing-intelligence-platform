# Marketing Intelligence Platform

AI-powered competitive analysis and customer segmentation with hybrid LLM scoring and comprehensive evaluation framework.

**Live Demo:** [marketing-intelligence-platform-su.streamlit.app](https://marketing-intelligence-platform-su.streamlit.app)  
**Access:** Password-protected portfolio demo. Contact su.h.latt3@gmail.com for password.

---

## Overview

The Marketing Intelligence Platform integrates real-time competitive intelligence with behavioral customer segmentation to provide actionable insights for marketing teams. Built with production-grade AI orchestration, hybrid LLM scoring, and comprehensive evaluation frameworks, the platform demonstrates enterprise-scale data processing patterns and responsible AI deployment.

**Key Innovation:** Hybrid relevance scoring combining rule-based pre-filtering with LLM semantic validation, plus comprehensive evaluation system including data quality scoring, ML model validation, and LLM output assessment with hallucination detection.

### Core Features

**Module 1: Competitive Intelligence**
- Multi-source data integration (News API, Wikipedia Pageviews)
- **Hybrid LLM scoring:** Rule-based pre-filter + Llama 3.3 semantic validation
- AI-powered synthesis using Llama 3.3 70B (via Groq)
- Smart deduplication (85% similarity threshold)
- Multi-dimensional relevance scoring
- Sentiment analysis and trend detection
- Interactive visualizations with timeline and sentiment breakdowns
- LLM output evaluation with citation tracking

**Module 2: Customer Intelligence**
- Behavior-based K-means clustering (5 segments)
- Example model based on production Segmint system (20M+ events, 28.95% conversion rates)
- Multi-dimensional behavioral analysis
- Conversion rate modeling with recency penalties
- Feature correlation analysis
- CSV export for CRM integration
- Note: Uses synthetic data for demonstration purposes

**Module 3: Data Quality & Evaluation Metrics**
- Hybrid scoring pipeline metrics (rule-based + LLM validation)
- Data quality scoring (0-100 composite score)
- LLM output evaluation (hallucination detection via citation rate)
- ML clustering metrics (silhouette score, Davies-Bouldin, elbow method)
- EDA and preprocessing documentation
- Comprehensive evaluation methodology

---

## Technical Architecture

### Hybrid Scoring Pipeline
```
News API (150 articles)
    ↓
Rule-Based Pre-Filter (top 100 candidates)
    ↓
LLM Batch Scoring (Llama 3.3 scores each 0-100 via Groq)
    ↓
Quality Threshold (keep articles ≥50)
    ↓
User Selection (top N by LLM score)
    ↓
Final Report Generation (Llama 3.3 via Groq)
```

### Tech Stack

**AI & ML**
- Groq API with Llama 3.3 70B (competitive intelligence synthesis + hybrid scoring)
- scikit-learn (K-means clustering, StandardScaler)
- Custom NLP (sentiment analysis, rule-based pre-filtering)

**Data Processing**
- pandas, numpy (data manipulation)
- concurrent.futures (parallel API calls)

**Framework & Visualization**
- Streamlit (UI/deployment)
- Plotly Express & Graph Objects (interactive charts)

**APIs**
- Groq API (Llama 3.3 70B), News API, Wikipedia Pageviews API

---

## Installation

### Prerequisites
- Python 3.9+
- [Groq API key](https://console.groq.com/) (FREE, unlimited usage)
- [News API key](https://newsapi.org/) (free tier: 100 requests/day)

### Setup

1. **Clone and install**
```bash
git clone https://github.com/sulatt3/marketing-intelligence-platform.git
cd marketing-intelligence-platform
pip install -r requirements.txt
```

2. **Configure API keys and password** (`.streamlit/secrets.toml`):
```toml
NEWSAPI_KEY = "your_newsapi_key_here"
GROQ_API_KEY = "your_groq_key_here"
DEMO_PASSWORD = "your_demo_password"
```

3. **Run locally**
```bash
streamlit run streamlit_app.py
```

---

## Usage

### Demo Access
The live demo is password-protected to preserve API credits for reviewers.

**For recruiters/interviewers:** Contact su.h.latt3@gmail.com for demo password.

### Competitive Intelligence

1. **Enter password** (first-time access)
2. **Enter company name** (e.g., "Anthropic", "OpenAI", "Perplexity")
3. **Adjust article count** (20-100, default: 40)
4. **Generate report** - Hybrid scoring + AI synthesis in 20-30 seconds

**What happens behind the scenes:**
- Collects 150+ articles from News API
- Rule-based pre-filter selects top 100 candidates
- Llama 3.3 70B batch-scores each article 0-100 for semantic relevance
- Filters to articles scoring ≥50 (LLM-approved)
- Selects top N articles based on your slider
- Generates comprehensive strategic report using Llama 3.3

**Output includes:**
- Executive summary
- Recent strategic moves
- Product launches & features
- Funding, partnerships, hiring
- Market interest trends
- Competitive threats and opportunities
- Strategic recommendations
- Downloadable markdown report

### Customer Intelligence

Auto-generated segments (1,000 synthetic customers) with:
- 5 behavioral segments with conversion rates
- Purchase behavior analysis
- Segment performance metrics
- Export functionality

**Note:** Synthetic data for demonstration. Segmentation logic replicates production system (20M+ events, 28.95% conversion).

### Data Quality & Metrics

**Competitive Intelligence Metrics:**
- Hybrid scoring pipeline breakdown
- LLM validation pass rates
- Data quality score
- LLM output evaluation (citation rate, specificity, completeness)

**Customer Intelligence Metrics:**
- Clustering quality (silhouette, Davies-Bouldin)
- Elbow method visualization
- Feature correlation analysis
- EDA statistics

---

## Key Technical Implementations

### 1. Hybrid LLM Scoring
**Two-stage relevance system:**
- **Stage 1:** Rule-based pre-filter (keyword matching, recency, source authority)
- **Stage 2:** LLM semantic validation (Llama 3.3 scores 0-100 via Groq)
- **Threshold:** Articles must score ≥50 to be included
- **Fallback:** Uses rule-based only if <20 articles collected

### 2. LLM Batch Scoring
Efficient single API call to score 100 articles simultaneously with structured JSON output.

### 3. Smart Deduplication
SequenceMatcher fuzzy matching (85% similarity threshold) eliminates near-duplicates.

### 4. Multi-Dimensional Relevance (Rule-Based)
7 scoring factors:
- Title/content analysis (70 points)
- Contextual keywords (15 points)
- Positioning in article (10 points)
- Recency weighting (20 points)
- Content quality (10 points)
- Source authority (5 points)
Max: 100 points

### 5. Behavior-Based Segmentation
Priority-based labeling ensures unique, meaningful segments:
1. Churn risk (recency >80 days)
2. High value (AOV >$1,000)
3. High frequency (purchases >4.5)
4. High engagement (clicks >23, purchases >2.5)
5. Low conversion (purchases <2.0)

### 6. Conversion Rate Estimation
```
Base Rate = (Avg Purchases / Avg Clicks) × 100
Recency Penalty = 1 / (1 + Days Since Purchase / 30)
Final Rate = Base Rate × Recency Penalty
```

### 7. LLM Output Evaluation
**Comprehensive framework:**
- Section completeness (9 expected sections)
- Citation rate (hallucination detection)
- Specificity score (dates, numbers, concrete facts)
- Length appropriateness (1,000-1,500 words target)

### 8. Production Safeguards
- Password protection (preserves API credits)
- 2-minute rate limiting between requests
- Request-in-progress tracking
- Usage tracking and budget monitoring
- Graceful error handling

---

## Evaluation Framework

### Hybrid Scoring Metrics
- Articles collected
- Rule-based filter pass rate
- LLM validation pass rate
- Quality threshold effectiveness

### Data Quality Metrics
- Completeness rate, deduplication rate, source diversity, temporal coverage, relevance distribution

### ML Model Metrics
- Silhouette score, Davies-Bouldin score, elbow method, segment balance, feature correlation

### LLM Output Metrics
- Section completeness, citation rate (hallucination detection), specificity score, length appropriateness

---

## Cost & Performance

### API Usage Per Report
- **News API:** 1 call (free tier: 100/day)
- **Groq API:** 2 calls (batch scoring + report generation)
- **Cost:** $0 - Completely free, unlimited usage
- **Time:** 15-20 seconds total (Groq is very fast)

### Unlimited Free Usage
- **No credits to manage** - Groq is completely free
- Password protection prevents abuse
- No daily quotas or rate limits for reasonable usage

---

## Project Structure

```
marketing-intelligence-platform/
├── streamlit_app.py          # Main application
├── requirements.txt           # Python dependencies
├── .streamlit/
│   └── secrets.toml          # API keys + password (not in git)
├── README.md                 # Documentation
└── .gitignore               # Git ignore rules
```

---

## Dependencies

```
streamlit>=1.28.0
groq>=0.4.0
newsapi-python==0.2.7
plotly>=5.17.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
requests>=2.31.0
```

---

## Deployment

### Streamlit Cloud (Recommended)

1. Push to GitHub
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. **Add secrets:**
   ```toml
   NEWSAPI_KEY = "your_key"
   GROQ_API_KEY = "your_key"
   DEMO_PASSWORD = "your_password"
   ```
4. Deploy

### Alternative Platforms
- **Heroku:** Use `Procfile` with `web: streamlit run streamlit_app.py`
- **AWS EC2:** Run with systemd service
- **Docker:** Create `Dockerfile` with Streamlit base image

---

## Production Considerations

### Rate Limiting
- **News API:** 100 requests/day (free tier)
- **Groq API:** Unlimited free usage
- **App safeguards:** 2-min cooldown, request tracking

### Scalability
- Session state management for multi-user environments
- Caching with TTL (1 hour) reduces redundant API calls
- Parallel API orchestration with ThreadPoolExecutor

### Monitoring
- Data quality dashboard
- LLM evaluation metrics
- Usage tracking
- Error handling with user-friendly messaging

---

## Roadmap

**Completed:**
- ✅ Competitive Intelligence with hybrid LLM scoring
- ✅ Customer Intelligence with behavioral segmentation
- ✅ Comprehensive evaluation framework (data + ML + LLM)
- ✅ Password protection and usage tracking
- ✅ Production deployment with safeguards

**Planned Enhancements:**
- [ ] User feedback loop and rating system
- [ ] Next Module: Marketing Insights & Recommendations engine
- [ ] Multi-company comparison mode
- [ ] Historical trend tracking and time-series analysis
- [ ] Custom prompt templates for different industries
- [ ] Export to PDF and enhanced report formats
- [ ] Automated alerts for competitive moves

---

## About

This project demonstrates full-stack AI engineering capabilities including:
- **Hybrid LLM Architecture:** Combining rule-based efficiency with semantic understanding
- Multi-API orchestration and data integration
- Prompt engineering for business intelligence synthesis
- Production ML patterns (K-means, feature engineering, evaluation)
- LLM output quality assessment and hallucination detection
- Interactive data visualization
- Scalable architecture design
- Responsible AI deployment practices

Built by [Su Latt](https://github.com/sulatt3) as part of an AI engineering portfolio.

**Background:** Senior Analytics and AI Manager. 6+ years experience managing 15-person team, $2B+ impact across Fortune 500 clients. This project showcases technical depth in AI systems, hybrid architectures, ML implementation, and production deployment.

**Key Achievement:** Customer segmentation logic based on production Segmint system that processed 20M+ customer events and achieved conversion rates up to 28.95%.

---

## Technical Highlights for Interviews

**Hybrid LLM Scoring:**
> "I implemented a two-stage hybrid scoring system. Rule-based pre-filtering handles the obvious cases for speed and cost-efficiency, then Llama 3.3 70B (via Groq) provides semantic validation by batch-scoring the top 100 candidates. Only articles scoring ≥50 are included in the final analysis. This combines deterministic efficiency with LLM semantic understanding at zero cost using Groq's free inference API."

**Cost Optimization:**
> "I chose Groq's Llama 3.3 70B for production deployment because it provides excellent quality (comparable to Claude/GPT-4) with unlimited free usage. For a portfolio demo that might be accessed by multiple recruiters, this eliminates budget constraints while maintaining professional-grade outputs. In production, I'd evaluate the cost-quality trade-off and potentially upgrade to Claude or GPT-4 if needed."

**Hallucination Detection:**
> "I track citation rate in LLM-generated reports - what percentage of provided source articles are actually referenced. Low citation rates (<40%) indicate potential hallucinations. I also measure specificity by counting dates, numbers, and concrete facts versus generic statements."

**Production Thinking:**
> "The platform includes password protection to preserve API credits, 2-minute rate limiting to prevent abuse, usage tracking for budget monitoring, and graceful degradation when APIs fail. These aren't just nice-to-haves - they're essential for any production AI system."

---

## License

MIT License

---

## Contact

**Su Latt**
- GitHub: [@sulatt3](https://github.com/sulatt3)
- Email: su.h.latt3@gmail.com
- LinkedIn: [su-l-67630a67](https://www.linkedin.com/in/su-l-67630a67/)

---

## Acknowledgments

- **Groq:** Free unlimited LLM inference with Llama 3.3 70B
- **Segmint System:** Customer segmentation logic from production system (20M+ events)
- **News API:** Real-time news aggregation
- **Wikipedia Pageviews API:** Market interest data
- **scikit-learn:** Machine learning implementation
