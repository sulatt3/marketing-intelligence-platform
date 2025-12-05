# Marketing Intelligence Platform

Multi-agent marketing intelligence platform combining competitive analysis, customer segmentation, and AI-driven campaign generation.

## Current Status

**Week 1 Complete:** Competitive Intelligence Module (fully functional)

## Features

### Competitive Intelligence Module
- Multi-source data collection (News API, Google Trends)
- Smart deduplication and relevance scoring (0-100)
- AI-powered synthesis with Gemini 2.5 Pro
- Interactive Streamlit dashboard
- Automated report generation and saving

## Tech Stack

- **Python 3.12**
- **Streamlit** - Web interface
- **Gemini 2.5 Pro** - AI synthesis
- **News API** - Article collection
- **Google Trends** - Search interest data
- **Modular architecture** - Clean, reusable code

## Project Structure
```
marketing-intelligence-platform/
├── modules/
│   ├── competitive/           # Competitive intelligence (COMPLETE)
│   │   ├── data_collector.py  # Multi-source data collection
│   │   ├── analyzer.py         # Processing and AI synthesis
│   │   └── __init__.py
│   ├── segmint/               # Customer intelligence (Week 2)
│   └── campaigns/             # Campaign generation (Week 3)
├── orchestrator.py            # Main coordinator
├── streamlit_app.py          # Web interface
├── requirements.txt
└── README.md
```

## Installation
```bash
# Clone repository
git clone https://github.com/sulatt3/marketing-intelligence-platform.git
cd marketing-intelligence-platform

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export NEWSAPI_KEY="your_news_api_key"
export GEMINI_API_KEY="your_gemini_api_key"
```

## Usage
```bash
# Run Streamlit app
streamlit run streamlit_app.py
```

Or use the modules programmatically:
```python
from modules.competitive import CompetitiveDataCollector, CompetitiveAnalyzer

# Collect data
collector = CompetitiveDataCollector()
raw_data = collector.collect_all("Anthropic", use_news=True, use_trends=True)

# Analyze and generate report
analyzer = CompetitiveAnalyzer()
processed = analyzer.process_data(raw_data, "Anthropic", top_n=40)
report = analyzer.generate_insights(processed, "Anthropic")

print(report)
```

## Development Timeline

- **Week 1 (Dec 4-7):** Competitive Intelligence Module - COMPLETE
- **Week 2 (Dec 11-17):** Customer Intelligence Module + Segmint Integration
- **Week 3 (Dec 18-24):** Campaign Generation Module
- **Week 4 (Dec 25-31):** Documentation + Final Deployment

## Week 1 Deliverables

- Data collection from News API and Google Trends
- Deduplication algorithm (85% similarity threshold)
- Multi-dimensional relevance scoring (0-100)
- AI synthesis with structured prompts
- Streamlit dashboard with metrics
- Automated report saving
- Clean modular architecture

## Skills Demonstrated

- Multi-API orchestration
- Data processing pipelines
- Prompt engineering for structured output
- Web application development
- Modular code architecture
- Git workflow and version control

## Roadmap

### Week 2: Customer Intelligence
- Segmint integration for customer segmentation
- High-intent customer identification
- At-risk customer detection
- Segment profiling

### Week 3: Campaign Generation
- Multi-channel content generation (Email, Google Ads, LinkedIn, SMS)
- Competitive positioning integration
- Segment-specific personalization
- AI-powered copy creation

### Week 4: Final Integration
- Complete orchestrator implementation
- Comprehensive documentation
- Production deployment
- Demo video

## Related Projects

- **[Competitive Intelligence Agent](https://github.com/sulatt3/competitive-intelligence-agent)** - Standalone version with visualizations

## Author

**Su Latt**
- Senior Manager, Marketing Analytics at Merkle
- Transitioning to AI Engineering
- [LinkedIn](https://www.linkedin.com/in/sulatt/)
- [GitHub](https://github.com/sulatt3)

## License

MIT License

---

**Note:** This is a portfolio project demonstrating AI engineering capabilities. Week 1 (Competitive Intelligence) is production-ready.
