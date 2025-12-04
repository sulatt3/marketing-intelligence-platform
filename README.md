# Marketing Intelligence Platform

> **Status:** Under Active Development (Dec 2024)

Multi-agent marketing intelligence platform that combines competitive analysis, customer segmentation (powered by Segmint), and AI-driven campaign generation.

## Overview

This platform integrates three core modules to help marketing teams make data-driven decisions:

1. **Competitive Intelligence Module** - Monitor competitors using News API, Google Trends, and web scraping
2. **Customer Intelligence Module** - Segment customers and identify high-intent/at-risk groups (powered by Segmint segmentation engine)
3. **Campaign Generation Module** - Generate personalized multi-channel campaigns using LLMs

## Tech Stack

- **Framework:** Python, Streamlit
- **AI/ML:** Google Gemini 2.0, scikit-learn, DistilBERT
- **Data:** pandas, numpy, DuckDB
- **APIs:** News API, Google Trends, FastAPI
- **Deployment:** Streamlit Cloud, Railway

## Project Structure
```
marketing-intelligence-platform/
├── modules/
│   ├── competitive/      # Competitive intelligence module
│   ├── segmint/         # Customer intelligence (Segmint integration)
│   └── campaigns/       # Campaign generation module
├── orchestrator.py      # Coordinates all modules
├── streamlit_app.py    # Main UI
└── requirements.txt
```

## Development Timeline

- **Week 1 (Dec 4-10):** Competitive Intelligence Module
- **Week 2 (Dec 11-17):** Customer Intelligence Module + Segmint Integration
- **Week 3 (Dec 18-24):** Campaign Generation Module
- **Week 4 (Dec 25-31):** Documentation + Deployment

## Setup
```bash
# Clone repo
git clone https://github.com/sulatt3/marketing-intelligence-platform.git
cd marketing-intelligence-platform

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your API keys to .env

# Run application
streamlit run streamlit_app.py
```

## Features (In Development)

### Competitive Intelligence
- [ ] News monitoring from multiple sources
- [ ] Google Trends analysis
- [ ] Sentiment analysis
- [ ] AI-powered competitive insights

### Customer Intelligence
- [ ] Customer segmentation (5 segments)
- [ ] High-intent customer identification
- [ ] At-risk customer detection
- [ ] Segment profiling

### Campaign Generation
- [ ] Multi-channel content generation (Email, Google Ads, LinkedIn, SMS)
- [ ] Competitive positioning integration
- [ ] Segment-specific personalization
- [ ] AI-powered copy creation

## Related Projects

- **[Competitive Intelligence Agent](https://github.com/sulatt3/competitive-intelligence-agent)** - Standalone competitive analysis tool
- **Segmint API** (Coming Soon) - Standalone customer segmentation service

## Author

**Su Latt**
- Senior Analytics Manager at Merkle
- [LinkedIn](https://www.linkedin.com/in/sulatt/)
- [GitHub](https://github.com/sulatt3)

## License

MIT License - see LICENSE file for details

---

**Note:** This is a portfolio project demonstrating AI engineering capabilities. For production use, additional security, monitoring, and testing would be required.
