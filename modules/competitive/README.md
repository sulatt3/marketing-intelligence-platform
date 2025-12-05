# Competitive Intelligence Module

AI-powered competitive intelligence analysis using multi-source data collection and synthesis.

## Status: PRODUCTION READY

## Features

- **Data Collection**
  - News API: 100+ articles from last 30 days
  - Google Trends: Search interest patterns and spike detection
  - Parallel fetching for speed (10 seconds)

- **Data Processing**
  - Deduplication (85% similarity threshold)
  - Multi-dimensional relevance scoring (0-100)
  - Smart ranking and filtering

- **AI Synthesis**
  - Gemini 2.5 Pro powered analysis
  - Structured report generation
  - Strategic recommendations

## Files

- `data_collector.py` - Multi-source data collection
- `analyzer.py` - Processing, scoring, and AI synthesis
- `__init__.py` - Module exports

## Usage
```python
from modules.competitive import CompetitiveDataCollector, CompetitiveAnalyzer

# Initialize
collector = CompetitiveDataCollector()
analyzer = CompetitiveAnalyzer()

# Collect data
raw_data = collector.collect_all("Anthropic", use_news=True, use_trends=True)

# Process and analyze
processed = analyzer.process_data(raw_data, "Anthropic", top_n=40)
report = analyzer.generate_insights(processed, "Anthropic")

print(report)
```

## Relevance Scoring Algorithm

The scoring system evaluates each item on multiple dimensions (0-100 scale):

- **Title Relevance (40 points)** - Company name in title
- **Content Relevance (20 points)** - Company name in snippet/content
- **Mention Frequency (20 points)** - Number of mentions (up to 4)
- **Recency Bonus (15 points)** - 15pts for <7 days, 10pts for <14 days, 5pts for <30 days
- **Engagement Bonus (5 points)** - Social metrics (Reddit score, Twitter likes)

## Report Structure

Generated reports include:

1. Executive Summary
2. Recent Key Moves (with dates)
3. Product Launches & Features
4. Funding, Partnerships & Hiring
5. Google Trends Analysis
6. Competitive Threats & Opportunities
7. Strategic Recommendations (3-5 items)
8. Timeline of Key Events
9. Sources

Typical report length: 1,400-1,500 words

## Performance

- Data collection: 10 seconds (parallel)
- Processing: 5 seconds
- AI synthesis: 20-30 seconds
- **Total: 45-60 seconds**

## Future Enhancements

- Reddit API integration
- Twitter/X API integration
- Web scraping for company websites
- Sentiment analysis
- Comparison mode for multiple companies
