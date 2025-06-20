# Trading Intelligence Agent 🤖📈

An AI-powered financial intelligence system that provides real-time market analysis, sentiment tracking, and automated trading insights for the Indian stock market.

## 🚀 Overview

The Trading Intelligence Agent is a sophisticated financial analysis platform that combines web scraping, AI-powered sentiment analysis, and real-time market data to deliver actionable trading insights. Built with FastAPI and powered by Google's Agent Development Kit (ADK), this system monitors multiple data sources and generates intelligent alerts for portfolio management.

## ✨ Key Features

### 🔐 **Core Infrastructure**
- **FastAPI Backend**: High-performance REST API with JWT authentication
- **PostgreSQL Database**: Robust data storage with SQLAlchemy ORM
- **Docker Support**: Containerized deployment for easy scaling
- **API Key Authentication**: Secure access control for all endpoints

### 📊 **Data Collection & Analysis**
- **Multi-Source Web Scraping**: Automated data collection from:
  - Economic Times, Moneycontrol, LiveMint
  - NSE/BSE official feeds
  - SEBI & RBI regulatory announcements
  - Social media sentiment tracking
- **Intelligent Source Discovery**: Automatically finds and validates new information sources
- **Data Validation Pipeline**: Ensures data quality and reliability scoring

### 🧠 **AI-Powered Analysis Engine**
- **Google ADK Integration**: Advanced NLP for financial text analysis
- **Sentiment Analysis**: Real-time sentiment scoring (-1.0 to 1.0 scale)
- **Market Impact Assessment**: 1-10 scale impact prediction
- **Asset Correlation**: Links news to specific stocks/indices
- **Confidence Scoring**: Reliability metrics for all predictions

### 📈 **Market Analytics** *(In Development)*
- **Option Chain Analysis**:
  - Put-Call Ratio (PCR) tracking
  - Max Pain calculations
  - Open Interest analytics
  - Breakout zone identification
- **Participant Flow Tracking**:
  - FII/DII activity monitoring
  - Proprietary desk tracking
  - Retail sentiment analysis
- **Position Sizing Calculator**:
  - Risk-based position recommendations
  - SPAN margin calculations
  - Leverage guidelines per SEBI rules

### 🔔 **Real-Time Alert System** *(In Development)*
- **WebSocket Live Updates**: Instant push notifications
- **Multi-Channel Delivery**: WebSocket, push notifications, email
- **Smart Alert Generation**: Based on:
  - Sentiment shifts
  - Market impact scores
  - Unusual option activity
  - Participant behavior changes

### 🔧 **Advanced Features**
- **Background Task Manager**: Scheduled scraping and analysis
- **Source Reliability Scoring**: Automatic source quality assessment
- **Comprehensive Monitoring**: Health checks and performance metrics
- **Scalable Architecture**: Celery-based distributed processing

## 📋 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend/Clients                      │
│                    (Web, Mobile, Trading Apps)               │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                      FastAPI Backend                         │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Auth      │  │   REST API   │  │   WebSocket      │  │
│  │   System    │  │   Endpoints  │  │   Server         │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                    Core Services Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │  Sentiment   │  │  Market      │  │  Alert          │  │
│  │  Engine      │  │  Impact      │  │  Service        │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │  Scraper     │  │  ADK         │  │  Monitoring     │  │
│  │  Manager     │  │  Integration │  │  Service        │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  Data Layer (PostgreSQL)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │  Users   │  │  Assets  │  │  News    │  │  Alerts   │  │
│  │          │  │          │  │  Items   │  │           │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────┘  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Sources  │  │Portfolio │  │  Option  │  │Participant│  │
│  │          │  │          │  │  Chain   │  │   Flow    │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────┘  │
└──────────────────────────────────────────────────────────────┘
```

## 🛠️ Technology Stack

- **Backend Framework**: FastAPI (Python 3.11+)
- **Database**: PostgreSQL with SQLAlchemy ORM
- **AI/ML**: Google ADK, Hugging Face Transformers
- **Web Scraping**: BeautifulSoup4, aiohttp
- **Task Queue**: Celery with Redis
- **Real-time**: WebSocket (FastAPI)
- **Monitoring**: Prometheus + Custom Metrics
- **Containerization**: Docker & Docker Compose

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- PostgreSQL 14+
- Redis (for Celery)
- Docker & Docker Compose (optional)

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/your-org/trading-intelligence-agent.git
cd trading-intelligence-agent
```

2. **Set up environment variables**
```bash
cp config.env.example .env
# Edit .env with your configuration
```

3. **Using Docker (Recommended)**
```bash
docker-compose up -d
```

4. **Manual Installation**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run database migrations
alembic upgrade head

# Start the application
uvicorn app.main:app --reload
```

## 🔌 API Endpoints

### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `GET /api/v1/auth/me` - Get current user

### Portfolio Management
- `GET /api/v1/portfolio/assets` - List portfolio assets
- `POST /api/v1/portfolio/assets` - Add asset to portfolio
- `DELETE /api/v1/portfolio/assets/{id}` - Remove asset

### Alerts
- `GET /api/v1/alerts` - Get recent alerts
- `POST /api/v1/alerts/{id}/acknowledge` - Acknowledge alert
- `WebSocket /api/v1/alerts/live` - Real-time alert stream

### Analysis
- `GET /api/v1/analysis/{asset_id}` - Get comprehensive analysis
- `GET /api/v1/sentiment/{asset_id}` - Get sentiment data
- `GET /api/v1/market-impact/{asset_id}` - Get market impact assessment

### Source Management
- `GET /api/v1/sources` - List all sources
- `POST /api/v1/sources/discover` - Discover new sources
- `GET /api/v1/sources/{id}/reliability` - Get source reliability

### Market Data
- `GET /api/v1/market/option-chain/{symbol}` - Option chain data
- `GET /api/v1/market/participant-flow` - Participant activity
- `GET /api/v1/market/position-sizing` - Position recommendations

## 📊 Usage Examples

### Adding an Asset to Monitor
```python
import requests

# Add NIFTY index to portfolio
response = requests.post(
    "http://localhost:8000/api/v1/portfolio/assets",
    headers={"X-API-Key": "your-api-key"},
    json={
        "symbol": "NIFTY",
        "segment": "index",
        "alert_threshold": 7.0
    }
)
```

### Getting Real-time Alerts
```javascript
// WebSocket connection for live alerts
const ws = new WebSocket('ws://localhost:8000/api/v1/alerts/live');
ws.onmessage = (event) => {
    const alert = JSON.parse(event.data);
    console.log('New alert:', alert);
};
```

### Fetching Analysis
```python
# Get comprehensive analysis for an asset
analysis = requests.get(
    "http://localhost:8000/api/v1/analysis/1",
    headers={"X-API-Key": "your-api-key"}
).json()

print(f"Sentiment: {analysis['sentiment']['score']}")
print(f"Market Impact: {analysis['sentiment']['market_impact']}")
print(f"PCR: {analysis['option_data']['current_pcr']}")
```

## 🔍 Monitoring & Health

- **Health Check**: `GET /health`
- **Metrics**: `GET /metrics` (Prometheus format)
- **System Status**: `GET /api/v1/monitoring/status`

## 🚧 Roadmap

### Phase 1 (Completed) ✅
- Core backend infrastructure
- Data collection framework
- AI-powered sentiment analysis
- Basic monitoring and health checks

### Phase 2 (In Progress) 🔄
- Real-time WebSocket alerts
- Option chain analytics
- Participant flow tracking
- Position sizing calculator

### Phase 3 (Planned) 📋
- Advanced portfolio optimization
- Backtesting framework
- Machine learning predictions
- Mobile application
- Advanced charting integration

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting PRs.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google ADK team for the powerful AI framework
- NSE/BSE for market data access
- Open-source community for the amazing tools

## 📞 Support

For support and queries:
- 📧 Email: support@tradingintelligence.ai
- 📚 Documentation: [docs.tradingintelligence.ai](https://docs.tradingintelligence.ai)
- 💬 Discord: [Join our community](https://discord.gg/trading-intelligence)

---

**Disclaimer**: This tool is for informational purposes only. Always do your own research before making investment decisions.