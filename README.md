# Trading Intelligence Agent

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive AI-powered financial intelligence platform that provides real-time market insights, advanced data analysis, and automated trading signals. This enterprise-grade system combines machine learning, real-time data processing, and sophisticated financial analytics to deliver actionable intelligence for institutional and retail traders.

## Core Capabilities

### 🎯 **Intelligence & Analytics**
- **AI-Powered Sentiment Analysis**: Real-time sentiment scoring with financial domain adaptation
- **Market Impact Assessment**: Quantitative analysis of news and events on asset prices
- **Technical Analysis Engine**: 50+ technical indicators with regime-aware calculations
- **Options Analytics**: Greeks calculation, volatility surface modeling, and options flow analysis
- **Asset Correlation Matrix**: Dynamic correlation tracking across asset classes and timeframes

### 📊 **Data Processing & Ingestion**
- **Multi-Source Data Aggregation**: NSE, BSE, financial news feeds, social media, and regulatory filings
- **Real-time Market Data**: Live price feeds, order book data, and participant flow tracking
- **Automated Web Scraping**: Intelligent source discovery with validation scoring
- **Data Quality Management**: Automated validation, cleansing, and anomaly detection

### ⚡ **Real-time Systems**
- **WebSocket Communications**: Sub-second alert delivery and live data streaming
- **Event-Driven Architecture**: Kafka-style message queuing with priority-based routing
- **Push Notifications**: Firebase and APNS integration for mobile alerts
- **Live Dashboard Updates**: Real-time portfolio and market condition monitoring

### 🚀 **Trading Infrastructure**
- **Portfolio Management**: Multi-account portfolio tracking with P&L attribution
- **Position Sizing Engine**: Risk-adjusted position sizing with regime-aware models
- **Risk Management**: SPAN margin calculation and real-time risk monitoring
- **Broker Integration**: Unified API for multiple brokers (Zerodha, etc.)

### 🔐 **Enterprise Security**
- **JWT Authentication**: Secure token-based authentication with refresh tokens
- **Rate Limiting**: Advanced rate limiting with user-specific quotas
- **API Security**: Comprehensive input validation and SQL injection protection
- **Audit Logging**: Complete audit trail for all trading and analysis activities

## 🏗️ System Architecture

### **Layered Architecture Design**

```
┌─────────────────────────────────────────────────────────────┐
│                     API Gateway Layer                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │    REST     │ │  WebSocket  │ │   Authentication &      ││
│  │   API v1    │ │  Manager    │ │     Rate Limiting       ││
│  └─────────────┘ └─────────────┘ └─────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                   Service Layer                             │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────────────┐  │
│  │  Sentiment   │ │    Market    │ │    Portfolio &      │  │
│  │   Analysis   │ │    Impact    │ │      Trading        │  │
│  │   Service    │ │   Service    │ │     Services        │  │
│  └──────────────┘ └──────────────┘ └─────────────────────┘  │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────────────┐  │
│  │    Alert     │ │    Source    │ │      ADK &          │  │
│  │   Engine     │ │  Discovery   │ │    Monitoring       │  │
│  └──────────────┘ └──────────────┘ └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                  Data Processing Layer                      │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────────────┐  │
│  │   Analysis   │ │     Task     │ │    Data Pipeline    │  │
│  │   Engines    │ │   Queuing    │ │   & Validation      │  │
│  │              │ │   (Celery)   │ │                     │  │
│  └──────────────┘ └──────────────┘ └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────────────┐  │
│  │ PostgreSQL   │ │    Redis     │ │   External APIs     │  │
│  │   Database   │ │    Cache     │ │  (NSE, News, etc.)  │  │
│  └──────────────┘ └──────────────┘ └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### **Technology Stack**

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **API** | FastAPI 0.115+ | High-performance async API framework |
| **Database** | PostgreSQL 15+ | Primary data storage with ACID compliance |
| **Cache** | Redis 7+ | Session storage, task queuing, real-time data |
| **Task Queue** | Celery 5.5+ | Distributed background task processing |
| **Authentication** | JWT + Passlib | Secure token-based authentication |
| **Real-time** | WebSockets | Bi-directional real-time communication |
| **Data Processing** | Pandas + NumPy | High-performance data analysis |
| **Web Scraping** | BeautifulSoup + aiohttp | Concurrent web scraping and data ingestion |
| **Monitoring** | Prometheus + Structlog | System metrics and structured logging |
| **Deployment** | Docker + Compose | Containerized deployment and orchestration |

### **Core Design Patterns**

- **Microservices Architecture**: Loosely coupled services with clear boundaries
- **Event-Driven Processing**: Asynchronous event handling with priority queuing  
- **Repository Pattern**: Clean separation between data access and business logic
- **Factory Pattern**: Service instantiation with dependency injection
- **Observer Pattern**: Real-time event propagation and alert delivery
- **Strategy Pattern**: Pluggable analysis algorithms and trading strategies

## 🔄 Data Flow & Workflows

### **Real-time Data Processing Workflow**

```mermaid
graph TD
    A[External Data Sources] --> B[Data Ingestion Layer]
    B --> C{Data Validation}
    C -->|Valid| D[Data Pipeline]
    C -->|Invalid| E[Error Handling]
    D --> F[Analysis Engines]
    F --> G[Alert Generation]
    F --> H[Database Storage]
    G --> I[WebSocket Manager]
    I --> J[Client Applications]
    H --> K[Dashboard Updates]
```

### **Task Processing Architecture**

The system employs a sophisticated task queue architecture with priority-based processing:

| Queue | Priority | Purpose | Example Tasks |
|-------|----------|---------|---------------|
| `high_priority` | 10 | Time-critical operations | Real-time price updates, critical alerts |
| `market_data` | 8 | Market data processing | Option chains, participant flow |
| `alerts` | 9 | Alert processing | Notification delivery, acknowledgments |
| `analysis` | 6 | Data analysis tasks | Sentiment analysis, correlation calculations |
| `scraping` | 5 | Data collection | News scraping, source discovery |
| `monitoring` | 3 | System health | Health checks, metrics collection |
| `maintenance` | 1 | Cleanup operations | Old data removal, log rotation |

### **Sentiment Analysis Pipeline**

1. **Text Ingestion**: News articles, social media, regulatory filings
2. **Preprocessing**: Text cleaning, financial entity recognition
3. **Model Selection**: Financial vs. general domain models
4. **Sentiment Scoring**: Multi-dimensional sentiment analysis
5. **Trend Analysis**: Historical sentiment pattern detection
6. **Alert Generation**: Sentiment-based trading signals
7. **Dashboard Updates**: Real-time sentiment visualization

### **Market Impact Assessment Workflow**

```python
# Example workflow for market impact assessment
1. Event Detection: News article published
2. Asset Identification: Extract mentioned securities (NER)
3. Sentiment Analysis: Quantify news sentiment (-1 to +1)
4. Impact Scoring: Calculate potential market impact (0-10)
5. Correlation Analysis: Check with historical patterns
6. Alert Generation: Trigger alerts for high-impact events
7. Portfolio Assessment: Evaluate impact on user positions
```

### **Alert Delivery & Acknowledgment System**

The platform implements a sophisticated alert delivery system with guaranteed delivery:

1. **Alert Generation**: Triggered by analysis engines or market conditions
2. **Priority Classification**: High/Medium/Low based on impact and urgency
3. **Multi-Channel Delivery**: WebSocket, Push Notifications, Email
4. **Acknowledgment Tracking**: User response monitoring with timeouts
5. **Escalation Logic**: Automatic escalation for unacknowledged critical alerts
6. **Delivery Confirmation**: End-to-end delivery verification

## Getting Started

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- An environment file `config.env` based on `config.env.example`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Hisham1404/Trade_backend.git
    cd Trade_backend
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `config.env` file by copying the example:
    ```bash
    cp config.env.example config.env
    ```
    Update `config.env` with your database credentials, API keys, and other settings.

### Running the Application

#### Without Docker

To run the application directly with Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The application will be available at `http://localhost:8000`.

#### With Docker

To build and run the application using Docker Compose:

```bash
docker-compose up --build
```

This will start the FastAPI application along with a PostgreSQL database and Redis instance.

## 🚀 API Reference

The platform provides a comprehensive REST API with real-time WebSocket capabilities. All endpoints follow RESTful conventions with consistent error handling and response formatting.

### **Core API Endpoints**

| Endpoint | Method | Description | Authentication |
|----------|--------|-------------|----------------|
| `/` | GET | API health and version information | Public |
| `/docs` | GET | Interactive API documentation (Swagger) | Public |
| `/health` | GET | System health check | Public |

### **Authentication & Security** (`/api/v1/auth`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /register` | Register new user account |
| `POST /login` | Authenticate user and get JWT token |
| `POST /refresh` | Refresh expired JWT token |
| `POST /logout` | Invalidate JWT token |
| `GET /profile` | Get current user profile |

### **Portfolio Management** (`/api/v1/portfolios`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /` | List user portfolios |
| `POST /` | Create new portfolio |
| `GET /{id}` | Get portfolio details |
| `PUT /{id}` | Update portfolio |
| `DELETE /{id}` | Delete portfolio |
| `POST /{id}/positions` | Add position to portfolio |
| `GET /{id}/performance` | Get portfolio performance metrics |

### **Market Data & Assets** (`/api/v1/assets`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /search?q={query}` | Search for assets by symbol or name |
| `GET /{symbol}` | Get detailed asset information |
| `GET /{symbol}/price` | Get current price and basic metrics |
| `GET /{symbol}/history` | Get historical price data |
| `GET /{symbol}/options` | Get options chain data |

### **Advanced Analytics** (`/api/v1/analysis`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /comprehensive/{symbol}` | Complete analysis suite for an asset |
| `POST /technical-indicators` | Calculate technical indicators |
| `POST /sentiment-trend` | Analyze sentiment trends |
| `POST /market-regime` | Market regime analysis |
| `GET /position-sizing/{symbol}` | Position sizing recommendations |
| `GET /correlation-matrix` | Asset correlation analysis |

### **News & Sentiment** (`/api/v1/news`, `/api/v1/sentiment`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /news` | Get latest news articles |
| `GET /news/{id}` | Get specific news article |
| `POST /sentiment/analyze` | Analyze text sentiment |
| `GET /sentiment/{symbol}` | Get asset sentiment data |
| `GET /sentiment/market-overview` | Overall market sentiment |

### **Alerts & Notifications** (`/api/v1/alerts`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /` | List user alerts |
| `POST /` | Create new alert |
| `GET /{id}` | Get alert details |
| `PUT /{id}` | Update alert |
| `DELETE /{id}` | Delete alert |
| `POST /{id}/acknowledge` | Acknowledge alert |

### **Market Impact Analysis** (`/api/v1/market-impact`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /{symbol}/summary` | Get impact summary for asset |
| `POST /analyze-news` | Analyze news impact on market |
| `GET /correlation-impact` | Cross-asset impact analysis |

### **Real-time WebSocket API**

| Endpoint | Type | Description |
|----------|------|-------------|
| `/ws/connect` | WebSocket | Real-time data and alert delivery |
| `/ws/market-data` | WebSocket | Live market data stream |
| `/ws/alerts` | WebSocket | Real-time alert notifications |

#### WebSocket Message Types

```json
// Alert Message
{
  "type": "alert",
  "alert_id": "12345",
  "title": "Price Alert",
  "message": "AAPL reached $150",
  "severity": "high",
  "timestamp": "2024-01-15T10:30:00Z",
  "requires_acknowledgment": true
}

// Market Data Update
{
  "type": "market_data",
  "symbol": "AAPL",
  "price": 150.25,
  "change": 2.50,
  "volume": 1250000,
  "timestamp": "2024-01-15T10:30:00Z"
}

// System Message
{
  "type": "system_message",
  "message": "System maintenance in 5 minutes",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### **Monitoring & Administration**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/monitoring/health` | GET | Detailed system health status |
| `/monitoring/metrics` | GET | Prometheus metrics |
| `/monitoring/celery` | GET | Background task queue status |
| `/monitoring/database` | GET | Database connection status |

### **Response Format**

All API responses follow a consistent format:

```json
{
  "success": true,
  "data": { /* Response data */ },
  "message": "Operation completed successfully",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_12345"
}
```

### **Error Handling**

Error responses include detailed information for debugging:

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid asset symbol",
    "details": { /* Detailed error information */ }
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_12345"
}
```

## ⚙️ Configuration

The system uses a hierarchical configuration approach with environment-specific overrides:

### **Core Settings** (`config.env`)

```bash
# Application
APP_NAME="Trading Intelligence Agent"
APP_VERSION="1.0.0"
DEBUG=false
ENVIRONMENT="production"

# Security
SECRET_KEY="your-super-secure-secret-key"
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database
DATABASE_URL="postgresql://user:password@localhost:5432/trading_db"
REDIS_URL="redis://localhost:6379/0"

# External APIs
ALPHA_VANTAGE_API_KEY="your_api_key"
FINNHUB_API_KEY="your_api_key"
NEWS_API_KEY="your_api_key"

# Celery Configuration
CELERY_BROKER_URL="redis://localhost:6379/0"
CELERY_RESULT_BACKEND="redis://localhost:6379/1"
CELERY_TASK_TIME_LIMIT=3600
CELERY_WORKER_PREFETCH_MULTIPLIER=1

# Push Notifications
FIREBASE_CREDENTIALS_PATH="/path/to/firebase-credentials.json"
APNS_KEY_PATH="/path/to/apns-key.p8"
APNS_KEY_ID="your_key_id"
APNS_TEAM_ID="your_team_id"

# Rate Limiting & Security
RATE_LIMIT_PER_MINUTE=60
ALLOWED_ORIGINS="http://localhost:3000,http://localhost:8080"

# Monitoring
PROMETHEUS_ENABLED=true
LOG_LEVEL="INFO"
```

### **Environment-Specific Configuration**

| Environment | Database | Debug | Celery Eager | Log Level |
|-------------|----------|-------|--------------|-----------|
| Development | SQLite | True | True | DEBUG |
| Testing | PostgreSQL | True | True | INFO |
| Staging | PostgreSQL | False | False | INFO |
| Production | PostgreSQL | False | False | WARNING |

### **Task Queue Configuration**

The Celery configuration supports different deployment scenarios:

```python
# Development (Eager execution)
CELERY_TASK_ALWAYS_EAGER = True

# Production (Distributed processing)
CELERY_TASK_ALWAYS_EAGER = False
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
CELERY_TASK_ACKS_LATE = True
```

## 🚀 Deployment

### **Docker Compose (Recommended)**

The simplest way to deploy the entire stack:

```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/trading_agent_db
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./config.env:/app/config.env

  celery_worker:
    build: .
    command: celery -A app.tasks.celery_app worker --loglevel=info --queues=high_priority,market_data,alerts,analysis
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/trading_agent_db
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

  celery_beat:
    build: .
    command: celery -A app.tasks.celery_app beat --loglevel=info
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/trading_agent_db
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=trading_agent_db
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data/

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### **Production Deployment**

For production deployments, consider:

#### **Load Balancing & High Availability**

```bash
# Deploy multiple API instances
docker-compose up --scale web=3

# Use nginx for load balancing
upstream trading_api {
    server web1:8000;
    server web2:8000;
    server web3:8000;
}
```

#### **Celery Worker Scaling**

```bash
# Scale workers by queue type
docker-compose up --scale celery_worker_high_priority=2
docker-compose up --scale celery_worker_analysis=3
docker-compose up --scale celery_worker_scraping=2
```

#### **Database Configuration**

```sql
-- Production PostgreSQL optimizations
-- postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.7
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
```

### **Monitoring Stack**

Deploy monitoring with Prometheus and Grafana:

```yaml
# monitoring/docker-compose.yml
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

  redis_exporter:
    image: oliver006/redis_exporter
    ports:
      - "9121:9121"
    environment:
      - REDIS_ADDR=redis://redis:6379

volumes:
  grafana_data:
```

### **Security Considerations**

#### **SSL/TLS Configuration**

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    
    location / {
        proxy_pass http://trading_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /ws/ {
        proxy_pass http://trading_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

#### **Database Security**

```bash
# Create dedicated database user
CREATE USER trading_agent WITH PASSWORD 'secure_password';
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO trading_agent;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO trading_agent;

# Enable SSL connections
ssl = on
ssl_cert_file = 'server.crt'
ssl_key_file = 'server.key'
```

### **Performance Optimization**

#### **Database Indexing**

```sql
-- Key indexes for performance
CREATE INDEX CONCURRENTLY idx_news_published_at ON news_items(published_at DESC);
CREATE INDEX CONCURRENTLY idx_alerts_user_status ON alerts(user_id, status);
CREATE INDEX CONCURRENTLY idx_portfolio_user_active ON portfolios(user_id) WHERE is_active = true;
CREATE INDEX CONCURRENTLY idx_participant_flow_date_symbol ON participant_flow(date, symbol);
```

#### **Redis Configuration**

```conf
# redis.conf
maxmemory 512mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### **Backup Strategy**

```bash
#!/bin/bash
# backup.sh

# Database backup
pg_dump -h localhost -U postgres trading_agent_db | gzip > backup_$(date +%Y%m%d_%H%M%S).sql.gz

# Redis backup
redis-cli --rdb backup_redis_$(date +%Y%m%d_%H%M%S).rdb

# Upload to S3 (optional)
aws s3 cp backup_*.gz s3://your-backup-bucket/database/
aws s3 cp backup_*.rdb s3://your-backup-bucket/redis/
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. 