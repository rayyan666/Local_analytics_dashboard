# Local Analytic Chatbot - System Architecture

## Overview

The Local Analytic Chatbot is a self-contained, AI-powered data analysis platform that combines an LLM (Large Language Model) with a modern web interface to provide professional data analysis, visualization, and business insights on local infrastructure.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Static)                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ index.html    │ dashboard.html    │ CSS & JavaScript    │   │
│  │ Chat UI       │ Data Explorer     │ Modern & Responsive │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↕
                         HTTP REST API
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend Server                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Endpoints:                                               │   │
│  │ • /upload - File upload & profiling                      │   │
│  │ • /chat - Single file analysis with code generation     │   │
│  │ • /analyze - Multi-file analysis & summaries            │   │
│  │ • /load-data - CSV data retrieval                        │   │
│  │ • /dashboard - Dashboard interface                      │   │
│  │ • /list-files - List uploaded files                     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
    │              │              │              │
    ↓              ↓              ↓              ↓
┌─────────┐ ┌────────────┐ ┌──────────┐ ┌──────────────┐
│ LLM     │ │ Executors  │ │ Reports  │ │ Data Utils   │
│ Adapter │ │ - Python   │ │ - PDF    │ │ - Profiler   │
│ (Llama) │ │ - SQL      │ │ - Charts │ │ - Connectors │
└─────────┘ └────────────┘ └──────────┘ └──────────────┘
    │              │              │              │
    ↓              ↓              ↓              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Data & Model Storage Layer                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ /data/uploads/      - CSV Files & User Data              │   │
│  │ /models/            - GGUF Model Files                   │   │
│  │ /data/[analysis]/   - Generated Reports & Outputs       │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### 1. Frontend Layer (`/static`)

#### `index.html` - Main Chat Interface
- **Purpose**: Primary user interface for single and multi-file data analysis
- **Features**:
  - File upload with drag-and-drop support
  - File list with multi-select checkboxes
  - Chat window for interactive analysis queries
  - Result display with JSON and chart visualization
  - Data profile summaries
  - Real-time response timing
  - PDF report download

- **Key JavaScript Functions**:
  - `uploadFile()` - Upload CSV files
  - `sendChat()` - Send analysis queries
  - `handleResult()` - Process LLM responses
  - `updateFileBadge()` - Update file selection UI
  - `appendMessage()` - Render chat messages

#### `dashboard.html` - Data Explorer
- **Purpose**: Live data exploration with interactive visualizations
- **Features**:
  - Real-time column filtering and selection
  - 6 chart types (bar, line, area, pie, doughnut, scatter)
  - Unique value browser
  - Data table view with scrolling
  - Responsive grid layout

- **Components**:
  - Sidebar (280px) - Controls and filters
  - Main content area - Charts and data tables
  - Scrollable containers - Dynamic heights

### 2. Backend Layer (`/backend`)

#### `fastapi_app.py` - Main Application Server
- **Framework**: FastAPI with Uvicorn
- **Port**: 8000
- **Features**:
  - RESTful API endpoints
  - File upload and processing
  - LLM query processing
  - Data loading and transformation
  - CSV profiling
  - PDF report generation

**Key Endpoints**:

```
POST /upload
├─ Accepts: CSV file
├─ Returns: File path + data profile
└─ Features: Auto-profiling, caching

POST /chat
├─ Accepts: User prompt + file path
├─ Returns: LLM-generated analysis & visualization
└─ Features: Code validation, safe execution

POST /analyze
├─ Accepts: User prompt + multiple file paths
├─ Returns: Professional summary or visualization
└─ Features: Multi-file context, smart routing

GET /load-data
├─ Accepts: File path (query param)
├─ Returns: CSV as JSON records
└─ Features: Normalization, type conversion

GET /list-files
├─ Returns: All uploaded CSV files
└─ Features: Sorted by modification time

GET /dashboard
└─ Returns: Dashboard HTML interface
```

#### `llm_adapters/llama_cpp_adapter.py` - LLM Interface
- **Model**: Mistral 7B Instruct (GGUF format)
- **Purpose**: Interface with local LLM
- **Features**:
  - Prompt generation with system prompts
  - Token optimization
  - Temperature and sampling controls
  - Streaming support

**Key Methods**:
- `generate()` - Generate text with parameters
- `create_chat()` - Structured chat responses

#### `executors/sandbox_executor.py` - Code Execution
- **Purpose**: Safely execute generated Python and SQL code
- **Safety Features**:
  - Whitelist-based execution
  - Timeout protection
  - Resource limits
  - Error handling and reporting

**Supported Operations**:
- Python: Data analysis, visualization (matplotlib)
- SQL: Database queries (SQLite3)

#### `reports/report_generator.py` - Report Creation
- **Purpose**: Generate PDF reports with charts and tables
- **Features**:
  - Chart embedding
  - Table formatting
  - Professional styling
  - Image conversion

#### `utils/` - Utility Modules

**`data_profiler.py`**:
- Deep statistical profiling of CSV files
- Detects: data types, missing values, outliers, correlations
- Generates summary statistics

**`code_validator.py`**:
- Validates generated Python/SQL code
- Checks for unsafe operations
- Prevents code injection

**`db_connectors.py`**:
- SQLite database connectivity
- Schema introspection
- Query execution

### 3. Data Layer

#### File Structure
```
/data/
├── uploads/           # User-uploaded CSV files
│   ├── {uuid}_{filename}.csv
│   ├── {uuid}_{filename}.csv
│   └── ...
├── reports/           # Generated PDF reports
│   ├── report_{uuid}.pdf
│   └── ...
└── cache/             # Temporary processed data
```

#### Caching Strategy
- In-memory DataFrame cache
- Max 5 files in cache (LRU)
- Automatic invalidation on file updates

### 4. LLM Processing Pipeline

#### System Prompts

**ENHANCED_SYSTEM_PROMPT** (For Analysis/Summaries):
- Instructs model to provide professional 2-4 sentence insights
- Includes metrics, anomalies, business implications
- Used for summary questions: "What's in this data?"

**SYSTEM_PROMPT** (For Code Generation):
- Instructs model to generate one-line Python code
- Includes column normalization rules
- Used for visualization questions: "Show me a chart"

#### Processing Flow

```
User Question
    ↓
[Detect Type: Summary vs Visualization]
    ↓
┌────────────────────┬─────────────────────┐
│                    │                     │
Summary Question     Visualization Question
    ↓                        ↓
Use ENHANCED_PROMPT  Use SYSTEM_PROMPT
    ↓                        ↓
Generate Insights    Generate Python Code
    ↓                        ↓
Return Text         Validate Code
    ↓                        ↓
Display in Chat     Execute Safely
                        ↓
                   Render Chart
                        ↓
                   Return Image
```

## Data Flow Examples

### Example 1: Single File Summary Query

```
1. User: "Summarize this dataset"
   ↓
2. Frontend → POST /analyze with file_path
   ↓
3. Backend:
   - Load CSV (5000 rows max)
   - Normalize columns
   - Build file context
   - Detect: Summary question
   ↓
4. LLM: Use ENHANCED_SYSTEM_PROMPT
   ↓
5. Generate: "Dataset has 5,420 users, 42% aged 25-35..."
   ↓
6. Return: message + summaries + file stats
   ↓
7. Frontend: Display in chat window
```

### Example 2: Multi-File Visualization Query

```
1. User: Selects 3 files, asks "Show comparison chart"
   ↓
2. Frontend → POST /analyze with file_paths[]
   ↓
3. Backend:
   - Load all 3 CSVs
   - Build combined context
   - Detect: Visualization question
   ↓
4. LLM: Use SYSTEM_PROMPT
   ↓
5. Generate: Python one-liner with plt.savefig()
   ↓
6. Validator: Check code safety
   ↓
7. Executor: Run code, capture chart as PNG
   ↓
8. Return: chart_png_base64 + message
   ↓
9. Frontend: Embed image in results
```

### Example 3: Dashboard Live Exploration

```
1. User: Opens /dashboard
   ↓
2. Frontend: Display data explorer
   ↓
3. User: Select file from dropdown
   ↓
4. Frontend: POST /load-data
   ↓
5. Backend: Return full CSV as JSON
   ↓
6. Frontend: Populate table and chart dropdowns
   ↓
7. User: Select chart type + columns
   ↓
8. Frontend: updateChart() with Chart.js
   ↓
9. Display: Interactive visualization
```

## Key Design Decisions

### 1. One-Line Python Code Generation
- **Why**: Ensures deterministic, controllable execution
- **Benefit**: Easier validation and error handling
- **Implementation**: Semicolon-separated statements

### 2. Local LLM (No API Dependency)
- **Why**: Privacy, no network latency, cost-free
- **Benefit**: Works offline, suitable for sensitive data
- **Trade-off**: Requires local GPU or CPU

### 3. Dual System Prompts
- **Why**: Different tasks need different instruction styles
- **Benefit**: Optimized responses for each use case
- **Implementation**: Auto-detection via keyword analysis

### 4. Sandbox Execution
- **Why**: Generated code could be malicious
- **Benefit**: Safety and security for users
- **Trade-off**: Some advanced operations disabled

### 5. In-Memory Caching
- **Why**: Avoid repeated disk I/O
- **Benefit**: Fast repeated queries on same file
- **Trade-off**: Memory usage for large datasets

## Security Considerations

### Code Execution Safety
- Whitelist-based allowed operations
- No file system access beyond data directory
- No network operations
- Timeout protection (30 seconds default)

### File Access Control
- All file paths validated against `/data/uploads/`
- Relative paths converted to absolute
- Path traversal attempts blocked
- UUID-based file naming prevents collisions

### Data Privacy
- No data sent to external APIs
- All processing local
- Optional PDF reports with embedded results
- Session-based file tracking

## Performance Optimization

### Optimizations Implemented
- **Token Limit**: 512 tokens max (faster generation)
- **Temperature**: 0.0-0.1 (deterministic output)
- **Sampling**: Top-P 0.85, Top-K 40 (quality control)
- **Data Limit**: 5000 rows per file (faster processing)
- **Chart Limit**: 50 bars, 20 pie slices, 1000 scatter points
- **Caching**: Last 5 files in memory (LRU)

### Latency Targets
- File upload + profile: < 2 seconds
- Chat query (non-code): 0.5-1 second
- Chat query (with visualization): 2-5 seconds
- Multi-file analysis: 1-2 seconds

## Scalability & Limitations

### Current Limitations
- Single server instance
- Max 5 files in memory simultaneously
- Max ~50k rows per file (performance)
- Local storage only
- Single GPU/CPU resource

### Future Scalability Options
- Add Redis caching layer
- Implement task queue (Celery)
- Add database backend
- Containerize with Docker
- Add multi-GPU support
- Implement API authentication

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Backend** | FastAPI | 0.95+ |
| **ASGI Server** | Uvicorn | 0.21+ |
| **LLM** | llama-cpp-python | 0.2+ |
| **Model** | Mistral 7B (GGUF) | Q4 |
| **Data Processing** | Pandas | 1.5+ |
| **Visualization** | Matplotlib | 3.5+ |
| **Frontend Framework** | Vanilla JS | ES6+ |
| **Chart Library** | Chart.js | 3.9+ |
| **PDF Generation** | ReportLab | 4.0+ |
| **Database** | SQLite3 | 3.0+ |

## Development Workflow

### Adding a New Feature

1. **Backend**:
   - Add endpoint in `fastapi_app.py`
   - Add utility function in `/utils/`
   - Test with curl/Python requests

2. **Frontend**:
   - Add UI in `index.html` or `dashboard.html`
   - Add JavaScript handler
   - Test in browser

3. **Testing**:
   - Validate input/output
   - Check error handling
   - Profile performance

### Common Tasks

**Add new analysis type**:
- Create system prompt variant
- Add question type detection
- Implement result handler

**Add new chart type**:
- Add to Chart.js config in dashboard
- Add to dropdown options
- Implement data preparation

**Add new data source**:
- Create connector in `/utils/`
- Add `/load-source` endpoint
- Add file selection UI

## Monitoring & Debugging

### Logging
- Server logs: stdout/stderr
- Chat logs: Available via `/chat` response
- Error tracking: Detailed traceback in responses

### Performance Monitoring
- Response time tracking (browser)
- LLM generation time (debug logs)
- File processing metrics
- Cache hit rate

### Debugging Tips
- Use `showRaw` toggle for raw LLM output
- Check browser console for JavaScript errors
- Review server logs for backend errors
- Profile with DevTools (Chrome)

## File Organization Best Practices

```
project/
├── README.md              # Quick start guide
├── ARCHITECTURE.md        # This file
├── THEORY.md             # Concepts & theory
├── backend/
│   ├── fastapi_app.py    # Main server
│   ├── llm_adapters/     # LLM integration
│   ├── executors/        # Code execution
│   ├── reports/          # Report generation
│   └── utils/            # Shared utilities
├── static/
│   ├── index.html        # Chat UI
│   ├── dashboard.html    # Explorer UI
│   └── style.css         # Shared styles
├── data/
│   ├── uploads/          # User files
│   ├── reports/          # Generated reports
│   └── cache/            # Temp data
└── models/
    └── *.gguf            # LLM weights
```

---

**Last Updated**: December 2025  
**Architecture Version**: 2.0  
**Model**: Mistral 7B Instruct  
**Python Version**: 3.9+
