# Agentic Assistant - Architecture Documentation

## System Overview

The Agentic Assistant is an intelligent query system that combines **SQL analytics** with **RAG (Retrieval-Augmented Generation)** to answer questions about automotive sales data and documentation.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              STREAMLIT UI                                       │
│                         (streamlit_app.py)                                      │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────────────────┐ │
│  │ Example Buttons │  │   Text Input     │  │     Results Display             │ │
│  │                 │  │   + Search       │  │  - Query, Answer, Citations     │ │
│  │                 │  │                  │  │  - SQL Query, Execution Trace   │ │
│  └────────┬────────┘  └────────┬─────────┘  └─────────────────────────────────┘ │
└───────────┼────────────────────┼────────────────────────────────────────────────┘
            │                    │
            └────────┬───────────┘
                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         AGENT ORCHESTRATOR                                      │
│                            (agent.py)                                           │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                         QUERY ROUTER (LLM)                              │    │
│  │          Analyzes question → Returns: SQL | RAG | HYBRID                │    │
│  └───────────────────────────────┬─────────────────────────────────────────┘    │
│                                  │                                              │
│           ┌──────────────────────┼──────────────────────┐                       │
│           ▼                      ▼                      ▼                       │
│  ┌────────────────┐    ┌────────────────┐    ┌──────────────────────┐           │
│  │   SQL PATH     │    │   RAG PATH     │    │    HYBRID PATH       │           │
│  └───────┬────────┘    └───────┬────────┘    └──────────┬───────────┘           │
│          │                     │                        │                       │
└──────────┼─────────────────────┼────────────────────────┼───────────────────────┘
           │                     │                        │
           ▼                     ▼                        ▼
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────────────┐
│     SQL TOOL         │ │     RAG TOOL         │ │      HYBRID FLOW             │
│    (agent.py)        │ │    (rag.py)          │ │                              │
│                      │ │                      │ │  1. SQL Tool (full flow)     │
│ 1. SQL Generator     │ │ 1. Doc Retriever     │ │  2. RAG Tool (full flow)     │
│    (LLM)             │ │    (Embedding)       │ │  3. Hybrid Combiner (LLM)    │
│ 2. SQL Executor      │ │ 2. Answer Generator  │ │                              │
│    (DuckDB)          │ │    (LLM)             │ └──────────────────────────────┘
│ 3. Result Formatter  │ │                      │
│    (LLM)             │ │                      │
└──────────┬───────────┘ └──────────┬───────────┘
           │                        │
           ▼                        ▼
┌───────────────────────┐ ┌──────────────────────────────────────────────────────┐
│      DuckDB           │ │                    FAISS INDEX                       │
│   (automotive.db)     │ │                  (vector_index/)                     │
│                       │ │                                                      │
│ - DIM_MODEL           │ │  ┌─────────────────┐    ┌─────────────────────────┐  │
│ - DIM_COUNTRY         │ │  │  faiss.index    │    │    documents.pkl        │  │
│ - DIM_ORDERTYPE       │ │  │  (vectors)      │    │    (metadata)           │  │
│ - FACT_SALES          │ │  └─────────────────┘    └─────────────────────────┘  │
│ - FACT_SALES_ORDERTYPE│ │                                                      │
└───────────────────────┘ └──────────────────────────────────────────────────────┘
           ▲                        ▲
           │                        │
┌──────────┴───────────┐ ┌──────────┴───────────────────────────────────────────┐
│    CSV FILES         │ │              PDF DOCUMENTS                            │
│  (data/data/)        │ │                                                       │
│                      │ │  ┌─────────────────┐    ┌─────────────────────────┐   │
│ - DIM_MODEL.csv      │ │  │  data/docs/     │    │    data/manuals/        │   │
│ - DIM_COUNTRY.csv    │ │  │  (Contracts)    │    │    (Owner's Manuals)    │   │
│ - DIM_ORDERTYPE.csv  │ │  └─────────────────┘    └─────────────────────────┘   │
│ - FACT_SALES.csv     │ │                                ▲                      │
│ - FACT_SALES_OT.csv  │ │                                │                      │
└──────────────────────┘ │                         ┌──────┴──────────┐           │
                         │                         │  Web Scraper    │           │
                         │                         │  (scraper.py)   │           │
                         │                         │  Toyota Portal  │           │
                         │                         └─────────────────┘           │
                         └───────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           EXTERNAL SERVICES                                     │
│                                                                                 │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────────────┐ │
│  │      OpenAI API             │    │    SentenceTransformers                 | |
│  │                             │    │                                         │ │
│  │  - Query Routing            │    │  - Document Embedding                   │ │
│  │  - SQL Generation           │    │  - Query Embedding                      │ │
│  │  - Result Formatting        │    │  - Similarity Search                    │ │
│  │  - RAG Answer Generation    │    │                                         │ │
│  │  - Hybrid Combining         │    │                                         │ │
│  └─────────────────────────────┘    └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Descriptions

### 1. Streamlit UI (`streamlit_app.py`)

**Purpose:** Web-based user interface for interacting with the assistant.

---

### 2. Agent Orchestrator (`agent.py`)

**Purpose:** Central coordinator that routes queries to appropriate tools and manages execution flow.

**How it works:**
1. Receives user question
2. Uses LLM to classify query type (SQL, RAG, or HYBRID)
3. Dispatches to appropriate handler(s)
4. Collects results and builds execution trace
5. Returns unified `QueryResult` with answer, citations, timing

**Classes:**
- `AgentOrchestrator`: Main orchestration logic
- `SQLTool`: SQL generation and execution
- `QueryResult`: Standardized response container
- `AgentExecutionError`: Error with partial trace

---

### 3. SQL Tool (`agent.py` - `SQLTool` class)

**Purpose:** Converts natural language to SQL and executes against DuckDB.

**How it works:**
1. **SQL Generator (LLM):** Takes question + schema → generates SQL query
2. **SQL Executor (DuckDB):** Runs query against in-memory database
3. **Result Formatter (LLM):** Converts tabular results to natural language

**Database Schema:**
- `DIM_MODEL`: Vehicle models (brand, segment, powertrain)
- `DIM_COUNTRY`: Countries and regions
- `DIM_ORDERTYPE`: Order types (Private, Fleet, Demo)
- `FACT_SALES`: Monthly sales by model and country
- `FACT_SALES_ORDERTYPE`: Sales with order type breakdown

---

### 4. RAG Tool (`rag.py`)

**Purpose:** Retrieves relevant document chunks and generates answers from unstructured content.

**How it works:**
1. **Document Ingestion:**
   - Loads PDFs (contracts, owner's manuals)
   - Chunks text using `RecursiveCharacterTextSplitter` (512 tokens, 128 overlap)
   - Generates embeddings using SentenceTransformers
   - Stores in FAISS index with cosine similarity

2. **Retrieval:**
   - Embeds user query using `google/embeddinggemma-300m`
   - Searches FAISS index for top-k similar chunks
   - Returns documents with similarity scores

3. **Answer Generation:**
   - Formats retrieved chunks as context
   - Uses LLM to synthesize answer with citations

---

### 5. Web Scraper (`scraper.py`)

**Purpose:** Automatically downloads owner's manuals from Toyota Europe portal.

**How it works:**
1. Uses Selenium WebDriver with headless Chrome
2. Navigates Toyota customer portal
3. Selects model, generation, and language
4. Downloads PDF manuals to `data/manuals/`

---

### 6. Configuration (`config.py`)

**Purpose:** Centralized configuration for all system parameters.

**Key Settings:**
| Category | Parameters |
|----------|------------|
| Paths | `BASE_DIR`, `DATA_DIR`, `DOCS_DIR`, `MANUALS_DIR`, `VECTOR_INDEX_DIR` |
| API | `OPENAI_API_KEY`, `OPENAI_MODEL` |
| Embeddings | `EMBEDDING_MODEL`, `TIKTOKEN_MODEL` |
| RAG | `CHUNK_SIZE=512`, `CHUNK_OVERLAP=128`, `DEFAULT_TOP_K=5` |
| Database | `DATABASE_PATH` |
| Control | `RELOAD_DATA` (rebuild index/database on startup) |

---

### 7. Prompts (`prompts.py`)

**Purpose:** Centralized storage for all LLM prompt templates.

**Prompt Types:**
- `SQL_GENERATION_PROMPT`: Schema + question → SQL
- `SQL_FORMAT_RESULTS_PROMPT`: Results → natural language
- `ROUTING_PROMPT`: Question → SQL/RAG/HYBRID classification
- `RAG_ANSWER_PROMPT`: Context + question → answer
- `HYBRID_COMBINE_PROMPT`: Merge SQL and RAG answers

---

### 8. Utilities (`utils.py`)

**Purpose:** Shared helper functions.

**Key Function:**
- `call_llm()`: Unified wrapper for OpenAI Responses API with configurable verbosity and reasoning effort
- `LLMResponse`: Dataclass for token tracking and cost calculation

---

### 9. Guardrails (`guardrails.py`)

**Purpose:** Security and safety checks to prevent unsafe interactions.

**How it works:**
1. **Input Guardrails:** Validates user input before processing
   - Blocks prompt injection attempts
   - Detects harmful content (violence, hate speech, explicit)
   - Filters PII requests
   - Identifies off-topic queries

2. **SQL Guardrails:** Validates generated SQL before execution
   - Blocks dangerous operations (DROP, DELETE, UPDATE, INSERT, TRUNCATE)
   - Prevents system table access
   - Detects SQL injection patterns
   - Enforces table allowlist

3. **Output Guardrails:** Validates LLM responses before returning
   - Checks for harmful content in responses
   - Detects PII leakage

**Key Classes:**
- `GuardrailViolation`: Enum of violation types
- `GuardrailResult`: Result container with pass/fail and details
- `GuardrailError`: Exception for blocked requests

**Violation Types:**
| Type | Description |
|------|-------------|
| `SQL_INJECTION` | SQL injection patterns detected |
| `SQL_DANGEROUS_OPERATION` | Non-SELECT operations |
| `OFF_TOPIC` | Questions outside automotive domain |
| `HARMFUL_CONTENT` | Violence, hate speech, explicit content |
| `PROMPT_INJECTION` | Attempts to override instructions |
| `PII_REQUEST` | Requests for personal information |

---

## Key Trade-offs

### Latency

I explored different approaches to balance speed and accuracy using OpenAI's latest models.

#### Model Comparison

I tested two main models:
- **`gpt-5.1-2025-11-13`**: Newest model with the `responses` API, allowing customizable thinking budget (reasoning effort) and verbosity level
- **`gpt-4.1-2025-04-14`**: Previous generation, no reasoning controls but faster

#### Reasoning Effort Impact (gpt-5.1)

The reasoning effort parameter significantly affects latency on complex queries (e.g., hybrid queries):

| Reasoning Level | Elapsed Time | Notes |
|-----------------|--------------|-------|
| `high` | ~140s | Impossible to use in interactive applications |
| `medium` | ~90s | Too slow for interactive applications |
| `low` | ~50s | Better, but still too slow  |
| `None` | ~35s | Acceptable, borderline latency |

#### Cross-Model Comparison

| Model | Hybrid Query Time | Relative Speed |
|-------|-------------------|----------------|
| gpt-5.1 (reasoning: None) | ~35s | Baseline |
| gpt-4.1 | ~23s | **50% faster** |

**Key Finding:** `gpt-4.1` is approximately **50% faster** than `gpt-5.1` on complex queries, making it the better choice for interactive applications. The quality is also not significantly different for this use case.

#### Mixed Reasoning Strategy (Failed Experiment)

I attempted to optimize by using different reasoning levels for different steps:
- **Low effort** for routing/retrieval
- **Medium effort** for SQL generation  
- **High effort** for combining results

**Result:** This approach still resulted in excessive latency and was abandoned in favor of using `gpt-4.1` consistently.

#### gpt-5.1-mini (Failed Experiment)

I also tested `gpt-5.1-mini`, hoping it would be:
- Fast enough for interactive use
- Cheap while still having reasoning capabilities

**Result:** Poor quality on both intermediate steps (SQL generation) and final answers. Not suitable for this use case.

#### Component Latency Breakdown

| Component | Typical Latency | Notes |
|-----------|-----------------|-------|
| Query Router (LLM) | 400-600ms | First LLM call, minimal tokens |
| SQL Generator (LLM) | 600-1200ms | Depends on query complexity |
| SQL Executor (DuckDB) | 5-50ms | In-memory, very fast |
| Document Retriever | 50-150ms | Local embedding + FAISS |
| Answer Generator (LLM) | 800-2000ms | Largest token output |
| **Total SQL Query** | **1.5-3s** | 3 LLM calls |
| **Total RAG Query** | **1.5-2.5s** | 2 LLM calls + embedding |
| **Total Hybrid Query** | **3-5s** | 5 LLM calls + embedding |

**Optimization Options:**
- Use smaller/faster models for routing
- Cache common queries
- Parallel SQL and RAG execution in hybrid mode
- Stream LLM responses to UI

---

### Cost

#### Token Pricing Comparison

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| gpt-5.1-2025-11-13 | $1.25 | $10.00 |
| gpt-4.1-2025-04-14 | $2.00 | $8.00 |

**Analysis:** While `gpt-4.1` is slightly more expensive on input tokens, it's **20% cheaper on output tokens**. Since LLM responses (output) typically dominate the token count, `gpt-4.1` ends up being more cost-effective overall.

#### Cost Per Query

| Query Type | Estimated Cost | Notes |
|------------|----------------|-------|
| SQL Query | ~$0.01 | 3 LLM calls |
| RAG Query | ~$0.015 | 2 LLM calls + larger context |
| Hybrid Query | ~$0.02 | 5 LLM calls |

#### Component Cost Breakdown

| Component | Cost Driver | Estimate per Query |
|-----------|-------------|-------------------|
| OpenAI API | Input/output tokens | $0.01-0.02 |
| Embedding Model | Local (free) | $0.00 |
| DuckDB | Local (free) | $0.00 |
| FAISS | Local (free) | $0.00 |

---

### Security

For safety, I employed **LLM-based guardrails** to prevent:
- **Harmful or malicious queries** (violence, hate speech, illegal content)
- **SQL injection attacks** (DROP, DELETE, UPDATE, unauthorized table access)
- **Off-topic queries** (recipes, weather, medical advice)
- **Prompt injection attempts** (attempts to override system instructions)
- **PII requests** (social security numbers, credit cards, passwords)


**Security Flow:**
```
User Input → Input Guardrail → Query Router → SQL Guardrail → Execute → Output Guardrail → Response
                    │                              │                            │
                    ▼                              ▼                            ▼
              Block if unsafe              Block if dangerous            Block if harmful
```

---


### How to use in Production Systems
- Use embedding models deployed on GPU instances for speed (and better quality, as GPUs can handle bigger embedding models)
- Cache SQL for repeated queries
- Better prompt engineering to reduce token usage and improve accuracy
- Use frameworks like LangChain for more flavorful agent management and features
- Use observability frameworks like Langfuse for tracing and monitoring LLM calls + prompt management (versioning)
- Implement data update pipelines to keep the vector index and database fresh (remove old/stale data, add new data)
- For scraping in particular, the ideal scenario would be a data partneship with Toyota to get direct access to the manuals and documentation, instead of relying on web scraping which can break if the website changes


---

## Data Flow Summary

```
                    ┌─────────────────────────────────────────┐
                    │              USER QUERY                 │
                    └─────────────────┬───────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────┐
                    │          INPUT GUARDRAIL (LLM)          │
                    │  Checks: prompt injection, harmful,     │
                    │          PII requests, off-topic        │
                    └─────────────────┬───────────────────────┘
                                      │
                         ┌────────────┴─────────────┐
                         │                          │
                   ✅ Allowed                  ❌ Blocked
                         │                          │
                         ▼                          ▼
┌──────────────────────────────────────────┐   ┌─────────────────┐
│            CLASSIFICATION                │   │  Safe Response  │
│                                          │   │  "I can only    │
│   "sales", "units"  →  SQL               │   │   help with..." │
│   "warranty", "manual"  →  RAG           │   └─────────────────┘
│   "compare sales and warranty"  →  HYBRID│
└─────────────────┬────────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌───────┐    ┌───────┐    ┌─────────┐
│  SQL  │    │  RAG  │    │ HYBRID  │
└───┬───┘    └───┬───┘    └────┬────┘
    │            │             │
    ▼            │             │
┌────────────────────┐         │
│  SQL GUARDRAIL     │◄────────┘
│  (LLM)             │
│  Validates query   │
│  before execution  │
└─────────┬──────────┘
          │
    ┌─────┴─────┐
    │           │
✅ Safe    ❌ Dangerous
    │           │
    ▼           ▼
┌───────┐  ┌─────────────┐
│DuckDB │  │Safe Response│
│Tables │  │"Read-only   │
└───┬───┘  │ operations" │
    │      └─────────────┘
    │            │
    └─────┬──────┘
          │
          ▼
    ┌───────────┐
    │   FAISS   │
    │   Index   │
    └─────┬─────┘
          │
          ▼
┌─────────────────────────────────────────┐
│         OUTPUT GUARDRAIL (LLM)          │
│   Checks: harmful content, PII leaks    │
└─────────────────┬───────────────────────┘
                  │
     ┌────────────┴────────────┐
     │                         │
✅ Safe                   ❌ Harmful
     │                         │
     ▼                         ▼
┌─────────────────────┐   ┌─────────────────┐
│  FORMATTED ANSWER   │   │  Safe Response  │
│  + Citations        │   │  "Response      │
│  + SQL Query        │   │   filtered"     │
│  + Trace            │   └─────────────────┘
└─────────────────────┘
```
