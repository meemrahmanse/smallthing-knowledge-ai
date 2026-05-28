# Backend Refactoring - Piccoli Studios AI v3.4

## Struttura Completata

```
backend/
├── __init__.py
├── config.py              # Configurazione centralizzata
├── logger.py              # Logging rotante
├── utils.py               # QueryNormalizer, SecurityManager, RateLimiter, LRUCache
├── extractors/
│   ├── __init__.py
│   └── pdf.py             # PDFExtractor (estrazione PDF)
├── rag/
│   ├── __init__.py
│   ├── chunker.py         # TableAwareChunker (text chunking)
│   └── keywords.py        # KeywordIndex (keyword search)
├── ml/
│   └── __init__.py
├── session/
│   └── __init__.py
├── api/
│   └── __init__.py
└── llm/
    └── __init__.py
```

## Prossimi Passi

1. **RAG Engine** (`backend/rag/engine.py`)
   - RAGEngine per retrieval
   - EmbeddingManager per embeddings
   - SemanticReranker per reranking

2. **ML Modules** (`backend/ml/`)
   - SemanticClassifier
   - MLManager

3. **Session Manager** (`backend/session/manager.py`)
   - ChatSession, ChatMessage
   - SessionManager
   - ConversationLogger

4. **LLM Clients** (`backend/llm/client.py`)
   - OllamaClient
   - ClaudeClient

5. **API Layer** (`backend/api/`)
   - models.py: Pydantic models
   - routes.py: FastAPI endpoints
   - dependencies.py: Dipendenze (verify_rate_limit, verify_api_key, ecc.)

6. **App principale** (nuovo `app.py`)
   - Importa tutti i moduli
   - Configura FastAPI
   - Monta file statici
   - Definisce lifespan

## Benefici del Refactoring

✅ **Manutenibilità**: Separazione delle responsabilità
✅ **Testing**: Moduli isolati e testabili
✅ **Scalabilità**: Facile aggiungere nuove feature
✅ **Chiarezza**: Codice organizzato e navigabile
