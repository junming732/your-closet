# Logging Guide

## Log Levels

The application uses 4 log levels:

| Level | When Used | Example |
|-------|-----------|---------|
| DEBUG | Detailed diagnostic info | "Pre-filter passed", "Retrieved chunk from page 5" |
| INFO | Normal operations | "Query intent: knowledge", "Weather retrieved successfully" |
| WARNING | Something unexpected but not critical | "Temperature not found, trying fallback" |
| ERROR | Something failed | "RAG retrieval failed", "Outfit generation failed" |

---

## Where Logs Appear

### Console (Terminal Output)
Shows: INFO, WARNING, ERROR only
Does NOT show: DEBUG logs

### Log File
Location: `logs/fashion_app_YYYYMMDD.log`
Shows: Everything (DEBUG, INFO, WARNING, ERROR)

---

## Log File Format

### Console Format (Simple)
```
LEVEL - module.name - message
```

Example:
```
INFO - src.app.wardrobe_app - Query intent: knowledge
ERROR - src.retrieval.gemini_rag - Document retrieval failed
```

### File Format (Detailed)
```
YYYY-MM-DD HH:MM:SS - module.name - LEVEL - function:line - message
```

Example:
```
2025-10-13 14:23:45 - src.app.wardrobe_app - INFO - chat_response:508 - Query intent: knowledge
2025-10-13 14:23:46 - src.retrieval.gemini_rag - ERROR - retrieve_docs:139 - Document retrieval failed
```

---

## What Gets Logged

### Tab 2 (Build Outfit)
| What | Level | Console | File |
|------|-------|---------|------|
| Starting RAG query | INFO | Yes | Yes |
| RAG retrieval failed | ERROR | Yes | Yes |
| RAG retrieval success | INFO | Yes | Yes |
| Outfit generation failed | ERROR | Yes | Yes |

### Tab 3 (Chat with Stylist)
| What | Level | Console | File |
|------|-------|---------|------|
| Query intent detected | INFO | Yes | Yes |
| Starting RAG query | INFO | Yes | Yes |
| RAG retrieval success | INFO | Yes | Yes |
| RAG retrieval failed | ERROR | Yes | Yes |
| Query classification failed | ERROR | Yes | Yes |
| Chat streaming failed | ERROR | Yes | Yes |

### RAG System
| What | Level | Console | File |
|------|-------|---------|------|
| Index loaded | INFO | Yes | Yes |
| Index created | INFO | Yes | Yes |
| Retrieving documents | DEBUG | No | Yes |
| Retrieved chunk details | DEBUG | No | Yes |
| Document retrieval failed | ERROR | Yes | Yes |
| Generation failed | ERROR | Yes | Yes |

### Safety Filters
| What | Level | Console | File |
|------|-------|---------|------|
| Blocked malicious input | INFO | Yes | Yes |
| Pre-filter passed | DEBUG | No | Yes |
| Running post-filter check | DEBUG | No | Yes |
| Post-filter result | DEBUG | No | Yes |
| Post-filter blocked content | WARNING | Yes | Yes |
| Post-filter check failed | ERROR | Yes | Yes |
| Gemini safety filter triggered | WARNING | Yes | Yes |
| Safety ratings passed | DEBUG | No | Yes |

### Weather API
| What | Level | Console | File |
|------|-------|---------|------|
| Missing API key | ERROR | Yes | Yes |
| Location required | WARNING | Yes | Yes |
| Weather retrieved successfully | INFO | Yes | Yes |
| Retrying API call | INFO | Yes | Yes |
| Max retries reached | ERROR | Yes | Yes |
| Temperature not found | WARNING | Yes | Yes |
| Temperature found in fallback | INFO | Yes | Yes |
| Fallback failed | ERROR | Yes | Yes |

