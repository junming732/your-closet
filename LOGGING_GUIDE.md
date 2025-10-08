# Logging System Guide

## Overview
The fashion stylist app now has a comprehensive centralized logging system with retry logic for API calls and detailed error tracking.

---

## **Features Implemented**

### ✅ **Centralized Logging**
- **Module:** `src/app/logger_config.py`
- **Log Location:** `logs/fashion_app_YYYYMMDD.log`
- **Console Output:** INFO level and above
- **File Output:** DEBUG level and above (detailed)

### ✅ **Retry Logic**
- **Weather API:** 3 retries with exponential backoff (2s, 4s, 8s)
- **Automatic retry** on server errors (5xx)
- **No retry** on client errors (4xx)

### ✅ **Comprehensive Coverage**
All modules now have logging:
- ✅ Weather API (`weather_utils.py`)
- ✅ Safety filters (`safety_utils.py`)
- ✅ RAG retrieval (`wardrobe_app.py` + `gemini_rag.py`)
- ✅ LLM streaming (`wardrobe_app.py` + `gemini_rag.py`)

---

## **Log Levels**

| Level | Usage | Example |
|-------|-------|---------|
| **DEBUG** | Detailed diagnostics | "Post-filter check on text: 'What should I...'" |
| **INFO** | Normal operations | "Weather API call successful: Location: Stockholm" |
| **WARNING** | Important alerts | "Safety filter triggered [pre-filter]" |
| **ERROR** | Failures | "Weather API call failed: HTTPError 404" |
| **CRITICAL** | System failures | (Reserved for catastrophic errors) |

---

## **Log File Format**

Each log entry includes:
```
2025-10-08 13:24:25 - src.app.safety_utils - WARNING - pre_filter_input:35 - Blocked input containing: 'ignore'
│                     │                       │         │                   │
│                     │                       │         │                   └─ Message
│                     │                       │         └─ Function:Line
│                     │                       └─ Level
│                     └─ Module name
└─ Timestamp
```

---

## **What Gets Logged**

### **1. Weather API Calls**
```
INFO - Calling Weather API - Stockholm with params: {'unitGroup': 'metric'}
INFO - Weather API call successful: Location: Stockholm
INFO - Weather retrieved successfully: 8°C, Partially cloudy
```

**On Failure:**
```
ERROR - Weather API call failed (attempt 1): HTTPError 500 - Server Error
INFO - Retrying in 2s...
ERROR - Weather API call failed (attempt 2): HTTPError 500 - Server Error
INFO - Retrying in 4s...
ERROR - Max retries (3) reached for Weather API
```

### **2. Safety Filter Triggers**
```
WARNING - Safety filter triggered [pre-filter]: Banned word detected: 'ignore' - Input: 'ignore all...'
INFO - Blocked input containing: 'ignore'
```

```
WARNING - Safety filter triggered [post-filter]: Content not fashion-related - Input: 'Tell me about cars'
WARNING - Post-filter blocked non-fashion content
```

```
WARNING - Safety filter triggered [gemini-safety]: Category: HARM_CATEGORY_DANGEROUS_CONTENT, Probability: MEDIUM
WARNING - Gemini safety filter triggered: HARM_CATEGORY_DANGEROUS_CONTENT (MEDIUM)
```

### **3. RAG Retrieval**
```
INFO - Retrieving RAG docs for query: 'Casual Spring Blue Striped Solid...'
INFO - Retrieved 3 documents successfully
INFO - RAG retrieval successful - Query: 'Casual Spring Blue...' - Docs: 3
```

**On Failure:**
```
ERROR - RAG retrieval failed: ConnectionError - Connection timeout
INFO - RAG retrieval failed - Query: 'Casual Spring...' - Docs: 0
```

### **4. LLM Streaming**
```
INFO - Calling Gemini API - generate_outfit_advice with params: {'temperature': 0.7}
INFO - Gemini API call successful: Generated 45 chunks
```

**On Failure:**
```
ERROR - Gemini API call failed: APIError - Rate limit exceeded
ERROR - Outfit generation streaming failed: APIError - Rate limit exceeded
```

---

## **How to Use**

### **View Logs in Real-time**
```bash
# Watch console output when running app
python src/app/main.py

# Or tail the log file
tail -f logs/fashion_app_$(date +%Y%m%d).log
```

### **Search Logs**
```bash
# Find all errors
grep "ERROR" logs/fashion_app_*.log

# Find weather API issues
grep "Weather API" logs/fashion_app_*.log

# Find safety filter triggers
grep "Safety filter" logs/fashion_app_*.log

# Find RAG failures
grep "RAG retrieval failed" logs/fashion_app_*.log
```

### **Analyze Log File**
```bash
# Count errors by type
grep "ERROR" logs/fashion_app_*.log | cut -d'-' -f5 | sort | uniq -c

# View last 50 lines
tail -n 50 logs/fashion_app_*.log

# View today's errors only
grep "ERROR" logs/fashion_app_$(date +%Y%m%d).log
```

---

## **Troubleshooting**

### **Problem: No log file created**
**Solution:**
```bash
# Manually create logs directory
mkdir -p logs
```

### **Problem: Logs too large**
**Solution:**
```bash
# Delete old logs (keep last 7 days)
find logs/ -name "*.log" -mtime +7 -delete

# Or compress old logs
gzip logs/fashion_app_$(date -d '7 days ago' +%Y%m%d).log
```

### **Problem: Can't find specific error**
**Solution:**
```bash
# Use verbose grep
grep -C 3 "error message" logs/fashion_app_*.log  # Shows 3 lines before/after
```

---

## **Configuration**

### **Change Log Level**
Edit `src/app/logger_config.py`:
```python
# For more verbose console output
console_handler.setLevel(logging.DEBUG)  # Shows everything

# For less verbose file logging
file_handler.setLevel(logging.INFO)  # Skip DEBUG messages
```

### **Disable Console Logging**
```python
# In logger_config.py, comment out:
# logger.addHandler(console_handler)
```

### **Change Log File Location**
```python
# In logger_config.py
LOGS_DIR = Path("/custom/path/to/logs")
```

---

## **Retry Logic Details**

### **Weather API Retries**
```python
max_retries = 3  # Default
timeout = 10     # 10 seconds per attempt

# Exponential backoff:
# Attempt 1: Immediate
# Attempt 2: Wait 2s (2^1)
# Attempt 3: Wait 4s (2^2)
# Attempt 4: Wait 8s (2^3) - but won't happen with max_retries=3
```

### **When Retries Happen**
✅ **Retry:** Server errors (500-599), Network timeouts, Connection errors
❌ **No Retry:** Client errors (400-499), Invalid API key, Invalid location

---

## **Best Practices**

1. **Check logs daily** during development
2. **Monitor ERROR level** messages in production
3. **Archive old logs** weekly to save space
4. **Use log analysis** to identify patterns
5. **Set up alerts** for repeated errors (external tool)

---

## **Example: Debugging a Failed Outfit Generation**

1. **Check the log file:**
   ```bash
   tail -n 100 logs/fashion_app_$(date +%Y%m%d).log
   ```

2. **Look for the sequence:**
   ```
   INFO - Retrieving RAG docs...          ← RAG started
   INFO - Retrieved 3 documents...        ← RAG succeeded
   INFO - Calling Gemini API...           ← LLM call started
   ERROR - Gemini API call failed...      ← Found the problem!
   ERROR - Outfit generation failed...    ← Resulted in user error
   ```

3. **Identify the root cause** from the error message

4. **Fix and verify** by checking logs again

---

## **Integration with Monitoring Tools**

### **Splunk / ELK / Datadog**
The log format is compatible with standard log aggregation tools:
```
timestamp - module - level - function:line - message
```

### **Alert on Errors**
Set up alerts for:
- 5+ API failures in 10 minutes
- Safety filter triggers (potential attack)
- RAG retrieval failures (knowledge base issue)

---

## **Summary**

✅ **Comprehensive logging** across all modules
✅ **Retry logic** for transient failures
✅ **Detailed error messages** for debugging
✅ **Structured format** for easy parsing
✅ **Safety filter tracking** for security monitoring
✅ **API call logging** for performance analysis

**Log file location:** `logs/fashion_app_YYYYMMDD.log`
**Test logging:** `python test_logging.py`
