# MLX-RAG Debugging and Logging Guide

## Overview
This guide documents effective debugging strategies and logging patterns discovered during development and troubleshooting of the MLX-RAG application.

## Common Debugging Scenarios

### 1. Request Flow Debugging
**Problem**: Frontend appears to send requests but backend doesn't show any activity.

**Root Cause**: Often logging configuration issues where logs are suppressed or not visible in terminal output.

**Solution Strategy**:
1. **Add explicit print statements** at the start of endpoint functions to guarantee visibility
2. **Add request middleware** to log ALL incoming HTTP requests
3. **Use both logger and print** - logger for production, print for guaranteed debugging visibility

**Code Pattern**:
```python
# Temporary debugging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    print(f"\n>>> REQUEST: {request.method} {request.url}")
    print(f">>> Headers: {dict(request.headers)}")
    response = await call_next(request)
    print(f">>> RESPONSE: {response.status_code}")
    return response

# Explicit endpoint debugging
@app.post("/v1/chat")
async def chat(message: str = Form(...), model: str = Form(...), ...):
    # Guaranteed visibility debugging
    print(f"ENDPOINT HIT - Message: {message!r}, Model: {model!r}")
    logger.info(f"Chat request received - model: {model}, rag_collection: {rag_collection}")
    # ... rest of endpoint logic
```

### 2. Logging Configuration Issues
**Problem**: Python logging sometimes doesn't appear in terminal output depending on configuration.

**Key Lessons**:
- **print() statements always work** regardless of logging configuration
- **logger.error()** has higher priority than logger.info() or logger.debug()
- **Use logger.error() for temporary debugging** to ensure visibility
- **Always clean up debug logging** after issue resolution

**Best Practices**:
- Start with explicit print statements for critical debugging
- Use structured logging with appropriate levels for production
- Add middleware logging for request flow debugging
- Remove temporary debug logging once issues are resolved

### 3. Request Validation
**Problem**: Requests might not be reaching the expected endpoint or might be malformed.

**Debugging Steps**:
1. Add middleware to log ALL incoming requests
2. Check request method, URL, and headers
3. Verify response status codes
4. Add explicit prints at endpoint entry points
5. Log request body/form data for POST requests

### 4. Frontend-Backend Communication
**Problem**: Uncertain if frontend is sending requests correctly or if backend is receiving them.

**Validation Strategy**:
1. **Frontend side**: Add console.log statements in JavaScript
2. **Backend side**: Add print statements and middleware logging
3. **Network level**: Check browser developer tools Network tab
4. **Response validation**: Verify response status codes and content

## Debugging Patterns

### Pattern 1: Request Flow Tracing
```python
# 1. Add middleware for global request logging
@app.middleware("http")
async def debug_requests(request, call_next):
    print(f">>> {request.method} {request.url}")
    response = await call_next(request)
    print(f">>> Response: {response.status_code}")
    return response

# 2. Add endpoint entry logging
@app.post("/endpoint")
async def endpoint_function(...):
    print(f"ENDPOINT HIT: {locals()}")  # Print all parameters
    # ... endpoint logic
```

### Pattern 2: Layered Logging Approach
```python
# Layer 1: Guaranteed visibility (temporary)
print(f"DEBUG: Critical info here")

# Layer 2: High-priority logging (temporary debugging)
logger.error(f"DEBUG: Detailed info here")

# Layer 3: Normal logging (production)
logger.info(f"Normal operation info")
logger.debug(f"Detailed debug info")
```

### Pattern 3: Data Validation Logging
```python
# Log received data for validation
logger.info(f"Received: param1={param1!r}, param2={param2!r}")

# Log processed data
logger.debug(f"Processed: {len(processed_data)} items")

# Log validation results
if validation_failed:
    logger.error(f"Validation failed: {error_details}")
```

## Cleanup Checklist

After debugging issues, ensure:
- [ ] Remove excessive print statements
- [ ] Remove temporary middleware logging
- [ ] Convert logger.error() back to appropriate levels
- [ ] Keep essential logging for production monitoring
- [ ] Document the root cause and solution

## Production Logging Best Practices

1. **Use appropriate log levels**:
   - `ERROR`: Actual errors that need attention
   - `WARNING`: Potential issues or fallbacks
   - `INFO`: Important operational information
   - `DEBUG`: Detailed information for debugging

2. **Include context in log messages**:
   - Request IDs, user IDs, model names
   - Relevant parameters and their values
   - Timing information for performance monitoring

3. **Structure log messages consistently**:
   ```python
   logger.info(f"Operation completed - model: {model}, duration: {duration}s, status: {status}")
   ```

4. **Avoid logging sensitive information**:
   - API keys, passwords, tokens
   - Personal user data
   - Full request/response bodies in production

## Tools and Techniques

### FastAPI Request Middleware
Useful for debugging request flow issues:
```python
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"{request.method} {request.url} - {response.status_code} - {duration:.3f}s")
    return response
```

### Exception Handling with Context
```python
try:
    # risky operation
    result = process_data(data)
except Exception as e:
    logger.error(f"Processing failed for {context_info}: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
```

### Debugging Complex Data Flows
```python
# Log data at each transformation step
logger.debug(f"Input data: {len(input_data)} items")
processed = transform_data(input_data)
logger.debug(f"After transform: {len(processed)} items")
validated = validate_data(processed)
logger.debug(f"After validation: {len(validated)} items, {len(processed) - len(validated)} filtered")
```

## Historical Issues Resolved

### Issue: Frontend/Backend Communication Failure
**Date**: August 31, 2025
**Symptoms**: Frontend sending chat requests, no backend activity visible
**Root Cause**: Logging configuration suppressing output in terminal
**Solution**: Added explicit print statements and request middleware
**Prevention**: Always use layered logging approach (print + logger) for critical debugging

### Issue: Request Data Not Visible
**Symptoms**: Endpoints being hit but request parameters not visible in logs
**Root Cause**: Using logger.info() which was configured to be suppressed
**Solution**: Used logger.error() and print() for guaranteed visibility
**Cleanup**: Converted back to appropriate log levels after debugging

## Future Improvements

1. **Structured Logging**: Consider using structured logging (JSON format) for better parsing
2. **Request Tracing**: Add request ID tracing across the entire request lifecycle
3. **Performance Monitoring**: Add timing logs for critical operations
4. **Health Check Logging**: Regular health check logs to verify system state
5. **Error Aggregation**: Consider error tracking services for production deployments

## Remember
- **Debugging is temporary** - always clean up after resolving issues
- **Document solutions** - record what worked for future reference
- **Test logging configuration** - verify logs appear where expected
- **Use multiple approaches** - combine print, logger, and middleware for thorough debugging
