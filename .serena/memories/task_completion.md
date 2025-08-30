# Task Completion Guidelines for MLX-RAG

## After Completing Code Changes

### 1. Testing
```bash
# Run relevant tests
uv run pytest tests/test_specific_feature.py

# Run full test suite for major changes
uv run pytest
```

### 2. Code Quality
```bash
# Python formatting (if available)
black src/ tests/

# Linting (if configured)
flake8 src/ tests/
```

### 3. Verification Steps
- **Server functionality**: Start the server and verify endpoints work
- **UI functionality**: Test the web admin interface at localhost:8000/admin
- **Model operations**: Test model loading, inference, and unloading
- **API compatibility**: Verify OpenAI-compatible endpoints work

### 4. Documentation Updates
- Update README.md if new features added
- Update API.md for new endpoints
- Add inline code documentation for complex functions

### 5. Build Testing (for major changes)
```bash
# Test standalone app build
uv run ./scripts/build_app.sh

# Verify the built app runs correctly
./dist/MLX-RAG.app/Contents/MacOS/MLX-RAG
```

## Specific Areas to Verify
- **Memory Management**: Check model loading/unloading works correctly
- **UI Responsiveness**: Verify admin interface updates properly
- **Error Handling**: Test error conditions and user feedback
- **Platform Compatibility**: Ensure Apple Silicon requirements are met