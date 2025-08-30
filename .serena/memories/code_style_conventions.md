# Code Style and Conventions for MLX-RAG

## Python Code Style
- **Indentation**: 4 spaces (standard Python)
- **Type Hints**: Used throughout codebase
- **Docstrings**: Google-style docstrings for functions and classes
- **Imports**: Organized with standard library first, then third-party, then local
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Line Length**: ~88-100 characters (follows Black formatter style)

## Frontend Code Style
- **HTML**: Semantic markup with Tailwind CSS classes
- **CSS**: Tailwind utility-first approach with minimal custom CSS
- **JavaScript**: ES6+ features, camelCase naming
- **Indentation**: 4 spaces for consistency with Python

## File Organization
- **Source code**: `src/mlx_rag/` directory
- **Templates**: `src/mlx_rag/templates/`
- **Tests**: `tests/` directory
- **Scripts**: `scripts/` directory for build scripts
- **Config**: `pyproject.toml` for Python dependencies and metadata

## Key Patterns
- **API Endpoints**: FastAPI with async/await
- **Error Handling**: Try-catch with proper logging
- **Model Management**: Queue-based system with status tracking
- **Configuration**: Environment variables and settings API
- **Database**: SQLAlchemy models with database.py abstraction

## Dependencies Management
- **Primary**: Use `uv` for faster dependency resolution
- **Fallback**: pip as alternative
- **Optional extras**: Different feature sets (audio, vision, rag-full, dev, app)