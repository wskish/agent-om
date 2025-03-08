# Agent-OM Code Guidelines

## Commands
- Run main app: `python main.py`
- Run specific tool: `python tool_example.py`
- Type check: `mypy *.py`
- Lint: `flake8 *.py`

## Code Style
- **Imports**: stdlib first, third-party second, local modules last
- **Types**: Use type hints for all functions and variables
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Models**: Use Pydantic for data validation and structure
- **Error handling**: Use try/except blocks with appropriate logging
- **Docstrings**: Required for all functions explaining purpose
- **Formatting**: Max line length 88 characters
- **Functions**: Single responsibility principle, clear input/output

## Architecture
This codebase handles AI tool integrations with various LLM models. The main components include tool execution frameworks and specialized tools.