# FEP-MCP Server

A FastMCP server for healthcare data analysis with member search, claims analysis, and database operations.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python server.py
```

## Environment Setup (Optional)

Create a `.env` file for database access:

```env
SQL_SERVER_NAME=your-server.database.windows.net
SQL_DATABASE_NAME=your-database
SQL_DB_USERNAME=your-username
SQL_DB_PASSWORD=your-password
```

## Available Tools

- **Member Management**: `find_member()`, `get_basic_info()`, `search_member_key()`
- **Data Analysis**: `run_claims_analysis()`, `run_graph_generation()`
- **Database**: `run_read_query()`, `run_write_query()`, `get_db_schema()`
- **Clinical**: `get_clinical_notes()`
- **Search**: `retrieve_from_vectorstore()`

## Requirements

- Python 3.8+
- Windows (for SQL Server drivers)
- Optional: Azure SQL Database access
