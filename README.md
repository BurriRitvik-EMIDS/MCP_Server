# FEP-MCP Server

A FastMCP server for healthcare data analysis with member search, claims analysis, and database operations.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python server.py
python client.py
```

## Environment Setup

Create a `.env` file for database access:

```env
# Required for database access
SQL_SERVER_NAME=your-server.database.windows.net
SQL_DATABASE_NAME=your-database
SQL_DB_USERNAME=your-username
SQL_DB_PASSWORD=your-password

# Optional: Override default timeout (in seconds)
MCP_SERVER_REQUEST_TIMEOUT=5

# Optional: SQL Driver version (defaults to "ODBC Driver 18 for SQL Server")
# SQL_DRIVER=ODBC Driver 18 for SQL Server
```

## Data Sources

The server works with the following data sources:
- Claims data from CSV (local or remote)
- Member data from JSON API
- Clinical notes from database
- Vector store for semantic search

## Available Tools

### Member Management
- **find_member**
  - **Description**: Find full member record by member_id.
  - **Inputs**:
    - `member_id` (string)

- **search_member_key**
  - **Description**: Search for a partial key inside a member's record.
  - **Inputs**:
    - `member_id` (string)
    - `search_key` (string)

- **get_basic_info**
  - **Description**: Return basic member info (for customer service) given a member_id.
  - **Inputs**:
    - `member_id` (string)

### Data Analysis
- **run_claims_analysis**
  - **Description**: Execute Python code against claims CSV file.
  - **Inputs**:
    - `python_code` (string)

- **run_graph_generation**
  - **Description**: Execute Python code for graph generation with access to both claims and clinical notes data. Execute Python code that generates a matplotlib plot, and return the image. The code must generate a visual output.
  - **Inputs**:
    - `python_code` (string)

### Database Operations
- **get_db_schema**
  - **Description**: Retrieve database schema (tables & columns).

- **run_read_query**
  - **Description**: Execute a read-only query against the database.
  - **Inputs**:
    - `query` (string)
    - `params` (object, optional)

- **run_write_query**
  - **Description**: Execute a write (INSERT/UPDATE/DELETE) query against the database.
  - **Inputs**:
    - `query` (string)
    - `params` (object, optional)

- **get_clinical_notes**
  - **Description**: Fetch clinical notes for a member from table clinical_report by member_id.
  - **Inputs**:
    - `member_id` (string)

### Vector Store
- **retrieve_from_vectorstore**
  - **Description**: Semantic retrieve top-k chunks from the configured PGVector store. Returns a list of objects with 'page_content' and optional 'metadata'.
  - **Inputs**:
    - `query` (string)
    - `k` (integer, default: 4)

## Requirements

- Python 3.12+
- Windows (for SQL Server drivers)
- ODBC Driver 17 or 18 for SQL Server
- Optional: Azure SQL Database access

## Dependencies

### Core Dependencies
- fastmcp
- langchain_experimental
- typer
- pyodbc
- uv
- requests
- python-dotenv
- pandas
- matplotlib
- numpy

### Vector Store Dependencies
- langchain-community
- langchain-postgres
- sentence-transformers
- psycopg2-binary

## Notes

- The server attempts to use a local CSV file for claims data if available, otherwise falls back to a remote URL
- Matplotlib is configured to use 'Agg' backend for non-interactive use
- Database operations require proper environment variables to be set
