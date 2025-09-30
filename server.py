from fastmcp import FastMCP
import os as _os
from typing import Optional, List, Tuple, Dict, Any, Union
import requests
import json
import io
from contextlib import redirect_stdout
from langchain_experimental.utilities import PythonREPL
import pyodbc
import time
import os
from dotenv import load_dotenv
import sys
from typing import List, Dict, Any

# Create MCP app with a longer request timeout to avoid 5s client-side timeouts
_REQUEST_TIMEOUT = float(_os.getenv(
    "MCP_SERVER_REQUEST_TIMEOUT", _os.getenv("MCP_REQUEST_TIMEOUT", "5")))
mcp = FastMCP("FEP-MCP")

# Pre-import heavy libraries once at server start to avoid per-call overhead
try:
    import pandas as _pd  # type: ignore
except Exception:  # pragma: no cover
    _pd = None  # type: ignore
try:
    import matplotlib as _mpl  # type: ignore
    _mpl.use("Agg")
    import matplotlib.pyplot as _plt  # type: ignore
except Exception:  # pragma: no cover
    _plt = None  # type: ignore

# ---- Shared constants ----
# Prefer a local CSV to avoid slow network fetches and MCP timeouts
CSV_REMOTE_URL = "https://genaipoddatademo.blob.core.windows.net/csvdata/claims_history(in)_new.csv?sp=r&st=2025-03-10T10:41:01Z&se=2026-03-10T18:41:01Z&spr=https&sv=2022-11-02&sr=b&sig=e3yQB2OaZcT1Lbp8tapOkLf5GSFVWE%2F8sak5UQ0od6A%3D"
# CSV_REMOTE_URL = r"D:\FEP\FEP-POC-ADK-v2\claims_history(in)_new (2) 1.csv"
CSV_URL = CSV_REMOTE_URL


def _init_csv_cache() -> None:
    """Set CSV_URL to a local file if available; otherwise fall back to remote.

    Avoid performing any network download at import time to prevent startup delays
    that can cause MCP tool calls to time out.
    """
    global CSV_URL
    try:
        local_path = os.path.join(
            os.path.dirname(__file__),
            "agents",
            "claim_agent",
            "data",
            "claims_history(in).csv",
        )
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            CSV_URL = local_path
            return
    except Exception:
        pass
    # Fallback remains the remote URL
    CSV_URL = CSV_REMOTE_URL


# Initialize at import time (fast, no network)
_init_csv_cache()

# ---- JSON Search Logic ----


def search_json(member_id: str, search_key: Optional[str] = None) -> Union[Dict, List[Tuple[str, Any]], None]:
    file_url = 'https://genaipoddatademo.blob.core.windows.net/csvdata/dummy_patient_data_new.json?sp=r&st=2025-03-24T12:44:37Z&se=2026-03-24T20:44:37Z&spr=https&sv=2024-11-04&sr=b&sig=vtt19xTWblJaIFdxByvdNRM6j5C8CoZxeTqAaRccI90%3D'
    try:
        response = requests.get(file_url)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"Error fetching or decoding JSON file: {e}", file=sys.stderr)
        return None

    member_record = next(
        (r for r in data if r.get('member_id') == member_id), None)
    if not member_record:
        return None

    if search_key is None:
        return member_record

    results: List[Tuple[str, Any]] = []

    def recursive_search(element: Any, key_to_find: str):
        if isinstance(element, dict):
            for key, value in element.items():
                if key_to_find.lower() in key.lower():
                    results.append((key, value))
                if isinstance(value, (dict, list)):
                    recursive_search(value, key_to_find)
        elif isinstance(element, list):
            for item in element:
                recursive_search(item, key_to_find)

    recursive_search(member_record, search_key)
    return results


def get_basic_member_info(member_record: Dict[str, Any]) -> Dict[str, Any]:
    allowed_keys = {"member_id", "first_name", "last_name",
                    "date_of_birth", "phone", "email", "address", "insurance_plan"}
    basic_info = {key: member_record[key]
                  for key in allowed_keys if key in member_record}
    if "insurance_plan" in member_record and isinstance(member_record["insurance_plan"], dict):
        insurance = member_record["insurance_plan"]
        allowed_insurance_keys = {"name", "plan_start_date",
                                  "plan_end_date", "premium_amount", "coverage_level"}
        basic_insurance = {k: insurance[k]
                           for k in allowed_insurance_keys if k in insurance}
        if basic_insurance:
            basic_info["insurance"] = basic_insurance
    return basic_info


def python_repl_tool(python_code: str) -> str:
    # Inject commonly used globals to speed up user code and avoid repeated imports
    initial_globals = {"csv_url": CSV_URL}
    if _pd is not None:
        initial_globals["pd"] = _pd
    python_repl = PythonREPL(_globals=initial_globals)
    # Ensure matplotlib doesn't try to open GUI windows; auto-export on show
    try:
        # Prefer pre-imported pyplot if available
        if _plt is None:
            import matplotlib  # type: ignore
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore
        else:
            plt = _plt  # type: ignore
        import base64 as _b64
        from io import BytesIO as _BytesIO

        def _auto_export_show(*args, **kwargs):
            try:
                buf = _BytesIO()
                try:
                    plt.tight_layout()
                except Exception:
                    pass
                plt.savefig(buf, format="png")
                plt.close()
                buf.seek(0)
                print(_b64.b64encode(buf.read()).decode("utf-8"))
            except Exception:
                # If export fails, just suppress to avoid blocking
                try:
                    plt.close()
                except Exception:
                    pass

        # Expose plt to the REPL and monkey-patch show
        python_repl.globals["plt"] = plt
        try:
            plt.show = _auto_export_show  # type: ignore
        except Exception:
            pass
    except Exception:
        pass
    try:
        f = io.StringIO()
        with redirect_stdout(f):
            try:
                execution_result = python_repl.run(python_code)
            except Exception as e:
                execution_result = f"[Execution Error] {e}"
        output = f.getvalue().strip()
        if not output:
            output = execution_result or "[No output]"
    except Exception as e:
        output = f"[Execution Error] {e}"
    return output


# ---- Azure SQL Database Logic ----
load_dotenv()

# Prefer newer ODBC Driver 18 on Windows if available; fall back to 17
sql_driver = os.getenv("SQL_DRIVER", "ODBC Driver 18 for SQL Server")
sql_server_name = os.getenv("SQL_SERVER_NAME")
sql_database_name = os.getenv("SQL_DATABASE_NAME")
sql_db_username = os.getenv("SQL_DB_USERNAME")
sql_db_password = os.getenv("SQL_DB_PASSWORD")

DB_AVAILABLE = all([sql_server_name, sql_database_name,
                   sql_db_username, sql_db_password])


def _build_connection_string(driver: str) -> str:
    return (
        f"DRIVER={driver};"
        f"SERVER={sql_server_name};"
        f"DATABASE={sql_database_name};"
        f"UID={sql_db_username};"
        f"PWD={sql_db_password};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=no;"
        f"Connection Timeout=30;"
    )


connection_string = _build_connection_string(
    sql_driver) if DB_AVAILABLE else None


def get_db_connection(retries=1, delay=1):
    if not DB_AVAILABLE:
        raise RuntimeError("Database configuration not available")
    for attempt in range(retries):
        try:
            return pyodbc.connect(connection_string, timeout=2)
        except pyodbc.Error as e:
            print(
                f"Attempt {attempt+1} - Error connecting to database: {e}", file=sys.stderr)
            # If driver might be missing/mismatched, attempt fallback to Driver 17 once
            if attempt == retries - 1:
                try:
                    alt_driver = "ODBC Driver 17 for SQL Server" if "18" in sql_driver else "ODBC Driver 18 for SQL Server"
                    alt_conn_str = _build_connection_string(alt_driver)
                    print(
                        f"Trying alternate SQL Server driver: {alt_driver}", file=sys.stderr)
                    return pyodbc.connect(alt_conn_str, timeout=2)
                except Exception:
                    pass
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise


def get_schema() -> List[Dict[str, Any]]:
    if not DB_AVAILABLE:
        return [{"error": "Database not configured. Set SQL_* env vars."}]
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                t.name AS table_name,
                c.name AS column_name,
                typ.name AS data_type
            FROM sys.tables AS t
            JOIN sys.columns AS c ON t.object_id = c.object_id
            JOIN sys.types AS typ ON c.user_type_id = typ.user_type_id
            WHERE t.is_ms_shipped = 0
            ORDER BY t.name, c.column_id;
        """)
        schema_data = cursor.fetchall()
        formatted_schema = {}
        for table, column, data_type in schema_data:
            if table not in formatted_schema:
                formatted_schema[table] = []
            formatted_schema[table].append(
                {"column": column, "type": data_type})
        return [{"table_name": table, "columns": cols} for table, cols in formatted_schema.items()]
    except pyodbc.Error as ex:
        return [{"error": f"Database schema retrieval failed: {ex}"}]
    finally:
        if conn:
            conn.close()


def execute_read_query(query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    if not DB_AVAILABLE:
        return [{"error": "Database not configured. Set SQL_* env vars."}]
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        if params:
            cursor.execute(query, list(params.values()))
        else:
            cursor.execute(query)
        columns = [column[0] for column in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return results
    except pyodbc.Error as ex:
        return [{"error": f"Database query failed: {ex}"}]
    finally:
        if conn:
            conn.close()


def execute_write_query(query: str,  params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not DB_AVAILABLE:
        return {"error": "Database not configured. Set SQL_* env vars."}
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        if params:
            cursor.execute(query, list(params.values()))
        else:
            cursor.execute(query)
        conn.commit()
        return {"rows_affected": cursor.rowcount}
    except pyodbc.Error as ex:
        if conn:
            conn.rollback()
        return {"error": f"Database write failed: {ex}"}
    finally:
        if conn:
            conn.close()

# ---- Tools exposed to MCP ----


@mcp.tool()
def find_member(member_id: str) -> Optional[Dict[str, Any]]:
    """Find full member record by member_id."""
    return search_json(member_id=member_id)


@mcp.tool()
def search_member_key(member_id: str, search_key: str) -> List[Tuple[str, Any]]:
    """Search for a partial key inside a member's record."""
    result = search_json(member_id=member_id, search_key=search_key)
    return result if result else []


@mcp.tool()
def get_basic_info(member_id: str) -> Optional[Dict[str, Any]]:
    """Return basic member info (for customer service) given a member_id."""
    member = search_json(member_id)
    if member:
        return get_basic_member_info(member)
    return None


@mcp.tool()
def run_claims_analysis(python_code: str) -> str:
    """Execute Python code against claims CSV file."""
    return python_repl_tool(python_code)


@mcp.tool()
def get_db_schema() -> List[Dict[str, Any]]:
    """Retrieve database schema (tables & columns)."""
    return get_schema()


@mcp.tool()
def run_read_query(query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Execute a read-only query against the database."""
    return execute_read_query(query, params)


@mcp.tool()
def run_write_query(query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute a write (INSERT/UPDATE/DELETE) query against the database."""
    return execute_write_query(query, params)


@mcp.tool()
def get_clinical_notes(member_id: str) -> List[Dict[str, Any]]:
    """Fetch clinical notes for a member from table clinical_report by member_id."""
    query = "SELECT * FROM clinical_report WHERE member_id = ?"
    return execute_read_query(query, params={"member_id": member_id})


@mcp.tool()
def retrieve_from_vectorstore(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """Semantic retrieve top-k chunks from the configured PGVector store.

    Returns a list of objects with 'page_content' and optional 'metadata'.
    """
    try:
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        from langchain_postgres.vectorstores import PGVector
        import os
        from dotenv import load_dotenv

        # Load environment variables
        load_dotenv()

        # Get required environment variables
        required_vars = [
            "POSTGRES_USER", "POSTGRES_PASSWORD",
            "POSTGRES_HOST", "POSTGRES_DB"
        ]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            return [{"error": f"Missing required environment variables: {', '.join(missing_vars)}"}]

        # Set up the embedding model
        model_name = "BAAI/bge-small-en"
        hf = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Construct the connection string
        connection = f"postgresql+psycopg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}/{os.getenv('POSTGRES_DB')}"

        # Create vector store and retriever
        vector_store = PGVector(
            embeddings=hf,
            collection_name="health_documents",
            connection=connection,
            use_jsonb=True,
        )
        retriever = vector_store.as_retriever()

        # Perform the search
        docs = retriever.get_relevant_documents(query)
        results: List[Dict[str, Any]] = []
        for d in docs[: max(1, int(k))]:
            metadata = {}
            # Include actual metadata fields from your table
            for key in ["doc_type", "source", "document_id", "page_number"]:
                if hasattr(d, key):
                    metadata[key] = getattr(d, key)
                elif isinstance(d.metadata, dict) and key in d.metadata:
                    metadata[key] = d.metadata[key]

            results.append({
                "page_content": getattr(d, "page_content", ""),
                "metadata": metadata  # This is now safe and matches your schema
            })

        return results
    except Exception as e:
        return [{"error": f"Retrieval failed: {str(e)}"}]


@mcp.tool()
def run_graph_generation(python_code: str) -> str:
    """Execute Python code for graph generation with access to both claims and clinical notes data.
    Execute Python code that generates a matplotlib plot, and return the image.
    The code must generate a visual output."""
    # Enhanced initial globals for graph generation
    initial_globals = {
        "csv_url": CSV_URL,  # claims data
        "claims_url": "https://genaipoddatademo.blob.core.windows.net/csvdata/claims_history(in)_new.csv?sp=r&st=2025-03-10T10:41:01Z&se=2026-03-10T18:41:01Z&spr=https&sv=2022-11-02&sr=b&sig=e3yQB2OaZcT1Lbp8tapOkLf5GSFVWE%2F8sak5UQ0od6A%3D",
        "clinical_url": "https://genaipoddatademo.blob.core.windows.net/csvdata/clinical_Notes1(in)_new.csv?sp=r&st=2025-03-10T10:39:26Z&se=2026-03-10T18:39:26Z&spr=https&sv=2022-11-02&sr=b&sig=IyrpvyffmLlyM6rLOzdu9aJ72bel2IFZEj35sMQCecw%3D"
    }
    if _pd is not None:
        initial_globals["pd"] = _pd

    # Add base64 and BytesIO to globals so they're available in the code
    import base64 as _b64
    from io import BytesIO as _BytesIO
    initial_globals["base64"] = _b64
    initial_globals["BytesIO"] = _BytesIO

    python_repl = PythonREPL(_globals=initial_globals)

    # Ensure matplotlib doesn't try to open GUI windows; auto-export on show
    try:
        # Prefer pre-imported pyplot if available
        if _plt is None:
            import matplotlib  # type: ignore
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore
        else:
            plt = _plt  # type: ignore

        def _auto_export_show(*args, **kwargs):
            try:
                buf = _BytesIO()
                try:
                    plt.tight_layout()
                except Exception:
                    pass
                plt.savefig(buf, format="png")
                plt.close()
                buf.seek(0)
                print(_b64.b64encode(buf.read()).decode("utf-8"))
            except Exception:
                # If export fails, just suppress to avoid blocking
                try:
                    plt.close()
                except Exception:
                    pass

        # Expose plt to the REPL and monkey-patch show
        python_repl.globals["plt"] = plt
        try:
            plt.show = _auto_export_show  # type: ignore
        except Exception:
            pass
    except Exception:
        pass

    try:
        f = io.StringIO()
        with redirect_stdout(f):
            try:
                execution_result = python_repl.run(python_code)
            except Exception as e:
                execution_result = f"[Execution Error] {e}"
        output = f.getvalue().strip()
        if not output:
            output = execution_result or "[No output]"
    except Exception as e:
        output = f"[Execution Error] {e}"
    return output


# ---- Run server ----
if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8000)
