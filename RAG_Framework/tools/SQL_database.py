"""
SQL Database Connector for RAG Framework.
Provides read-only SQL database querying capabilities as a tool for the LLM.
"""

import re
import sqlite3
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


class DatabaseType(Enum):
    """Supported database types."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


@dataclass
class DatabaseConfig:
    """Configuration for a database connection."""
    db_type: DatabaseType
    connection_string: str
    description: str
    max_rows: int = 100
    timeout: int = 30
    allowed_tables: Optional[List[str]] = None

    def __post_init__(self):
        # Convert string to enum if needed
        if isinstance(self.db_type, str):
            self.db_type = DatabaseType(self.db_type.lower())


class SQLValidator:
    """Validates SQL queries for safety."""

    # Keywords that indicate write operations
    BLOCKED_KEYWORDS = [
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE',
        'REPLACE', 'MERGE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE', 'CALL',
        'SET', 'LOCK', 'UNLOCK', 'RENAME', 'LOAD', 'INTO OUTFILE',
        'INTO DUMPFILE', 'ATTACH', 'DETACH'
    ]

    # SQL injection patterns
    INJECTION_PATTERNS = [
        r';\s*(?:DROP|DELETE|UPDATE|INSERT|CREATE|ALTER|TRUNCATE)',  # Chained dangerous commands
        r'--\s*$',  # SQL comment at end (potential injection)
        r'/\*.*\*/',  # Block comments
        r'UNION\s+ALL\s+SELECT.*FROM\s+(?:sqlite_master|information_schema)',  # Schema extraction
        r'@@\w+',  # MySQL system variables
        r'\bSLEEP\s*\(',  # Time-based injection
        r'\bBENCHMARK\s*\(',  # Time-based injection
        r'\bWAITFOR\s+DELAY',  # SQL Server time delay
    ]

    @classmethod
    def validate_query(cls, query: str, allowed_tables: Optional[List[str]] = None) -> tuple[bool, str]:
        """
        Validate a SQL query for safety.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query or not query.strip():
            return False, "Empty query"

        query_upper = query.upper().strip()

        # Must start with SELECT
        if not query_upper.startswith('SELECT'):
            return False, "Only SELECT queries are allowed"

        # Check for blocked keywords
        for keyword in cls.BLOCKED_KEYWORDS:
            # Use word boundary to avoid false positives (e.g., "SELECTED" shouldn't match "SELECT")
            pattern = r'\b' + keyword + r'\b'
            if re.search(pattern, query_upper):
                return False, f"Query contains blocked keyword: {keyword}"

        # Check for multiple statements
        # Count semicolons not inside quotes
        in_single_quote = False
        in_double_quote = False
        semicolon_count = 0

        for i, char in enumerate(query):
            if char == "'" and (i == 0 or query[i-1] != '\\'):
                in_single_quote = not in_single_quote
            elif char == '"' and (i == 0 or query[i-1] != '\\'):
                in_double_quote = not in_double_quote
            elif char == ';' and not in_single_quote and not in_double_quote:
                semicolon_count += 1

        if semicolon_count > 1:
            return False, "Multiple statements are not allowed"

        # Check for injection patterns
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, query_upper, re.IGNORECASE):
                return False, "Query contains potentially dangerous pattern"

        # Check table whitelist if provided
        if allowed_tables:
            # Extract table names from query (basic extraction)
            # This is a simplified approach - complex queries might need more sophisticated parsing
            from_match = re.search(r'\bFROM\s+([^\s,;()]+)', query, re.IGNORECASE)
            if from_match:
                table_name = from_match.group(1).strip('`"[]')
                if table_name.lower() not in [t.lower() for t in allowed_tables]:
                    return False, f"Table '{table_name}' is not in the allowed tables list"

            # Check JOIN tables
            join_matches = re.findall(r'\bJOIN\s+([^\s,;()]+)', query, re.IGNORECASE)
            for table_name in join_matches:
                table_name = table_name.strip('`"[]')
                if table_name.lower() not in [t.lower() for t in allowed_tables]:
                    return False, f"Table '{table_name}' is not in the allowed tables list"

        return True, ""


class SQLConnector:
    """
    Manages SQL database connections and executes read-only queries.
    Implements singleton pattern for global access.
    """

    _instance: Optional['SQLConnector'] = None

    def __init__(self, configs: Optional[Dict[str, DatabaseConfig]] = None):
        """
        Initialize the SQL connector with database configurations.

        Args:
            configs: Dictionary mapping database names to their configurations
        """
        self.configs: Dict[str, DatabaseConfig] = configs or {}
        self._connections: Dict[str, Any] = {}

    def add_database(self, name: str, config: DatabaseConfig) -> None:
        """Add a database configuration."""
        self.configs[name] = config
        # Close existing connection if any
        if name in self._connections:
            self._close_connection(name)

    def remove_database(self, name: str) -> None:
        """Remove a database configuration."""
        if name in self._connections:
            self._close_connection(name)
        if name in self.configs:
            del self.configs[name]

    def _get_connection(self, db_name: str) -> Any:
        """Get or create a connection to the specified database."""
        if db_name not in self.configs:
            raise ValueError(f"Database '{db_name}' is not configured")

        config = self.configs[db_name]

        if db_name in self._connections:
            return self._connections[db_name]

        if config.db_type == DatabaseType.SQLITE:
            # Open SQLite in read-only mode
            conn_string = config.connection_string
            if '?' not in conn_string:
                conn_string += '?mode=ro'
            elif 'mode=' not in conn_string:
                conn_string += '&mode=ro'

            # Use URI format for read-only mode
            conn = sqlite3.connect(f'file:{conn_string}', uri=True, timeout=config.timeout)
            conn.row_factory = sqlite3.Row
            self._connections[db_name] = conn

        elif config.db_type == DatabaseType.POSTGRESQL:
            try:
                import psycopg2
                import psycopg2.extras
            except ImportError:
                raise ImportError("psycopg2 is required for PostgreSQL support. Install with: pip install psycopg2-binary")

            conn = psycopg2.connect(config.connection_string)
            conn.set_session(readonly=True, autocommit=True)
            self._connections[db_name] = conn

        elif config.db_type == DatabaseType.MYSQL:
            try:
                import mysql.connector
            except ImportError:
                raise ImportError("mysql-connector-python is required for MySQL support. Install with: pip install mysql-connector-python")

            conn = mysql.connector.connect(
                **self._parse_mysql_connection_string(config.connection_string),
                connection_timeout=config.timeout
            )
            self._connections[db_name] = conn

        return self._connections[db_name]

    def _parse_mysql_connection_string(self, conn_string: str) -> dict:
        """Parse MySQL connection string into connection parameters."""
        # Support format: mysql://user:password@host:port/database
        import urllib.parse

        parsed = urllib.parse.urlparse(conn_string)
        return {
            'host': parsed.hostname or 'localhost',
            'port': parsed.port or 3306,
            'user': parsed.username,
            'password': parsed.password,
            'database': parsed.path.lstrip('/') if parsed.path else None
        }

    def _close_connection(self, db_name: str) -> None:
        """Close a database connection."""
        if db_name in self._connections:
            try:
                self._connections[db_name].close()
            except Exception:
                pass
            del self._connections[db_name]

    def execute_query(self, db_name: str, query: str) -> Dict[str, Any]:
        """
        Execute a read-only SQL query.

        Args:
            db_name: Name of the configured database
            query: SQL SELECT query to execute

        Returns:
            Dictionary with query results or error information
        """
        # Validate the query
        if db_name not in self.configs:
            return {
                "success": False,
                "error": f"Database '{db_name}' is not configured. Use list_databases to see available databases."
            }

        config = self.configs[db_name]
        is_valid, error_msg = SQLValidator.validate_query(query, config.allowed_tables)

        if not is_valid:
            return {
                "success": False,
                "error": f"Query validation failed: {error_msg}"
            }

        try:
            conn = self._get_connection(db_name)
            cursor = conn.cursor()

            # Set timeout for the query if supported
            if config.db_type == DatabaseType.SQLITE:
                # SQLite uses connection timeout set during connect
                pass
            elif config.db_type == DatabaseType.POSTGRESQL:
                cursor.execute(f"SET statement_timeout = '{config.timeout * 1000}'")
            elif config.db_type == DatabaseType.MYSQL:
                cursor.execute(f"SET max_execution_time = {config.timeout * 1000}")

            cursor.execute(query)

            # Fetch results with row limit
            rows = cursor.fetchmany(config.max_rows)

            # Get column names
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
            else:
                columns = []

            # Convert rows to list of dicts
            if config.db_type == DatabaseType.SQLITE:
                results = [dict(row) for row in rows]
            else:
                results = [dict(zip(columns, row)) for row in rows]

            # Check if there are more rows
            has_more = cursor.fetchone() is not None

            cursor.close()

            return {
                "success": True,
                "database": db_name,
                "query": query,
                "columns": columns,
                "rows": results,
                "row_count": len(results),
                "truncated": has_more,
                "max_rows": config.max_rows
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Query execution failed: {str(e)}"
            }

    def list_databases(self) -> Dict[str, Any]:
        """List all configured databases with their descriptions."""
        databases = []
        for name, config in self.configs.items():
            databases.append({
                "name": name,
                "type": config.db_type.value,
                "description": config.description,
                "max_rows": config.max_rows,
                "allowed_tables": config.allowed_tables
            })

        return {
            "success": True,
            "databases": databases,
            "count": len(databases)
        }

    def get_schema(self, db_name: str) -> Dict[str, Any]:
        """
        Get the schema (tables and columns) for a database.

        Args:
            db_name: Name of the configured database

        Returns:
            Dictionary with schema information
        """
        if db_name not in self.configs:
            return {
                "success": False,
                "error": f"Database '{db_name}' is not configured"
            }

        config = self.configs[db_name]

        try:
            conn = self._get_connection(db_name)
            cursor = conn.cursor()

            tables = []

            if config.db_type == DatabaseType.SQLITE:
                # Get tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                table_names = [row[0] for row in cursor.fetchall()]

                for table_name in table_names:
                    # Skip if not in allowed tables
                    if config.allowed_tables and table_name.lower() not in [t.lower() for t in config.allowed_tables]:
                        continue

                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = []
                    for col in cursor.fetchall():
                        columns.append({
                            "name": col[1],
                            "type": col[2],
                            "nullable": not col[3],
                            "primary_key": bool(col[5])
                        })

                    tables.append({
                        "name": table_name,
                        "columns": columns
                    })

            elif config.db_type == DatabaseType.POSTGRESQL:
                cursor.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                """)
                table_names = [row[0] for row in cursor.fetchall()]

                for table_name in table_names:
                    if config.allowed_tables and table_name.lower() not in [t.lower() for t in config.allowed_tables]:
                        continue

                    cursor.execute("""
                        SELECT column_name, data_type, is_nullable,
                               CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END as is_primary
                        FROM information_schema.columns c
                        LEFT JOIN (
                            SELECT ku.column_name
                            FROM information_schema.table_constraints tc
                            JOIN information_schema.key_column_usage ku ON tc.constraint_name = ku.constraint_name
                            WHERE tc.table_name = %s AND tc.constraint_type = 'PRIMARY KEY'
                        ) pk ON c.column_name = pk.column_name
                        WHERE c.table_name = %s
                    """, (table_name, table_name))

                    columns = []
                    for col in cursor.fetchall():
                        columns.append({
                            "name": col[0],
                            "type": col[1],
                            "nullable": col[2] == 'YES',
                            "primary_key": col[3]
                        })

                    tables.append({
                        "name": table_name,
                        "columns": columns
                    })

            elif config.db_type == DatabaseType.MYSQL:
                cursor.execute("SHOW TABLES")
                table_names = [row[0] for row in cursor.fetchall()]

                for table_name in table_names:
                    if config.allowed_tables and table_name.lower() not in [t.lower() for t in config.allowed_tables]:
                        continue

                    cursor.execute(f"DESCRIBE {table_name}")
                    columns = []
                    for col in cursor.fetchall():
                        columns.append({
                            "name": col[0],
                            "type": col[1],
                            "nullable": col[2] == 'YES',
                            "primary_key": col[3] == 'PRI'
                        })

                    tables.append({
                        "name": table_name,
                        "columns": columns
                    })

            cursor.close()

            return {
                "success": True,
                "database": db_name,
                "description": config.description,
                "tables": tables,
                "table_count": len(tables)
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get schema: {str(e)}"
            }

    def close_all(self) -> None:
        """Close all database connections."""
        for db_name in list(self._connections.keys()):
            self._close_connection(db_name)


# Global singleton instance
_sql_connector: Optional[SQLConnector] = None


def get_sql_connector() -> Optional[SQLConnector]:
    """Get the global SQL connector instance."""
    return _sql_connector


def initialize_sql_connector(configs: Optional[Dict[str, DatabaseConfig]] = None) -> SQLConnector:
    """
    Initialize the global SQL connector instance.

    Args:
        configs: Dictionary mapping database names to their configurations

    Returns:
        The initialized SQLConnector instance
    """
    global _sql_connector
    _sql_connector = SQLConnector(configs)
    return _sql_connector
