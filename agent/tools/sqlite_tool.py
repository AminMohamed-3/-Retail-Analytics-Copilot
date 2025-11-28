"""SQLite database tool with schema introspection and query execution."""
import sqlite3
from typing import Dict, List, Any
from config import get_db_connection


class SQLiteTool:
    """Tool for interacting with SQLite database."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path
    
    def _quote_identifier(self, identifier: str) -> str:
        """Quote SQL identifier if needed."""
        if ' ' in identifier or '-' in identifier or not identifier.replace('_', '').isalnum():
            return f'"{identifier}"'
        return identifier
    
    def get_schema(self, table_name: str = None) -> str:
        """Get schema for a table or all tables."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        schema_info = []
        
        if table_name:
            quoted_name = self._quote_identifier(table_name)
            cursor.execute(f"PRAGMA table_info({quoted_name})")
            columns = cursor.fetchall()
            schema_info.append(f"Table: {table_name}")
            for col in columns:
                schema_info.append(f"  {col[1]} ({col[2]})")
        else:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
            views = [row[0] for row in cursor.fetchall()]
            
            schema_info.append("Tables:")
            for table in tables:
                quoted_name = self._quote_identifier(table)
                cursor.execute(f"PRAGMA table_info({quoted_name})")
                columns = cursor.fetchall()
                schema_info.append(f"\n{table}:")
                for col in columns:
                    schema_info.append(f"  {col[1]} ({col[2]})")
            
            if views:
                schema_info.append("\nViews:")
                for view in views:
                    schema_info.append(f"  {view}")
        
        conn.close()
        return "\n".join(schema_info)
    
    def get_table_names(self) -> List[str]:
        """Get list of all table names."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables
    
    def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute SQL query and return results."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        result = {
            "success": False,
            "columns": [],
            "rows": [],
            "error": None,
            "row_count": 0
        }
        
        try:
            cursor.execute(query)
            result["columns"] = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            result["rows"] = [dict(row) for row in rows]
            result["row_count"] = len(rows)
            result["success"] = True
        except sqlite3.Error as e:
            result["error"] = str(e)
            result["success"] = False
        finally:
            conn.close()
        
        return result
    
    def get_full_schema_summary(self) -> str:
        """Get concise schema summary for LLM context."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        summary_parts = []
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        
        summary_parts.append("DATABASE SCHEMA:")
        summary_parts.append("=" * 50)
        
        key_tables = {
            "Orders": "orders (view)",
            "Order Details": "order_items (view)",
            "Products": "products (view)",
            "Customers": "customers (view)",
            "Categories": "Categories (table)",
        }
        
        for table_name, display_name in key_tables.items():
            if table_name in tables:
                quoted_name = self._quote_identifier(table_name)
                cursor.execute(f"PRAGMA table_info({quoted_name})")
                columns = cursor.fetchall()
                col_names = [col[1] for col in columns]
                summary_parts.append(f"\n{display_name}:")
                summary_parts.append(f"  Columns: {', '.join(col_names)}")
        
        summary_parts.append("\nJOINS:")
        summary_parts.append("  orders o JOIN order_items od ON o.OrderID = od.OrderID")
        summary_parts.append("  order_items od JOIN products p ON od.ProductID = p.ProductID")
        summary_parts.append("  products p JOIN Categories cat ON p.CategoryID = cat.CategoryID")
        summary_parts.append("  orders o JOIN customers c ON o.CustomerID = c.CustomerID")
        
        summary_parts.append("\nALIASES: o=orders, od=order_items, p=products, cat=Categories, c=customers")
        summary_parts.append("REVENUE: SUM(od.UnitPrice * od.Quantity * (1 - od.Discount))")
        
        conn.close()
        return "\n".join(summary_parts)


if __name__ == "__main__":
    tool = SQLiteTool()
    print("Schema Summary:")
    print(tool.get_full_schema_summary())
    print("\n\nTesting query:")
    result = tool.execute_query("SELECT COUNT(*) as total FROM orders")
    print(f"Success: {result['success']}")
    print(f"Rows: {result['rows']}")
