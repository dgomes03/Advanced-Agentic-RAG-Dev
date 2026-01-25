#!/usr/bin/env python3
"""
Create a demo SQLite database for testing SQL database tools in the RAG Framework.

This script creates a sample database with products, customers, and orders tables,
populates them with realistic sample data, and updates the config.py to include
the database configuration.

Usage:
    python scripts/create_demo_database.py
"""

import os
import sys
import sqlite3
from datetime import datetime, timedelta
import random

# Add parent directory to path (RAG_Framework directory and its parent)
script_dir = os.path.dirname(os.path.abspath(__file__))
rag_framework_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(rag_framework_dir)

sys.path.insert(0, project_root)

from RAG_Framework.core.config import PROJECT_ROOT


def create_demo_database():
    """Create the demo SQLite database with sample data."""

    # Database path
    db_path = os.path.join(PROJECT_ROOT, "RAG_Framework", "data", "demo_data.db")

    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Remove existing database if present
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database at {db_path}")

    # Connect and create tables
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("Creating tables...")

    # Create products table
    cursor.execute('''
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock INTEGER NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create customers table
    cursor.execute('''
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            country TEXT NOT NULL,
            joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create orders table
    cursor.execute('''
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            total_price REAL NOT NULL,
            order_date TIMESTAMP NOT NULL,
            status TEXT DEFAULT 'completed',
            FOREIGN KEY (customer_id) REFERENCES customers(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    ''')

    print("Inserting sample data...")

    # Sample products
    products = [
        ('Laptop Pro 15"', 'Electronics', 1299.99, 45, 'High-performance laptop with 16GB RAM'),
        ('Wireless Mouse', 'Electronics', 29.99, 200, 'Ergonomic wireless mouse with USB receiver'),
        ('USB-C Hub', 'Electronics', 49.99, 150, '7-in-1 USB-C hub with HDMI'),
        ('Mechanical Keyboard', 'Electronics', 89.99, 80, 'RGB mechanical keyboard with Cherry MX switches'),
        ('Monitor 27"', 'Electronics', 399.99, 35, '4K IPS monitor with HDR support'),
        ('Webcam HD', 'Electronics', 79.99, 120, '1080p webcam with built-in microphone'),
        ('Desk Lamp', 'Office', 34.99, 90, 'LED desk lamp with adjustable brightness'),
        ('Office Chair', 'Furniture', 249.99, 25, 'Ergonomic office chair with lumbar support'),
        ('Standing Desk', 'Furniture', 599.99, 15, 'Electric height-adjustable standing desk'),
        ('Notebook Set', 'Office', 12.99, 300, 'Pack of 5 spiral notebooks'),
        ('Pen Set', 'Office', 8.99, 400, 'Pack of 10 ballpoint pens'),
        ('Headphones', 'Electronics', 149.99, 60, 'Noise-cancelling wireless headphones'),
        ('Tablet 10"', 'Electronics', 449.99, 40, '10-inch tablet with stylus support'),
        ('Backpack', 'Accessories', 59.99, 75, 'Laptop backpack with USB charging port'),
        ('Mouse Pad XL', 'Accessories', 19.99, 180, 'Extended mouse pad for gaming')
    ]

    cursor.executemany('''
        INSERT INTO products (name, category, price, stock, description)
        VALUES (?, ?, ?, ?, ?)
    ''', products)

    # Sample customers
    customers = [
        ('John Smith', 'john.smith@email.com', 'USA'),
        ('Emma Johnson', 'emma.j@email.com', 'UK'),
        ('Michael Brown', 'mbrown@email.com', 'Canada'),
        ('Sarah Davis', 'sarah.d@email.com', 'Australia'),
        ('James Wilson', 'jwilson@email.com', 'USA'),
        ('Emily Taylor', 'etaylor@email.com', 'UK'),
        ('Robert Anderson', 'randerson@email.com', 'Germany'),
        ('Lisa Martinez', 'lmartinez@email.com', 'Spain'),
        ('David Garcia', 'dgarcia@email.com', 'Mexico'),
        ('Jennifer Lee', 'jlee@email.com', 'South Korea'),
        ('William Chen', 'wchen@email.com', 'China'),
        ('Amanda White', 'awhite@email.com', 'Canada'),
        ('Christopher Harris', 'charris@email.com', 'USA'),
        ('Jessica Thomas', 'jthomas@email.com', 'Australia'),
        ('Daniel Robinson', 'drobinson@email.com', 'UK')
    ]

    cursor.executemany('''
        INSERT INTO customers (name, email, country)
        VALUES (?, ?, ?)
    ''', customers)

    # Generate sample orders (last 90 days)
    orders = []
    statuses = ['completed', 'completed', 'completed', 'shipped', 'processing', 'cancelled']

    for _ in range(100):
        customer_id = random.randint(1, len(customers))
        product_id = random.randint(1, len(products))
        quantity = random.randint(1, 5)

        # Get product price
        cursor.execute('SELECT price FROM products WHERE id = ?', (product_id,))
        price = cursor.fetchone()[0]
        total_price = round(price * quantity, 2)

        # Random date within last 90 days
        days_ago = random.randint(0, 90)
        order_date = datetime.now() - timedelta(days=days_ago)

        status = random.choice(statuses)

        orders.append((customer_id, product_id, quantity, total_price, order_date.strftime('%Y-%m-%d %H:%M:%S'), status))

    cursor.executemany('''
        INSERT INTO orders (customer_id, product_id, quantity, total_price, order_date, status)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', orders)

    conn.commit()
    conn.close()

    print(f"Demo database created at: {db_path}")
    print(f"  - {len(products)} products")
    print(f"  - {len(customers)} customers")
    print(f"  - {len(orders)} orders")

    return db_path


def update_config(db_path):
    """Update config.py to include the demo database configuration."""

    config_path = os.path.join(PROJECT_ROOT, "RAG_Framework", "core", "config.py")

    with open(config_path, 'r') as f:
        content = f.read()

    # Check if demo_db is already configured
    if '"demo_db"' in content or "'demo_db'" in content:
        print("\nDemo database already configured in config.py")
        return

    # Find the SQL_DATABASE_CONFIGS section and add the demo_db configuration
    old_config = '''SQL_DATABASE_CONFIGS = {
    # Example configuration (uncomment and modify to use):
    # "demo_db": {
    #     "db_type": "sqlite",  # or "postgresql", "mysql"
    #     "connection_string": "/path/to/database.sqlite",
    #     "description": "Demo database with sample data",
    #     "max_rows": 100,
    #     "timeout": 30,
    #     "allowed_tables": ["products", "customers", "orders"],  # optional whitelist
    # }
}'''

    new_config = f'''SQL_DATABASE_CONFIGS = {{
    "demo_db": {{
        "db_type": "sqlite",
        "connection_string": "{db_path}",
        "description": "Demo database with products, customers, and orders for testing SQL queries",
        "max_rows": 100,
        "timeout": 30,
        "allowed_tables": ["products", "customers", "orders"],
    }}
}}'''

    if old_config in content:
        content = content.replace(old_config, new_config)
        with open(config_path, 'w') as f:
            f.write(content)
        print(f"\nUpdated config.py with demo_db configuration")
    else:
        print("\nWarning: Could not find expected SQL_DATABASE_CONFIGS pattern in config.py")
        print("Please manually add the following configuration to SQL_DATABASE_CONFIGS:")
        print(f'''
    "demo_db": {{
        "db_type": "sqlite",
        "connection_string": "{db_path}",
        "description": "Demo database with products, customers, and orders for testing SQL queries",
        "max_rows": 100,
        "timeout": 30,
        "allowed_tables": ["products", "customers", "orders"],
    }}
''')


def test_database(db_path):
    """Run some test queries to verify the database."""

    print("\nTesting database with sample queries...")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Test 1: Count products
    cursor.execute("SELECT COUNT(*) FROM products")
    count = cursor.fetchone()[0]
    print(f"  Total products: {count}")

    # Test 2: Top 5 products by price
    cursor.execute("SELECT name, price FROM products ORDER BY price DESC LIMIT 5")
    print("  Top 5 products by price:")
    for row in cursor.fetchall():
        print(f"    - {row[0]}: ${row[1]}")

    # Test 3: Orders per country
    cursor.execute("""
        SELECT c.country, COUNT(o.id) as order_count, SUM(o.total_price) as total_revenue
        FROM customers c
        JOIN orders o ON c.id = o.customer_id
        GROUP BY c.country
        ORDER BY total_revenue DESC
        LIMIT 5
    """)
    print("  Top 5 countries by revenue:")
    for row in cursor.fetchall():
        print(f"    - {row[0]}: {row[1]} orders, ${row[2]:.2f} revenue")

    conn.close()
    print("\nDatabase tests completed successfully!")


if __name__ == "__main__":
    print("=" * 60)
    print("RAG Framework - Demo Database Setup")
    print("=" * 60)

    # Create the database
    db_path = create_demo_database()

    # Update the config
    update_config(db_path)

    # Test the database
    test_database(db_path)

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nYou can now start the RAG server and test SQL queries:")
    print("  1. Ask: 'What databases are available?'")
    print("  2. Ask: 'What is the schema of demo_db?'")
    print("  3. Ask: 'What are the top 5 products by price?'")
    print("  4. Ask: 'How many orders were placed last month?'")
    print("=" * 60)
