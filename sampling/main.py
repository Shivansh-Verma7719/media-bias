import os
import sys
from dotenv import load_dotenv
import pandas as pd
import psycopg2

# Add parent directory to path to import helpers
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

load_dotenv(os.path.join(parent_dir, '.env'))

def main():
    db_url = os.getenv("POOLER_DATABASE_URL")
    if not db_url:
        print("Error: POOLER_DATABASE_URL not found in .env")
        return

    print("Connecting to PostgreSQL...")
    try:
        conn = psycopg2.connect(db_url)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return

    # Query to aggregate articles by company symbol and year
    query = """
    SELECT 
        c.symbol, 
        CAST(EXTRACT(YEAR FROM a.published_at) AS INTEGER) AS year, 
        COUNT(a.id) AS article_count
    FROM 
        articles_sample a
    JOIN 
        top_companies c ON a.company_id = c.id
    WHERE 
        a.published_at IS NOT NULL
    GROUP BY 
        c.symbol, 
        EXTRACT(YEAR FROM a.published_at)
    ORDER BY 
        c.symbol, year
    """
    
    print("Executing query and fetching aggregations...")
    df = pd.read_sql(query, conn)
    conn.close()
    
    if df.empty:
        print("No articles found.")
        return

    print(f"Fetched {len(df)} aggregated records.")

    # Create the 2D matrix: rows=symbol, columns=year, values=article_count
    print("Pivoting data to create matrix...")
    matrix = df.pivot(index='symbol', columns='year', values='article_count').fillna(0).astype(int)
    
    # Add row totals
    matrix['Total'] = matrix.sum(axis=1)
    
    # Add column totals as a new row
    # We must reset index to avoid 'Total' being indexed as a symbol string type mismatch if any
    # So we simply append a series
    total_series = matrix.sum(axis=0)
    total_series.name = 'Total'
    matrix = pd.concat([matrix, total_series.to_frame().T])
    
    # Save to CSV
    output_file = os.path.join(current_dir, 'stratified_sample_matrix.csv')
    matrix.to_csv(output_file)
    print(f"Successfully saved matrix to {output_file}")
    
    # Print a preview
    print("\nPreview of the matrix:")
    print(matrix.head())
    print("\nSummary statistics (total articles per year):")
    print(matrix.loc['Total'])

if __name__ == '__main__':
    main()

