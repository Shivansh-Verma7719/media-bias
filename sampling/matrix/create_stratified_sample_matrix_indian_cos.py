import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

def main():
    # Load .env from base directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = current_dir # sampling/
    project_dir = os.path.dirname(base_dir) # media-bias/
    env_path = os.path.join(project_dir, '.env')
    
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        print(f"Warning: .env file not found at {env_path}")
    
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        raise ValueError("DATABASE_URL not found in .env")
        
    print("Connecting to database...")
    conn = psycopg2.connect(db_url)
    
    # Query to get article counts per company per year
    query = """
    WITH yearly_counts AS (
        SELECT 
            c.symbol,
            EXTRACT(YEAR FROM a.published_at) AS year,
            COUNT(a.id) AS article_count
        FROM indian_cos.articles a
        JOIN indian_cos.companies c ON a.company_id = c.id
        WHERE c.symbol IS NOT NULL AND a.published_at IS NOT NULL
        GROUP BY c.symbol, EXTRACT(YEAR FROM a.published_at)
    )
    SELECT symbol, year, article_count
    FROM yearly_counts
    ORDER BY symbol, year;
    """
    
    print("Fetching yearly article counts per company...")
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    if not rows:
        print("No data found!")
        return
        
    # Convert exactly to dataframe
    df = pd.DataFrame(rows, columns=['symbol', 'year', 'article_count'])
    
    # Pivot to get years as columns, mapping missing years to 0
    # The target format is exactly like stratified_sample_matrix.csv
    pivot_df = df.pivot(index='symbol', columns='year', values='article_count').fillna(0).astype(int)
    
    # Calculate row totals
    pivot_df['Total'] = pivot_df.sum(axis=1)
    
    # Sort by Total in descending order to match the top N pattern
    pivot_df = pivot_df.sort_values(by='Total', ascending=False)
    
    # Filter for top 30 companies to match original stratified sample
    top_30_df = pivot_df.head(30)
    
    # Ensure column names are integers (for years), then Total at the end
    # Get only the year columns and sort them integer wise
    year_cols = [c for c in top_30_df.columns if c != 'Total']
    sorted_year_cols = sorted(year_cols, key=int)
    
    # Reorder columns
    ordered_cols = sorted_year_cols + ['Total']
    top_30_df = top_30_df[ordered_cols]
    
    # Also calculate the grand total row
    top_30_df.loc['Total'] = top_30_df.sum(axis=0)
    
    # Format the year columns to string for the CSV export
    # e.g., '2015.0' -> '2015'
    top_30_df.columns = [str(int(c)) if c != 'Total' else 'Total' for c in ordered_cols]
    
    # Save the dataframe without an index label explicitly to match CSV empty first col head
    output_path = os.path.join(current_dir, 'stratified_sample_matrix_indian_cos.csv')
    top_30_df.index.name = None
    
    top_30_df.to_csv(output_path)
    
    print(f"Successfully generated matrix for top 30 Indian companies at: {output_path}")

if __name__ == "__main__":
    main()
