import os
import sys
import pandas as pd
import psycopg2
from dotenv import load_dotenv

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

    csv_path = os.path.join(current_dir, 'stratified_sample_matrix.csv')
    df = pd.read_csv(csv_path, index_col=0)
    
    # Exclude the "Total" row if present
    if 'Total' in df.index:
        df = df.drop('Total')
        
    year_cols = [c for c in df.columns if c != 'Total' and c != 'Unnamed: 0']
    
    kept_companies = []
    booted_companies = []
    
    results_data = []
    
    print("Evaluating companies (Tolerance: <= 3000 articles at most 3 times):")
    for symbol, row in df.iterrows():
        # count years where articles <= 3000
        # the row data might be strings if anything got messed up, so convert to numeric
        values = pd.to_numeric(row[year_cols], errors='coerce').fillna(0)
        faults = sum(values <= 3000)
        
        if faults <= 3:
            kept_companies.append((symbol, faults))
            status = "KEPT"
        else:
            booted_companies.append((symbol, faults))
            status = "BOOTED"
            
        results_data.append({"Company": symbol, "Faults": faults, "Status": status})
        print(f"{symbol: <6} | Faults: {faults: <2} | Status: {status}")
        
    print("\nSummary:")
    print(f"Kept: {[s[0] for s in kept_companies]}")
    print(f"Booted: {[s[0] for s in booted_companies]}")
    
    # Export to Excel
    try:
        results_df = pd.DataFrame(results_data)
        excel_path = os.path.join(current_dir, 'stratified_selection_results.xlsx')
        results_df.to_excel(excel_path, index=False)
        print(f"Exported selection results to {excel_path}")
    except Exception as e:
        print(f"Error exporting to Excel (maybe openpyxl is missing?): {e}")
        csv_fallback = os.path.join(current_dir, 'stratified_selection_results.csv')
        results_df.to_csv(csv_fallback, index=False)
        print(f"Exported fallback CSV to {csv_fallback}")
    
    # Now connect to database and insert records
    print("\nConnecting to PostgreSQL...")
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    kept_symbols = [s[0] for s in kept_companies]
    
    if kept_symbols:
        placeholders = ', '.join(['%s'] * len(kept_symbols))
        
        columns = [
            "id", "title", "content", "url", "source", "published_at", 
            "media_outlet_id", "company_id", "social_data", "created_at", 
            "content_scraped", "last_scraped_at", "scraping_retry_count", 
            "scraping_error", "raw_content", "pos_score", "neutral_score", 
            "neg_score", "link_health", "link_ok"
        ]
        
        select_cols = []
        for c in columns:
            if c == "neutral_score":
                select_cols.append("NULL AS neutral_score")
            elif c == "neg_score":
                # Ensure neg_score passes too if it is also missing, though error was only for neutral_score
                # Let's just assume it also might be missing or handle it exactly as user said for neutral
                select_cols.append("a.neg_score")
            else:
                select_cols.append("a." + c)
                
        cols_str = ", ".join(select_cols)
        target_cols_str = ", ".join(columns)
        
        insert_query = f"""
        INSERT INTO articles_stratified ({target_cols_str})
        SELECT {cols_str}
        FROM articles_sample a
        JOIN top_companies c ON a.company_id = c.id
        WHERE c.symbol IN ({placeholders});
        """
        
        print(f"Inserting articles into articles_stratified for {len(kept_symbols)} companies...")
        
        try:
            cur.execute(insert_query, kept_symbols)
            row_count = cur.rowcount
            print(f"Successfully inserted {row_count} articles.")
            conn.commit()
        except Exception as e:
            print(f"Error executing insert: {e}")
            conn.rollback()
    
    cur.close()
    conn.close()
    print("Done.")

if __name__ == '__main__':
    main()
