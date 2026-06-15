#!/usr/bin/env python3
import os
import psycopg2
from dotenv import load_dotenv

def resolve_db_url() -> str:
    load_dotenv()
    db_url = os.getenv("POOLER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("Missing POOLER_DATABASE_URL/DATABASE_URL in environment.")
    return db_url

def main():
    # Exposure mapping from sector-wise-exposure.md
    exposure_map = {
        "Energy": -1.00,
        "Financials": -0.49,
        "Industrials": -0.45,
        "Consumer Discretionary": -0.18,
        "Communication Services": 0.20,
        "Information Technology": 0.57,
        "Consumer Staples": 0.76,
        "Health Care": 1.00
    }

    db_url = resolve_db_url()
    
    print(f"Connecting to database...")
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        print("Updating top_companies table...")
        
        updated_count = 0
        for sector, exposure in exposure_map.items():
            # Update exposure weight matching the sector (case-insensitive)
            query = "UPDATE top_companies SET exposure = %s WHERE LOWER(sector) = LOWER(%s)"
            cur.execute(query, (exposure, sector))
            updated_count += cur.rowcount
            print(f"  Sector: {sector} -> Exposure: {exposure} ({cur.rowcount} rows affected)")
        
        conn.commit()
        print(f"\nSuccessfully updated {updated_count} total rows.")
        
    except Exception as e:
        print(f"Error: {e}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()
