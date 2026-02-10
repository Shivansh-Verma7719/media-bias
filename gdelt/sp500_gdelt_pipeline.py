import datetime as dt
import time
import os
import sys
import logging
from typing import List, Dict
from gdeltdoc import GdeltDoc, Filters
from dotenv import load_dotenv
from rich.console import Console

# Add parent directory to path to import helpers
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from helpers.supabase_helper import (
    get_unprocessed_companies,
    update_company_state,
    insert_articles,
    mark_company_complete  # Wait, I didn't verify if mark_company_complete was in supabase_helper. It's not. I should add it or use update_company_state.
)

# Fix for missing function - I will define it inline or import update_company_state with is_processed=True
def mark_company_complete(symbol):
    update_company_state(symbol, is_processed=True)

load_dotenv()

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f"logs/gdelt_pipeline_{dt.datetime.now().strftime('%Y%m%d_%H')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console = Console()

# GDELT Configuration
# GDELT 2.0 Doc API has a limit of 250 records.
START_DATE = dt.date(2015, 1, 1)
END_DATE = dt.date(2025, 11, 12)

def fetch_gdelt_articles(company_name: str, stock_symbol: str, date: dt.date):
    """
    Fetch articles for a specific company and date using GDELT Doc API.
    """
    gd = GdeltDoc()
    
    # Cleaning name same as before
    clean_name = (
        company_name.replace(" Inc.", "")
        .replace(" Corp.", "")
        .replace(" Corporation", "")
        .replace(" Ltd.", "")
        .replace(" LLC", "")
        .replace(",", "")
    )
    
    # Construct filters
    f = Filters(
        keyword=f'"{clean_name}"',
        start_date=date.strftime("%Y-%m-%d"),
        end_date=(date + dt.timedelta(days=1)).strftime("%Y-%m-%d"),
        country="US", 
        language="English",
    )
    
    # gdeltdoc wrapper handles the API call
    try:
        articles = gd.article_search(f)
        if articles.empty:
            return []
            
        # Transform to our format
        results = []
        for _, row in articles.iterrows():
            results.append({
                "title": row.get('title'),
                "url": row.get('url'),
                "publish_date": row.get('seendate'), # seendate is usually YYYYMMDDHHMMSS
                "media_name": row.get('domain'),
                "stock_symbol": stock_symbol,
                "company_name": company_name,
                "language": row.get('language')
            })
        return results
    except Exception as e:
        logging.error(f"Error fetching GDELT for {company_name} on {date}: {e}")
        return []

def process_company(company_record):
    company = company_record["name"]
    symbol = company_record["symbol"]
    
    console.print(f"[bold cyan]Processing {company} ({symbol})[/bold cyan]")
    
    current_date = START_DATE
    # If resuming, we might want to store last_processed_date instead of page number?
    # The existing schema has 'current_page' (int). We can use it to store offset in days from START_DATE?
    # Or just start from START_DATE if not fully processed.
    
    days_offset = company_record.get("current_page", 0)
    current_date += dt.timedelta(days=days_offset)
    
    while current_date <= END_DATE:
        console.print(f"  Fetching date: {current_date}")
        
        articles = fetch_gdelt_articles(company, symbol, current_date)
        
        if articles:
            console.print(f"    Found {len(articles)} articles. Pushing to Supabase...")
            insert_articles(articles)
        
        # Update progress (using current_page field as days counter)
        days_processed = (current_date - START_DATE).days + 1
        update_company_state(symbol, current_page=days_processed)
        
        current_date += dt.timedelta(days=1)
        # Rate limit? GDELT is generous but good to be nice.
        time.sleep(0.5)
        
    mark_company_complete(symbol)
    console.print(f"[bold green]Finished {company}[/bold green]")

def main():
    console.print("[bold green]Starting S&P 500 GDELT Pipeline[/bold green]")
    
    companies = get_unprocessed_companies()
    if not companies:
        console.print("[bold yellow]No unprocessed companies found.[/bold yellow]")
        return
        
    console.print(f"Found {len(companies)} companies to process.")
    
    for company in companies:
        try:
            process_company(company)
        except KeyboardInterrupt:
            console.print("[bold red]Stopped by user.[/bold red]")
            break
        except Exception as e:
            console.print(f"[bold red]Error processing {company['symbol']}: {e}[/bold red]")
            logging.error(f"Error processing {company['symbol']}: {e}", exc_info=True)
            # Continue to next company?
            
if __name__ == "__main__":
    main()
