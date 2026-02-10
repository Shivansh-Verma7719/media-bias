import os
from dotenv import load_dotenv
from supabase import create_client, Client
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_supabase_client = None

def get_supabase_client() -> Client:
    """Singleton to get Supabase client."""
    global _supabase_client
    if _supabase_client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        _supabase_client = create_client(url, key)
    return _supabase_client


def get_domain(url):
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.replace('www.', '')
    except:
        return None


def get_unprocessed_companies(limit=None):
    """
    Fetch companies that haven't been fully processed yet.
    """
    supabase = get_supabase_client()
    try:
        query = supabase.table("companies").select("*").eq("is_processed", False).order("last_updated", nulls_first=True)
        if limit:
            query = query.limit(limit)
        response = query.execute()
        return response.data
    except Exception as e:
        logger.error(f"Error fetching unprocessed companies: {e}")
        return []


def update_company_state(symbol, current_page=None, is_processed=None, last_error=None):
    """
    Update the processing state for a company.
    """
    supabase = get_supabase_client()
    try:
        data = {}
        if current_page is not None:
            data["current_page"] = current_page
        if is_processed is not None:
            data["is_processed"] = is_processed
        if last_error is not None:
            data["last_error"] = last_error
        
        if not data:
            return

        supabase.table("companies").update(data).eq("symbol", symbol).execute()
    except Exception as e:
        logger.error(f"Error updating company state for {symbol}: {e}")


def _resolve_media_outlets(outlet_map_domain_name):
    """
    Ensure media outlets exist and return a map of domain -> id.
    outlet_map_domain_name: dict {domain: name}
    """
    if not outlet_map_domain_name:
        return {}

    supabase = get_supabase_client()
    domains = list(outlet_map_domain_name.keys())
    resolved = {}

    try:
        # 1. Fetch existing
        # Supabase 'in' filter: .in_("domain", domains)
        existing_response = supabase.table("media_outlets").select("id, domain").in_("domain", domains).execute()
        for row in existing_response.data:
            resolved[row["domain"]] = row["id"]

        # 2. Insert missing
        missing_domains = [d for d in domains if d not in resolved]
        if missing_domains:
            to_insert = [
                {"domain": d, "name": outlet_map_domain_name[d], "social_handles": {}}
                for d in missing_domains
            ]
            # Upsert ignoring duplicates just in case of race env
            # on_conflict="domain" is standard if domain is unique constraint
            insert_response = supabase.table("media_outlets").upsert(to_insert, on_conflict="domain", ignore_duplicates=False).execute()
            
            # If insert returns data, use it. If not (e.g. ignore_duplicates?), we might need to fetch again.
            # Usually upsert returns the inserted/updated rows.
            for row in insert_response.data:
                resolved[row["domain"]] = row["id"]
            
            # If we still miss some (e.g. race condition where it existed but we didn't fetch it first time, 
            # and upsert didn't return it because it did nothing?), fetch again.
            if len(resolved) < len(domains):
                 existing_response_2 = supabase.table("media_outlets").select("id, domain").in_("domain", missing_domains).execute()
                 for row in existing_response_2.data:
                    resolved[row["domain"]] = row["id"]

        return resolved
    except Exception as e:
        logger.error(f"Error resolving media outlets: {e}")
        return resolved


def _resolve_companies(company_map_symbol_name):
    """
    Ensure companies exist and return a map of symbol -> id.
    company_map_symbol_name: dict {symbol: name}
    """
    if not company_map_symbol_name:
        return {}

    supabase = get_supabase_client()
    symbols = list(company_map_symbol_name.keys())
    resolved = {}

    try:
        # 1. Fetch existing
        existing_response = supabase.table("companies").select("id, symbol").in_("symbol", symbols).execute()
        for row in existing_response.data:
            resolved[row["symbol"]] = row["id"]

        # 2. Insert missing
        missing_symbols = [s for s in symbols if s not in resolved]
        if missing_symbols:
            to_insert = [
                {"symbol": s, "name": company_map_symbol_name[s]}
                for s in missing_symbols
            ]
            insert_response = supabase.table("companies").upsert(to_insert, on_conflict="symbol", ignore_duplicates=False).execute()
            
            for row in insert_response.data:
                resolved[row["symbol"]] = row["id"]
                
             # Double check
            if len(resolved) < len(symbols):
                 existing_response_2 = supabase.table("companies").select("id, symbol").in_("symbol", missing_symbols).execute()
                 for row in existing_response_2.data:
                    resolved[row["symbol"]] = row["id"]

        return resolved
    except Exception as e:
        logger.error(f"Error resolving companies: {e}")
        return resolved


def insert_articles(articles_data):
    """
    Insert a list of article dictionaries into the database.
    Expects articles_data to contain:
    - title
    - url
    - publish_date (or published_at)
    - media_name (for resolution)
    - stock_symbol (for resolution)
    - company_name (for resolution)
    """
    if not articles_data:
        return

    try:
        # 1. Prepare resolution maps
        domains_to_resolve = {}
        companies_to_resolve = {}
        
        for article in articles_data:
            url = article.get('url')
            domain = get_domain(url)
            if domain:
                domains_to_resolve[domain] = article.get('media_name') or domain
                
            sym = article.get('stock_symbol')
            if sym:
                companies_to_resolve[sym] = article.get('company_name') or sym

        # 2. Resolve IDs
        outlet_map = _resolve_media_outlets(domains_to_resolve)
        company_map = _resolve_companies(companies_to_resolve)

        # 3. Prepare inserts
        records = []
        for article in articles_data:
            url = article.get('url')
            domain = get_domain(url)
            sym = article.get('stock_symbol')
            
            outlet_id = outlet_map.get(domain)
            company_id = company_map.get(sym)
            
            # Map fields to Supabase schema
            # Schema from db_helper: title, content, url, source, published_at, media_outlet_id, company_id
            records.append({
                "title": article.get('title'),
                "url": url,
                "source": article.get('media_name'),
                "published_at": article.get('publish_date'),
                "media_outlet_id": outlet_id,
                "company_id": company_id,
                # Content is initially None/Null as per original pipeline
            })

        if records:
            supabase = get_supabase_client()
            # on_conflict="url" DO NOTHING -> ignore_duplicates=True
            supabase.table("articles").upsert(records, on_conflict="url", ignore_duplicates=True).execute()
            
    except Exception as e:
        logger.error(f"Error inserting articles batch: {e}")
