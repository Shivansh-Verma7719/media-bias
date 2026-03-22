import pandas as pd
import re
from tqdm import tqdm
import argparse

def load_company_names(company_file=None):
    """
    Load company names. Replace this with your actual source.
    """
    if company_file:
        with open(company_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    # Default list based on the project's S&P 500 and Nifty 50 contexts
    # These are the aliases from your mtsc pipeline + Nifty approximations
    return [
        "Apple", "Airbnb", "Bank of America", "BofA", "FedEx", "Federal Express",
        "General Motors", "Goldman Sachs", "Goldman", "Intel", "McDonald's", 
        "McDonald", "Morgan Stanley", "Microsoft", "Nasdaq", "Netflix", "Nike", 
        "Oracle", "Progressive", "Starbucks", "AT&T", "Tesla", "Uber", "Visa", 
        "Wells Fargo", "Walmart", "Wal-Mart", "Infosys", "TCS", "Reliance",
        "HDFC", "ICICI", "SBI"
    ]

def main():
    parser = argparse.ArgumentParser(description="Stage 0: Pre-filter article titles by company names.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input CSV containing raw titles")
    parser.add_argument("--output", "-o", type=str, default="00_filtered_titles.csv", help="Output CSV for filtered titles")
    parser.add_argument("--companies", "-c", type=str, default=None, help="Text file with one company name per line")
    parser.add_argument("--title_col", type=str, default="title", help="Name of the title column")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Original dataset size: {len(df)} rows")
    
    if args.title_col not in df.columns:
        raise ValueError(f"Column '{args.title_col}' not found in input CSV. Available columns: {df.columns.tolist()}")
    
    company_names = load_company_names(args.companies)
    print(f"Loaded {len(company_names)} company names for filtering.")
    
    # Compile regex pattern with word boundaries (case-insensitive)
    # Sort by length descending to match longer multi-word names first
    sorted_companies = sorted(company_names, key=len, reverse=True)
    pattern_str = r'\b(?:' + '|'.join(map(re.escape, sorted_companies)) + r')\b'
    pattern = re.compile(pattern_str, re.IGNORECASE)
    
    # Register tqdm for pandas
    tqdm.pandas(desc="Filtering titles")
    
    # Drop NAs
    df = df.dropna(subset=[args.title_col])
    
    # Apply regex search
    mask = df[args.title_col].astype(str).progress_apply(lambda x: bool(pattern.search(x)))
    filtered_df = df[mask].copy()
    
    print(f"Filtered dataset size: {len(filtered_df)} rows")
    
    filtered_df.to_csv(args.output, index=False)
    print(f"Saved filtered data to {args.output}")

if __name__ == "__main__":
    main()
