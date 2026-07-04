import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from linearmodels.iv.absorbing import AbsorbingLS
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import psycopg2
from dotenv import load_dotenv

PRE_START = "2015-01-01"
POST_END = "2025-12-31"
POST_START = "2020-01-01"

def resolve_db_url() -> str:
    load_dotenv()
    db_url = os.getenv("POOLER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("Missing POOLER_DATABASE_URL/DATABASE_URL in environment.")
    return db_url

def fetch_did_stock_data(db_url: str):
    """
    Fetches firm stock returns and exposure data for DiD analysis.
    """
    # 1. Fetch universe with exposure
    universe_query = f"""
        SELECT DISTINCT
            a.company_id::text AS company_id,
            c.symbol AS symbol,
            c.exposure AS exposure
        FROM public.articles_no_title_deduped a
        JOIN public.top_companies c ON c.id = a.company_id
        WHERE a.published_at::date BETWEEN '{PRE_START}' AND '{POST_END}'
          AND c.symbol IS NOT NULL
          AND c.exposure IS NOT NULL
    """
    
    with psycopg2.connect(db_url) as conn:
        universe = pd.read_sql(universe_query, conn)
        
    if universe.empty:
        raise RuntimeError("No firms found with exposure and symbols.")
        
    symbols = universe['symbol'].unique().tolist()
    
    # 2. Fetch prices
    price_query = f"""
        SELECT
            ticker AS symbol,
            date::date AS date,
            close::double precision AS close
        FROM public.stock_prices
        WHERE ticker = ANY(%s)
          AND date::date BETWEEN '{PRE_START}' AND '{POST_END}'
          AND close IS NOT NULL
        ORDER BY ticker, date
    """
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(price_query, (symbols,))
            rows = cur.fetchall()
            
    px = pd.DataFrame(rows, columns=["symbol", "date", "close"])
    px["date"] = pd.to_datetime(px["date"])
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px = px.dropna(subset=["close"]).sort_values(["symbol", "date"])

    px["daily_return"] = px.groupby("symbol")["close"].pct_change()
    px = px.dropna(subset=["daily_return"])
    
    # 3. Merge
    df = universe.merge(px, on="symbol", how="inner")
    
    print(f"Fetched {len(df)} daily return observations from the database.")
    return df

class DiDFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_out = X.copy()
        
        # Construct Post-COVID Indicator (Post_t)
        X_out['post_2020'] = (X_out['date'] >= pd.Timestamp(POST_START)).astype(int)
        
        # Create Key Interaction Term (Post_t * Exposure_i)
        X_out['post_x_exposure'] = X_out['post_2020'] * X_out['exposure']
        
        # Drop rows with NaNs in critical columns
        critical_cols = ['daily_return', 'company_id', 'date', 'post_x_exposure']
        X_out = X_out.dropna(subset=critical_cols)
        
        return X_out

def run_did_regression(df_engineered):
    y = df_engineered['daily_return']
    X = df_engineered[['post_x_exposure']]
    
    # Fixed effects to absorb (Firm and Time)
    fixed_effects = df_engineered[['company_id', 'date']].astype('category')
    
    mod = AbsorbingLS(y, X, absorb=fixed_effects)
    res = mod.fit(cov_type='robust')
    
    return res

def export_results_to_pdf(res, filepath="results/rq2_did_regression_results.pdf"):
    summary_text = res.summary.as_text()
    
    report_text = (
        f"Continuous Difference-in-Differences (DiD) Results - Stock Returns\n"
        f"{'='*60}\n\n"
        f"Model Specification: Return_it = \\alpha_i + \\delta_t + \\beta(Post_t \\times Exposure_i) + \\epsilon_{{it}}\n\n"
        f"{summary_text}\n\n"
        f"Interpretation:\n"
        f"The 'post_x_exposure' coefficient represents the DiD estimator (\\beta).\n"
        f"A statistically significant coefficient (p-value < 0.05) indicates that the\n"
        f"COVID-19 shock had a measurable impact on firm daily stock returns based on\n"
        f"their sector exposure, controlling for firm and daily time fixed effects."
    )
    
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.axis('off')
    ax.text(0.01, 0.99, report_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    with PdfPages(filepath) as pdf:
        pdf.savefig(fig)
        plt.close()
    
    print(f"Results successfully saved to {filepath}")

if __name__ == "__main__":
    try:
        db_url = resolve_db_url()
        print("Connecting to database and fetching data...")
        df = fetch_did_stock_data(db_url)
        
        if df.empty:
            print("No data found in the database. Exiting.")
        else:
            print("Transforming features...")
            engineer = DiDFeatureEngineer()
            df_engineered = engineer.transform(df)
            
            print("Running Fixed Effects regression (this may take a moment)...")
            res = run_did_regression(df_engineered)
            
            print("Generating PDF Report...")
            os.makedirs("results", exist_ok=True)
            export_results_to_pdf(res, filepath="results/rq2_did_regression_results.pdf")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during execution: {e}")
