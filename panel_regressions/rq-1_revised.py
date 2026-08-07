import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from linearmodels.iv.absorbing import AbsorbingLS
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import os
import psycopg2
from dotenv import load_dotenv

# ==========================================
# 0. DATABASE CONNECTION UTILITIES
# ==========================================
def resolve_db_url() -> str:
    load_dotenv()
    db_url = os.getenv("POOLER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("Missing POOLER_DATABASE_URL/DATABASE_URL in environment.")
    return db_url

def fetch_did_data(db_url: str):
    """
    Fetches article bias data and firm exposure from the database.
    """
    query = """
        SELECT 
            a.company_id,
            a.media_outlet_id,
            a.published_at,
            a.pos_score,
            a.neg_score,
            c.exposure
        FROM public.articles_no_title_deduped a
        JOIN public.top_companies c ON a.company_id = c.id
        WHERE a.pos_score IS NOT NULL 
          AND a.neg_score IS NOT NULL
          AND c.exposure IS NOT NULL
    """
    with psycopg2.connect(db_url) as conn:
        df = pd.read_sql(query, conn)
    
    print(f"Fetched {len(df)} rows from the database.")
    return df

# ==========================================
# 1. CUSTOM TRANSFORMERS (FEATURE ENGINEERING)
# ==========================================
class DiDFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer to construct DiD variables from the raw DB schema.
    """
    def __init__(self, exposure_mapping=None):
        # exposure_mapping: dict mapping company_id to continuous exposure shock
        self.exposure_mapping = exposure_mapping or {}

    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_out = X.copy()
        
        # 1. Create Target Variable (Bias Index) if not already aggregated
        # Assuming Outcome Y_it = pos_score - neg_score
        if 'bias_score' not in X_out.columns:
            X_out['bias_score'] = X_out['pos_score'].fillna(0) - X_out['neg_score'].fillna(0)
            
        # 2. Construct Time Fixed Effects (Year-Month)
        X_out['published_at'] = pd.to_datetime(X_out['published_at'])
        X_out['year_month'] = X_out['published_at'].dt.to_period('M').astype(str)
        
        # 3. Construct Post-COVID Indicator (Post_t)
        X_out['post_2020'] = (X_out['published_at'].dt.year >= 2020).astype(int)
        
        # 4. Exposure Metric (Exposure_i)
        # Exposure is now fetched directly from the DB join
        if 'exposure' not in X_out.columns:
            # Fallback if somehow not present, but should be there from fetch_did_data
            X_out['exposure'] = 0.0
            
        # 5. Create Key Interaction Term (Post_t * Exposure_i)
        X_out['post_x_exposure'] = X_out['post_2020'] * X_out['exposure']
        
        # Drop rows with NaNs in critical columns to prevent pipeline failure
        critical_cols = ['bias_score', 'company_id', 'media_outlet_id', 'year_month', 'post_x_exposure']
        X_out = X_out.dropna(subset=critical_cols)
        
        return X_out

# ==========================================
# 2. REGRESSION MODELING
# ==========================================
def run_did_regression(df_engineered):
    """
    Runs the Fixed Effects regression using AbsorbingLS to properly handle 
    high-dimensional fixed effects and compute p-values/standard errors.
    """
    # Dependent variable
    y = df_engineered['bias_score']
    
    # Independent variable(s) of interest
    X = df_engineered[['post_x_exposure']]
    
    # Fixed effects to absorb
    fixed_effects = df_engineered[['company_id', 'media_outlet_id', 'year_month']].astype('category')
    
    # Fit the model
    # Use robust standard errors
    mod = AbsorbingLS(y, X, absorb=fixed_effects)
    res = mod.fit(cov_type='robust')
    
    return res

# ==========================================
# 3. PDF REPORT GENERATOR
# ==========================================
def export_results_to_pdf(res, filepath="results/rq1_did_regression_results.pdf"):
    """
    Exports the full regression summary to a formatted PDF.
    """
    summary_text = res.summary.as_text()
    
    # Format Text
    report_text = (
        f"Continuous Difference-in-Differences (DiD) Results\n"
        f"{'='*60}\n\n"
        f"Model Specification: Y_it = \\alpha_i + \\delta_t + \\beta(Post_t \\times Exposure_i) + \\gamma_j + \\epsilon_{{it}}\n\n"
        f"{summary_text}\n\n"
        f"Interpretation:\n"
        f"The 'post_x_exposure' coefficient represents the DiD estimator (\\beta).\n"
        f"A statistically significant coefficient (p-value < 0.05) indicates that the\n"
        f"COVID-19 shock had a measurable impact on the media bias score for companies\n"
        f"based on their sector exposure, controlling for firm, outlet, and time fixed effects."
    )
    
    # Generate PDF using matplotlib
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.axis('off')
    ax.text(0.01, 0.99, report_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    with PdfPages(filepath) as pdf:
        pdf.savefig(fig)
        plt.close()
    
    print(f"Results successfully saved to {filepath}")

# ==========================================
# 4. EXECUTION (MOCK DATA FOR DEMONSTRATION)
# ==========================================
if __name__ == "__main__":
    try:
        db_url = resolve_db_url()
        print("Connecting to database and fetching data...")
        df = fetch_did_data(db_url)
        
        if df.empty:
            print("No data found in the database. Exiting.")
        else:
            print("Transforming features...")
            engineer = DiDFeatureEngineer()
            df_engineered = engineer.transform(df)
            
            print("Running Fixed Effects regression (this may take a moment)...")
            res = run_did_regression(df_engineered)
            
            print("Generating PDF Report...")
            # Ensure results directory exists
            os.makedirs("results", exist_ok=True)
            export_results_to_pdf(res, filepath="results/rq1_did_regression_results.pdf")
            
    except Exception as e:
        print(f"Error during execution: {e}")