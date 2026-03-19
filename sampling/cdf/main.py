import os
import matplotlib.pyplot as plt
import numpy as np
import psycopg2
from dotenv import load_dotenv

def main():
    # Load .env from base directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(current_dir))
    env_path = os.path.join(base_dir, '.env')
    
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        print(f"Warning: .env file not found at {env_path}")

    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        raise ValueError("DATABASE_URL not found in .env")

    print("Connecting to database...")
    conn = psycopg2.connect(db_url)

    # Use a query that groups by company_id correctly
    query = """
    SELECT company_id, COUNT(id) as article_count
    FROM indian_cos.articles
    WHERE company_id IS NOT NULL
    GROUP BY company_id
    """

    print("Fetching data...")
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        print("No data found!")
        return

    # Extract article counts
    counts = np.array([row[1] for row in rows], dtype=float)
    
    print(f"Loaded {len(counts)} companies.")

    # Calculate CDF
    counts = np.sort(counts)
    n = len(counts)
    cdf = np.arange(1, n + 1)

    # Calculate percentiles
    p25 = np.percentile(counts, 25)
    p50 = np.percentile(counts, 50)
    p75 = np.percentile(counts, 75)
    p90 = np.percentile(counts, 90)
    
    print(f"Percentiles - 25th: {p25}, 50th: {p50}, 75th: {p75}, 90th: {p90}")

    # Plot both linear and log side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('CDF: Article Distribution Across Indian Companies', fontsize=16, fontweight='bold')
    
    colors = {'25th': '#1abc9c', '50th': '#f39c12', '75th': '#e74c3c', '90th': '#9b59b6'}
    line_color = 'royalblue'

    for ax, is_log in [(ax1, False), (ax2, True)]:
        ax.plot(counts, cdf, linestyle='-', color=line_color, linewidth=2)
        ax.fill_between(counts, cdf, color=line_color, alpha=0.2)
        
        ax.set_ylabel('Cumulative Number of Companies', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.7, which='both')
        ax.set_ylim(0, n + 5)
        
        if is_log:
            min_pos = np.min(counts[counts > 0]) if np.any(counts > 0) else 1
            ax.set_xlim(left=min_pos * 0.8)
            ax.set_title('CDF: Log Scale', fontsize=14, fontweight='bold')
            ax.set_xlabel('Number of Articles per Company (log scale)', fontsize=12)
            ax.set_xscale('log')
            
            # Add stats box
            stats_text = (f"Total Companies: {n}\n"
                          f"Total Articles: {int(np.sum(counts)):,}\n"
                          f"Mean: {np.mean(counts):.1f}\n"
                          f"Median: {np.median(counts):.1f}\n"
                          f"Std Dev: {np.std(counts):.1f}\n"
                          f"Min: {int(np.min(counts))}\n"
                          f"Max: {int(np.max(counts)):,}\n"
                          f"Zero-count: {np.sum(counts == 0)}")
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right', bbox=props)
        else:
            ax.set_xlim(left=0)
            ax.set_title('CDF: Linear Scale', fontsize=14, fontweight='bold')
            ax.set_xlabel('Number of Articles per Company', fontsize=12)
            
            # Format x-axis with M and K
            from matplotlib.ticker import FuncFormatter
            def millions_formatter(x, pos):
                if x >= 1e6:
                    return f'{x*1e-6:g}M'
                elif x >= 1e3:
                    return f'{x*1e-3:g}K'
                else:
                    return f'{x:g}'
            ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))

        # Highlight percentiles
        percentiles = [
            (25, p25, int(n * 0.25), colors['25th']),
            (50, p50, int(n * 0.50), colors['50th']),
            (75, p75, int(n * 0.75), colors['75th']),
            (90, p90, int(n * 0.90), colors['90th'])
        ]
        
        for perc, p_val, num_companies, color in percentiles:
            ax.axhline(y=num_companies, color=color, linestyle='--', alpha=0.6)
            if p_val > 0 or not is_log:
                ax.axvline(x=p_val, color=color, linestyle='--', alpha=0.6)
            
            # Formulate text representation
            text_x = p_val * (1.10 if is_log else 1.02)
            if not is_log and p_val == 0:
                text_x = np.max(counts) * 0.02
                
            ax.text(text_x, num_companies + (n * 0.01), 
                    f"{perc}th: {int(p_val)} articles ({num_companies} co.)", 
                    color=color, fontsize=9, alpha=0.9)

    plt.tight_layout()
    output_path = os.path.join(current_dir, 'cdf_articles_by_company.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    main()
