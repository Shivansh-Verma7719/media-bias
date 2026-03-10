import datetime as dt
from gdeltdoc import GdeltDoc, Filters
import traceback

def test_gdelt():
    gd = GdeltDoc()
    f = Filters(
        keyword='"Boeing"',
        start_date="2015-11-22",
        end_date="2015-11-23",
        country="US",
        language="en",
    )
    
    print("Searching...")
    try:
        articles = gd.article_search(f)
        print("Success!")
        print(articles)
    except Exception as e:
        print(f"Caught exception: {type(e).__name__}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_gdelt()
