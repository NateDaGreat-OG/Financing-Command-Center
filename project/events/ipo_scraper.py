import requests
from bs4 import BeautifulSoup
from massive import RESTClient
def fetch_massive_ipos(limit=10):
    try:
        client = RESTClient(app.config["MASSIVE_API_KEY"])
        ipos = client.vx.list_ipos(order="desc", limit=limit, sort="listing_date")
        return list(ipos)
    except Exception as e:
        return {"error": str(e)}

import requests
from bs4 import BeautifulSoup

def fetch_benzinga_ipos():
    url = "https://www.benzinga.com/money/ipos"
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        ipos = []
        rows = soup.select("table tbody tr")

        for row in rows:
            cols = [c.get_text(strip=True) for c in row.find_all("td")]
            if len(cols) >= 3:
                ipos.append({
                    "company": cols[0],
                    "symbol": cols[1],
                    "expected_date": cols[2]
                })

        return ipos

    except Exception as e:
        return {"error": str(e)}

def fetch_marketbeat_ipos():
    url = "https://www.marketbeat.com/ipos/"
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        ipos = []
        rows = soup.select("table tbody tr")

        for row in rows:
            cols = [c.get_text(strip=True) for c in row.find_all("td")]
            if len(cols) >= 4:
                ipos.append({
                    "company": cols[0],
                    "symbol": cols[1],
                    "price_range": cols[2],
                    "expected_date": cols[3]
                })

        return ipos

    except Exception as e:
        return {"error": str(e)}
