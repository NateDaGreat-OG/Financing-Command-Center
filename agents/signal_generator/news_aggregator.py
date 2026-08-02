from collections import defaultdict

def aggregate_by_company(headlines, ticker_list):
    company_news = defaultdict(list)

    for h in headlines:
        title = h["title"].upper()
        for ticker in ticker_list:
            if ticker in title:
                company_news[ticker].append(h)

    return company_news
