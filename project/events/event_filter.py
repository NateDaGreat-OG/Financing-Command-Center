from datetime import datetime, timedelta

def filter_next_3_months(events):
    cutoff = datetime.utcnow() + timedelta(days=90)
    filtered = []

    for e in events:
        if not e["published"]:
            continue

        try:
            pub_date = datetime.strptime(e["published"], "%a, %d %b %Y %H:%M:%S %Z")
        except:
            continue

        if pub_date <= cutoff:
            filtered.append(e)

    return filtered
