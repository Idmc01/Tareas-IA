from duckduckgo_search import ddg

def search_web(query: str, limit: int = 5):
    results = ddg(query, max_results=limit)
    if not results:
        return []
    out = []
    for r in results:
        out.append({
            "title": r.get("title") or "",
            "snippet": r.get("body") or r.get("snippet",""),
            "url": r.get("href") or r.get("url","")
        })
    return out