from typing import List, Dict
from ddgs import DDGS

__all__ = ["search_web"]


# realiza una búsqueda  en DuckDuckGo y devuelve resultados normalizados
def search_web(query: str, limit: int = 5) -> List[Dict[str, str]]:
 
    #comprobar si DDGS está disponible
    if DDGS is None:
        raise RuntimeError(
            "'ddgs' no está instalado"
        )

    # validar parámetros
    if not isinstance(query, str) or not query.strip():
        raise ValueError("la consulta 'query' debe ser una cadena no vacía")

    # limitar el número de resultados
    if not isinstance(limit, int) or limit < 1 or limit >= 11:
        raise ValueError("el parámetro 'limit' debe ser un entero entre 1 y 10")

    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=limit)
    except Exception as e:
        raise RuntimeError(f"Error ejecutando la búsqueda en DuckDuckGo: {e}") from e

    if not results:
        return []

    # normalizar resultados
    out: List[Dict[str, str]] = []
    for r in results:
        out.append({
            "title": r.get("title", ""),
            "snippet": r.get("body", ""),
            "url": r.get("href", "")
        })

    return out