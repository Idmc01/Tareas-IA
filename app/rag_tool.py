import os
import json
import numpy as np
import pandas as pd
import faiss
from typing import List, Dict, Tuple
from openai import OpenAI
from pathlib import Path

#detecta la raíz del proyecto y las carpetas data/ y vectordb/
def get_project_paths() -> Tuple[str, str, str]:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "vectordb").exists() or (parent / "data").exists():
            return str(parent), str(parent / "data"), str(parent / "vectordb")
    fallback = here.parent.parent.parent
    return str(fallback), str(fallback / "data"), str(fallback / "vectordb")


#esto carga el indice FAISS y el DataFrame (ya sea sliding o recursive)
def load_index_and_df(strategy: str):
    base_dir, data_dir, vectordb_dir = get_project_paths()
    if strategy == "sliding":
        index_path = os.path.join(vectordb_dir, "faiss_index_sliding.bin")
        df_path = os.path.join(data_dir, "chunks_sliding_v1.parquet")
        text_col = "chunk"
        meta_name = "faiss_index_sliding_metadata.json"
    else:
        index_path = os.path.join(vectordb_dir, "faiss_index_recursive.bin")
        df_path = os.path.join(data_dir, "chunks_recursive_v1.parquet")
        text_col = "chunk_text"
        meta_name = "faiss_index_recursive_metadata.json"

    index = faiss.read_index(index_path)
    try:
        df = pd.read_parquet(df_path, engine="pyarrow")
    except Exception:
        df = pd.read_parquet(df_path, engine="fastparquet")

    metadata = None
    meta_path = os.path.join(vectordb_dir, meta_name)
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        pass

    return index, df, text_col, metadata

#generar embedding normalizado para la query
def embed_query(client: OpenAI, text: str) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=[text])
    emb = np.array([resp.data[0].embedding], dtype=np.float32)
    # normalizar L2 (RECORDAR QUE los indices fueron creados con vectores normalizados)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norms, 1e-12, None)
    return emb


#funcion que busca top-k chunks en el indice seleccionado -> construye contexto 
#return (resultados, contexto_etiquetado)
def retrieve_from_index(
    client: OpenAI,
    query: str,
    strategy: str,
    k: int = 5,
) -> Tuple[List[Dict], str]:
    index, df, text_col, metadata = load_index_and_df(strategy)
    q_emb = embed_query(client, query)
    distances, indices = index.search(q_emb, k)

    resultados: List[Dict] = []
    etiquetas_contexto = []

    for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
        if idx == -1:
            continue
        row = df.iloc[int(idx)]
        fuente = row.get("fuente", "")
        autor = row.get("autor", "")
        semana = row.get("semana", "")
        fecha = row.get("fecha", "")
        titulo = row.get("titulo", "")
        texto = str(row.get(text_col, ""))
        preview = (texto[:300] + "...") if len(texto) > 300 else texto

        resultados.append({
            "rank": rank,
            "score": float(score),
            "fuente": fuente,
            "autor": autor,
            "semana": semana,
            "fecha": fecha,
            "titulo": titulo,
            "texto": texto,
            "preview": preview,
            "row_index": int(idx),
        })
        etiquetas_contexto.append(f"[{rank}] Fuente: {fuente} | Autor: {autor} | Semana: {semana} | Título: {titulo}\n{texto}\n")

    contexto = "\n\n".join(etiquetas_contexto)
    return resultados, contexto
