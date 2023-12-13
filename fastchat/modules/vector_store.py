import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from bs4 import BeautifulSoup

import requests
import json
from trafilatura import fetch_url, extract

from fastchat.modules.embedder_adapter import Embedder, get_embedder
from fastchat.modules.download_url import download_urls, get_urls_from_search_engine, get_contents_from_search_engine

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

class ChromaCollector():
    def __init__(self, embedder_name: str = "ddobokki/klue-roberta-base-nli-sts-ko-en"):
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.embedder = get_embedder(embedder_name)
        self.collection = self.chroma_client.create_collection(name="context", embedding_function=self.embedder.embed)
        self.ids = []

    def add(self, texts: list[str]):
        if len(texts) == 0:
            return

        last_id = int(self.ids[-1][2:]) if self.ids else -1
        new_ids = [f"id{i+last_id+1}" for i in range(len(texts))]
        self.ids += new_ids
        self.collection.add(documents=texts, ids=new_ids)

    def get_documents_ids_distances(self, search_strings: list[str], n_results: int):
        n_results = min(len(self.ids), n_results)
        if n_results == 0:
            return [], [], []

        result = self.collection.query(query_texts=search_strings, n_results=n_results, include=['documents', 'distances'])
        documents = result['documents'][0]
        ids = list(map(lambda x: int(x[2:]), result['ids'][0]))
        distances = result['distances'][0]
        return documents, ids, distances

    # Get chunks by similarity and then sort by insertion order
    def get_sorted(self, search_strings: list[str], n_results: int) -> list[str]:
        documents, ids, distances = self.get_documents_ids_distances(search_strings, n_results)
        return [x for _, x in sorted(zip(ids, documents))]

    # Multiply distance by factor within [0, time_weight] where more recent is lower
    def apply_time_weight_to_distances(self, ids: list[int], distances: list[float], time_weight: float = 1.0) -> list[float]:
        if len(self.ids) <= 1:
            return distances.copy()

        return [distance * (1 - _id / (len(self.ids) - 1) * time_weight) for _id, distance in zip(ids, distances)]

    def clear(self):
        self.collection.delete(ids=self.ids)
        self.ids = []

def add_chunks_to_collector(chunks, collector):
    # collector.clear()
    collector.add(chunks)
    
def feed_data_into_collector(collector, corpus, chunk_len=400, chunk_sep=''):
    chunk_len = int(chunk_len)
    chunk_sep = chunk_sep.replace(r'\n', '\n')

    if chunk_sep:
        data_chunks = corpus.split(chunk_sep)
        data_chunks = [[data_chunk[i:i + chunk_len] for i in range(0, len(data_chunk), chunk_len)] for data_chunk in data_chunks]
        data_chunks = [x for y in data_chunks for x in y]
    else:
        data_chunks = [corpus[i:i + chunk_len] for i in range(0, len(corpus), chunk_len)]

    add_chunks_to_collector(data_chunks, collector)

def feed_file_into_collector(collector, file, chunk_len=400, chunk_sep=''):
    # text = file.decode('utf-8')
    if file.lower().endswith('.pdf'):
        reader = PdfReader(file)
        # print(len(reader.pages))
        text = []
        for page in reader.pages:
            text.append(page.extract_text())
        feed_data_into_collector(collector, '\n'.join(text), chunk_len, chunk_sep)
    elif file.lower().endswith('.docx'):
        import docx2txt

        text = docx2txt.process(file) #, "../langchain/images/"
        feed_data_into_collector(collector, text, chunk_len, chunk_sep)
    else:
        # try:
        with open(file, 'r') as f:
            text = f.read()
        feed_data_into_collector(collector, text, chunk_len, chunk_sep)

def feed_urls_into_collector(collector, urls, num_workers=5, chunk_len=400, chunk_sep='', strict=False):
    for url in urls:
        response = requests.get(url, headers=headers, timeout=3)
        if response.status_code == 200:
            downloaded = response.content

        extracted = extract(downloaded)
        
        if extracted:
            feed_data_into_collector(collector, extracted, chunk_len, chunk_sep)
        
def feed_webquery_into_collector(collector, query, num_workers=5, chunk_len=400, chunk_sep='', search_engine="http://127.0.0.1:8080/", num_urls=3):
    urls = get_urls_from_search_engine(query, search_engine, num_urls)
    feed_urls_into_collector(collector, urls, num_workers, chunk_len, chunk_sep)
    
def get_websearch_result(collector, query, max_urls=3, chunk_len=300, search_engine="http://127.0.0.1:8080/", num_urls=3):
    query_url = f"{search_engine}?q={query}&format=json"

    response = requests.get(query_url)
    data = json.loads(response.text)
    
    search_data = [
        {
            "idx": idx,
            "url": _data["url"],
            "title": _data["title"],
            "snippet": _data["content"],
            "raw_data": [],
        } for idx, _data in enumerate(data['results'][:max_urls * 2])
    ]
    
    new_ids = []
    data_chunks = []
    total_cnt = 0
    valid_cnt = 0
    for _data in search_data:
        # downloaded = fetch_url(_data["url"])
        total_cnt += 1
        yield f"*Searching {query}... [{total_cnt}] {_data['title']}...*", [], True
        try:
            response = requests.get(_data["url"], headers=headers, timeout=3)
            if response.status_code == 200:
                downloaded = response.content
            else:
                continue
        except:
            continue
        
        extracted = extract(downloaded)
        if extracted:
            temp_chucks = [extracted[i:i + chunk_len] for i in range(0, len(extracted), chunk_len)]
            data_chunks += temp_chucks
            new_ids += [f"idx{_data['idx']}_{i}" for i in range(len(temp_chucks))]
            valid_cnt += 1
            
        if valid_cnt >= max_urls:
            break
            
    if len(data_chunks) == 0:
        yield "There is no valid context. Please ask again.", [], False
        return

    collection = collector.collection
    
    collection.add(documents=data_chunks, ids=new_ids)
    try:
        result = collection.query(query_texts=[query], n_results=5, include=['documents', 'distances'])
    except:
        yield "There is no valid context. Please ask again.", [], False
        return
    
    for ri, _idx in enumerate(result['ids'][0]):
        search_idx = int(_idx[3:].split('_')[0])
        search_data[search_idx]['raw_data'].append(result['documents'][0][ri])
        
    search_sentences = []
    search_urls = []
    for _data in search_data:
        if _data['raw_data']:
            raw_data = '\n'.join(_data['raw_data'])
            search_sentences.append((
                f"{_data['title']}\n"
                f"{_data['snippet']}"
                f"{raw_data}"
            ) + '\n')
            search_urls.append(_data["url"])
    
    yield '\n\n'.join(search_sentences), search_urls, False
    return 