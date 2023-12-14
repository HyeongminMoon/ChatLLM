import concurrent.futures

import requests
import json


def download_single(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(url, headers=headers, timeout=5)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception("Failed to download URL")


def download_urls(urls, threads=1):
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        for url in urls:
            future = executor.submit(download_single, url)
            futures.append(future)

        results = []
        results_urls = []
        i = 0
        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
                results_urls.append(urls[idx])
                i += 1
                # yield f"{i}/{len(urls)}", results
            except Exception:
                pass

        return results, results_urls


def get_urls_from_search_engine(
    query, search_engine="http://127.0.0.1:8080/", max_urls=3
):
    query_url = f"{search_engine}?q={query}&format=json"

    response = requests.get(query_url)
    data = json.loads(response.text)

    urls = []
    # sorted_result = sorted(data['results'], key=lambda x:('google' in x['engines'], x['score']), reverse=True)
    sorted_result = sorted(data["results"], key=lambda x: x["score"], reverse=True)
    for result in sorted_result[:max_urls]:
        if result["score"] >= 1.0:
            urls.append(result["url"])

    return urls


def get_contents_from_search_engine(
    query, search_engine="http://127.0.0.1:8080/", max_urls=3
):
    query_url = f"{search_engine}?q={query}&format=json"

    response = requests.get(query_url)
    data = json.loads(response.text)

    urls = []
    search_sentences = []
    for search_result in data["results"][:max_urls]:
        search_sentences.append(
            (f"{search_result['title']}\n" f"{search_result['content']}")
        )
        urls.append(search_result["url"])

    return "\n\n".join(search_sentences), urls
