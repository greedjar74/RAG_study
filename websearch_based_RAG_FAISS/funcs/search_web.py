# 웹에서 기업 면접 정보 크롤링을 위한 함수
# 찾은 링크를 반환한다.

import requests

# 웹 검색
def search_web(query, num_results=5, api_key=None):
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": num_results}
    response = requests.post("https://google.serper.dev/search", json=payload, headers=headers)
    return [item["link"] for item in response.json().get("organic", [])]