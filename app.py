import ipaddress
import re
import socket
import time
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from IPython.display import clear_output
import requests


def is_safe_url(url: str) -> bool:
    try:
        parsed = urlparse(url)

        if parsed.scheme != "https":
            return False

        hostname = parsed.hostname
        if not hostname:
            return False

        if hostname in ["localhost", "127.0.0.1", "0.0.0.0", "::1"]:
            return False

        ip_list = socket.getaddrinfo(hostname, None)
        for item in ip_list:
            ip_str = item[4][0]
            ip = ipaddress.ip_address(ip_str)

            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
            ):
                return False

        return True
    except Exception:
        return False


def build_url_candidates(keyword: str) -> list[str]:
    cleaned = re.sub(r"[^a-zA-Z0-9\s-]", "", keyword.lower()).strip()
    words = cleaned.split()

    if not words:
        return []

    base_names = ["".join(words), "-".join(words)]
    tlds = ["com", "org", "io", "ai", "co", "net", "dev"]

    all_patterns = []
    for tld in tlds:
        for name in base_names:
            if name:
                all_patterns.append(f"https://{name}.{tld}")
                all_patterns.append(f"https://www.{name}.{tld}")

    return all_patterns


def fetch_and_check_keyword(url: str, search_keyword: str) -> dict:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(
            url, headers=headers, timeout=2.5, allow_redirects=True, stream=True
        )

        if response.status_code != 200:
            return {"success": False, "final_url": None}

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return {"success": False, "final_url": None}

        response.encoding = response.apparent_encoding or "utf-8"
        content_text = response.text

        try:
            soup = BeautifulSoup(content_text, "html.parser")
            page_text = soup.get_text().lower()
        except Exception:
            page_text = content_text.lower()

        target = search_keyword.strip().lower()
        target_words = target.split()

        has_keyword = (target in page_text) or any(
            w in page_text for w in target_words if len(w) > 2
        )

        final_url = response.url.rstrip("/")

        return {
            "success": True,
            "has_keyword": has_keyword,
            "final_url": final_url,
        }

    except Exception:
        return {"success": False, "final_url": None}


user_query = input()

if user_query.strip():
    candidates = build_url_candidates(user_query)

    valid_candidates_list = []
    seen_urls = set()

    for i in range(min(15, len(candidates))):
        target_url = candidates[i]

        clear_output(wait=True)
        print(user_query)
        print(f"{target_url} under verification")

        if not is_safe_url(target_url):
            time.sleep(0.1)
            continue

        result = fetch_and_check_keyword(target_url, user_query)

        if result["success"] and result["has_keyword"]:
            final_url = result["final_url"] or target_url
            if final_url not in seen_urls and target_url not in seen_urls:
                valid_candidates_list.append(target_url)
                seen_urls.add(target_url)
                seen_urls.add(final_url)

        time.sleep(0.1)

    clear_output(wait=True)
    print(user_query)
    print()

    if valid_candidates_list:
        print("I found this.")
        for url in valid_candidates_list:
            print(url)
    else:
        print("Error400.")
