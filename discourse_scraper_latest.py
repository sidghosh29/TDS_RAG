import requests
import time
import json
from urllib.parse import quote
import os

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Discourse Configuration
DISCOURSE_SEARCH_URL_BASE = "https://discourse.onlinedegree.iitm.ac.in/search.json"
DISCOURSE_T_TOKEN = os.getenv("DISCOURSE_T_TOKEN")
COOKIES = {
    "_t": DISCOURSE_T_TOKEN
}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    #"User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
}
SEARCH_QUERY_PARAMS = "#courses:tds-kb after:2025-01-01 before:2025-04-15 order:latest"

session = requests.Session()
session.headers.update(HEADERS)
session.cookies.update(COOKIES)

def fetch_all_discourse_pages(search_query):
    """
    Fetches all Discourse search result pages for the given query.
    Returns a list of page JSONs (each page as a dict).
    """
    all_pages = []
    page = 1
    encoded_query = quote(search_query)
    while True:
        current_url = f"{DISCOURSE_SEARCH_URL_BASE}?q={encoded_query}&page={page}"
        print(f"Fetching: {current_url}")
        try:
            response = session.get(current_url)
           # print(response.status_code, response.reason, response.headers)
            response.raise_for_status()
            data = response.json()
            posts_in_page = data.get("posts", [])
            all_pages.append(data)
            print(f"Page {page}: {len(posts_in_page)} posts")
            if not posts_in_page:
                print(f"No posts found on page {page}. Stopping.")
                break
            page += 1
            time.sleep(1)  # Here I am being polite to the server
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break
    return all_pages

if __name__ == "__main__":
    
    if not DISCOURSE_T_TOKEN:
        print("ERROR: DISCOURSE_T_TOKEN is not set. Please check your .env file or environment variables.")
    else:
        print("Initiating Data Scraping from TDS Discourse.")

        all_pages = fetch_all_discourse_pages(SEARCH_QUERY_PARAMS)
        output_file = "tds_discourse.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_pages, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(all_pages)} pages of Discourse data to {output_file}")

        # --- Fetch full thread details for each topic and save in subfolder ---
        threads_folder = "discourse_threads"
        os.makedirs(threads_folder, exist_ok=True)

        # Collect all topic IDs from all pages
        topic_ids = set()
        for page in all_pages:
            topics = page.get("topics", [])
            for topic in topics:
                topic_id = topic.get("id")
                if topic_id:
                    topic_ids.add(topic_id)

        print(f"Found {len(topic_ids)} unique topic IDs. Fetching full thread details...")
        for idx, topic_id in enumerate(topic_ids, 1):
            # First, fetch the first page to get posts_count
            base_thread_url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_id}.json"
            thread_file = os.path.join(threads_folder, f"thread_{topic_id}.json")
            if os.path.exists(thread_file):
                print(f"[{idx}/{len(topic_ids)}] Skipping topic {topic_id} (already downloaded)")
                continue
            try:
                print(f"[{idx}/{len(topic_ids)}] Fetching thread {topic_id} page 1 ...")
                resp = session.get(base_thread_url)
                resp.raise_for_status()
                thread_data = resp.json()
                posts = thread_data.get("post_stream", {}).get("posts", [])
                posts_count = thread_data.get("posts_count", len(posts))
                # Discourse default: 20 posts per page
                posts_per_page = 20
                total_pages = (posts_count + posts_per_page - 1) // posts_per_page
                # Fetch remaining pages if any
                for page_num in range(2, total_pages + 1):
                    page_url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_id}.json?page={page_num}"
                    print(f"[{idx}/{len(topic_ids)}] Fetching thread {topic_id} page {page_num} ...")
                    page_resp = session.get(page_url)
                    page_resp.raise_for_status()
                    page_data = page_resp.json()
                    page_posts = page_data.get("post_stream", {}).get("posts", [])
                    posts.extend(page_posts)
                    time.sleep(1)  # Be polite to the server
                # Combine all posts into the first page's thread_data
                thread_data["post_stream"]["posts"] = posts
                with open(thread_file, "w", encoding="utf-8") as tf:
                    json.dump(thread_data, tf, indent=2, ensure_ascii=False)
                print(f"Saved thread {topic_id} (all {len(posts)} posts) to {thread_file}")
                time.sleep(1)  # Be polite to the server
            except Exception as e:
                print(f"Error fetching thread {topic_id}: {e}")
        print("All threads fetched and saved in 'discourse_threads' folder.")