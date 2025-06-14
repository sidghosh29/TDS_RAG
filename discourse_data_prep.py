import os
import time
import requests
import json
import html2text
from google import genai
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv() 

def getImageDescription(image_path):
   
    client = genai.Client(api_key=os.getenv("GOOGLE_GENAI_API_KEY"))
    my_file = client.files.upload(file=image_path)
    # with open(image_path, "rb") as image_file:
    #     image_data = image_file.read()
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[my_file,"Describe the content of this image in detail, focusing on any text, objects, or relevant features that could help answer questions about it."],
    )
    return response.text

# print(getImageDescription("Archive/example.png"))

def convertTopicJsonToMarkdown(json_path, output_path):
    h = html2text.HTML2Text()
    h.ignore_links = False  # Set to True if you want to remove links
    h.ignore_images = False  # Set to True if you want to remove images
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lines = []
    posts = data.get("post_stream", {}).get("posts", [])
    for post in posts:
        author = post.get("name", "")
        date = post.get("created_at", "")
        content_html = post.get("cooked", "")

        # Parse HTML and process images
        soup = BeautifulSoup(content_html, "html.parser")
        for img in soup.find_all("img"):
            img_url = img.get("src")
            # Skip emoji images
            if "emoji" in (img.get("class") or []) or (img_url and ("/emoji/" in img_url or "/images/emoji/" in img_url)):
                # Replace emoji image with its alt text if available, else remove
                alt_text = img.get("alt") or ""
                img.replace_with(alt_text)
                continue
            # Handle relative URLs
            if img_url and img_url.startswith("/"):
                img_url = "https://discourse.onlinedegree.iitm.ac.in" + img_url
            if img_url:
                try:
                    # Download image to a temp file
                    img_data = requests.get(img_url).content
                    tmp_path = "tmp_image.jpg"
                    with open(tmp_path, "wb") as tmpf:
                        tmpf.write(img_data)
                    # Get description
                    desc = getImageDescription(tmp_path)
                    time.sleep(4)
                    # Replace <img> with description text
                    img.replace_with(f"[Image description: {desc}]")
                    os.remove(tmp_path)
                except Exception as e:
                    img.replace_with("[Image could not be described]")
        # Convert modified HTML to Markdown
        content_md = h.handle(str(soup))
       # content_md = h.handle(content_html)


        lines.append(f"### {author} ({date})\n\n{content_md}\n---\n")
    
    with open(output_path, "w", encoding="utf-8") as out:
        out.writelines(lines)

if __name__== "__main__":
    THREADS_DIR = "discourse_threads"
    OUTPUT_DIR = "discourse_threads_md"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for fname in os.listdir(THREADS_DIR):
        if not fname.endswith(".json"):
            continue
        thread_path = os.path.join(THREADS_DIR, fname)
        outname = fname.replace(".json", ".md")
        outpath = os.path.join(OUTPUT_DIR, outname)
        # Skip if already processed
        if os.path.exists(outpath):
            print(f"Skipping {fname} (already processed)")
            continue
        convertTopicJsonToMarkdown(thread_path, outpath)
        print(f"Converted {fname} -> {outname}")
        # break  # Remove this line to process all files
















