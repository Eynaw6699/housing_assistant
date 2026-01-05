import os
import shutil
import argparse
import asyncio
import aiohttp
import time
import base64
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup, Tag
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import hashlib
import json
import config
from utils import setup_logger
from ingestion_state import IngestionStateManager

logger = setup_logger("ingest")

STATE_FILE = os.path.join(config.DATA_DIR, "ingestion_state.json")


async def process_images_with_vlm(session: aiohttp.ClientSession, element: Tag, base_url: str):
    """
    Finds <img> tags within the specified element, downloads valid images, 
    and uses a VLM to generate descriptions.
    """
    if not element:
        return

    images = element.find_all('img')
    if not images:
        return

    logger.info(f"Scanning {len(images)} images in content area...")
    
    encoded_images = []
    # Filtering Logic
    valid_imgs = []
    
    for img in images:
        src = img.get('src', '')
        alt = img.get('alt', '').lower()
        class_list = img.get('class', [])
        
        # Skip SVGs (usually icons)
        if src.lower().endswith('.svg'):
            continue
            
        # Skip common keywords in alt/class/src
        ignore_terms = ['icon', 'logo', 'banner', 'spacer', 'button', 'arrow', '装饰']
        if any(term in src.lower() for term in ignore_terms) or \
           any(term in alt for term in ignore_terms) or \
           any(term in str(c).lower() for term in class_list for c in class_list):
            continue

        # Skip small images based on attributes (if available)
        width = img.get('width')
        height = img.get('height')
        if width and width.isdigit() and int(width) < 100:
            continue
        if height and height.isdigit() and int(height) < 100:
            continue
            
        valid_imgs.append(img)
        
    logger.info(f"Found {len(valid_imgs)} potentially relevant images to process.")

    for img in valid_imgs:
        src = img.get('src')
        if not src:
            logger.warning("Found image without src attribute, skipping.")
            continue
            
        full_url = urljoin(base_url, src)
        logger.info(f"Processing image url: {full_url}")
        
        try:
            # Download Image
            async with session.get(full_url, timeout=10) as resp:
                if resp.status == 200:
                    img_data = await resp.read()
                    
                    # Size check (skip images < 5KB)
                    if len(img_data) < 5 * 1024: 
                        continue

                    # Convert to base64
                    b64_data = base64.b64encode(img_data).decode('utf-8')
                    
                    # Output user feedback
                    logger.info(f"Analyzing image: {src.split('/')[-1]}")

                    endpoint = f"{config.OLLAMA_BASE_URL}/api/generate"
                    payload = {
                        "model": config.VLM_MODEL_ID,
                        "prompt": "Describe this image in detail. If it is a flowchart, table, or infographic, transcribe the key information or process steps. Be concise.",
                        "images": [b64_data],
                        "stream": False
                    }
                    
                    async with session.post(endpoint, json=payload) as model_resp:
                        if model_resp.status == 200:
                            res_json = await model_resp.json()
                            description = res_json.get("response", "").strip()
                            
                            if description:
                                description_text = f"\n\n> **[Image Description]**: {description}\n\n"
                                
                                # Robustly create new tag
                                new_tag = Tag(name="p")
                                new_tag.string = description_text
                                
                                img.replace_with(new_tag)
                            else:
                                _use_alt_fallback(img)
                        else:
                            # 404 likely means model not found
                            if model_resp.status == 404:
                                logger.error(f"Model '{config.VLM_MODEL_ID}' not found on Ollama. Please run `ollama pull {config.VLM_MODEL_ID}`.")
                                return # Stop processing other images to avoid spamming errors
                            
                            logger.warning(f"VLM failed (Status {model_resp.status}).")
                            _use_alt_fallback(img)
                else:
                    pass # Silent fail for download
        except Exception as e:
            logger.warning(f"Error processing image {full_url}: {e}")
            # _use_alt_fallback(img) # Optional: keep original image or alt

def _use_alt_fallback(img_tag):
    """Helper to replace img with alt text if available."""
    alt = img_tag.get('alt')
    
    # Only keep alt if it looks meaningful (more than 2 words or long enough)
    if alt and len(alt) > 5:
        # Just insert text, don't break structure too much
        img_tag.replace_with(f" [Image: {alt}] ")
    else:
        # Remove empty/useless images to clean up text
        img_tag.decompose()

def html_table_to_markdown(table: Tag) -> str:
    """
    Converts a BeautifulSoup table tag to a Markdown table string.
    """
    rows = []
    # Find all rows
    for tr in table.find_all('tr'):
        cells = []
        # Handle both th and td
        for cell in tr.find_all(['th', 'td']):
            text = cell.get_text(separator=" ", strip=True)
            cells.append(text)
        rows.append(cells)

    if not rows:
        return ""

    # Normalize column count (sometimes rows have different lengths)
    max_cols = max(len(r) for r in rows)
    if max_cols == 0:
        return ""

    markdown_lines = []
    
    # Header
    header_row = rows[0]
    # Pad header if needed (rare but possible)
    header_row += [""] * (max_cols - len(header_row))
    markdown_lines.append("| " + " | ".join(header_row) + " |")
    
    # Separator
    markdown_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    
    # Data rows
    for row in rows[1:]:
        row += [""] * (max_cols - len(row))
        markdown_lines.append("| " + " | ".join(row) + " |")
        
    return "\n" + "\n".join(markdown_lines) + "\n"

def get_main_content(soup: BeautifulSoup) -> Tag:
    """Isolates the main content area."""
    # HDB Specific Selectors
    main = soup.find('div', class_='main-content') or \
           soup.find('div', id='content') or \
           soup.find('div', class_='layout__content') or \
           soup.find('main') or \
           soup
    return main

def extract_content_with_tables(element: Tag) -> str:
    """
    Extracts content from a Tag, converting HTML tables to Markdown.
    """
    if not element:
        return ""

    # Remove unwanted elements
    for script in element(["script", "style", "nav", "footer", "header", "noscript", "iframe", "form"]):
        script.decompose()
    
    # Convert tables to markdown
    for table in element.find_all('table'):
        markdown_table = html_table_to_markdown(table)
        table.replace_with(f"\n{markdown_table}\n")

    text = element.get_text(separator='\n')
    
    # Basic clean up of multiple newlines
    lines = (line.strip() for line in text.splitlines())
    cleaned_text = '\n'.join(line for line in lines if line)
    
    return cleaned_text

def extract_metadata(url: str, soup: BeautifulSoup) -> dict:
    parsed_url = urlparse(url)
    path_segments = [p for p in parsed_url.path.split('/') if p]
    
    category = path_segments[-2] if len(path_segments) > 1 else "general"
    subject = path_segments[-1] if path_segments else "index"
    
    title = soup.title.string.strip() if soup.title else subject
    
    return {
        "source": url,
        "category": category,
        "subject": subject,
        "title": title
    }

async def fetch_url(session: aiohttp.ClientSession, url: str, state_manager: IngestionStateManager) -> Document:
    """
    Fetches a URL, checking for modifications via headers or content hash.
    Returns a Document object if new/modified, or None if unchanged.
    """
    logger.info(f"Checking {url}...")
    
    # Check HTTP Headers (HEAD request)
    try:
        async with session.head(url, headers={'User-Agent': config.USER_AGENT}, timeout=5) as resp:
            if resp.status == 200:
                last_modified = resp.headers.get('Last-Modified')
                etag = resp.headers.get('ETag')
                
                # If headers indicate no change, we might skip, but some servers are unreliable.
                # However, if state_manager says "not modified" based on these, we trust it to save bandwidth.
                if not state_manager.is_modified(url, new_last_modified=last_modified, new_etag=etag):
                    logger.info(f"Skipping {url} (Headers indicate no change).")
                    return None
            else:
                pass # HEAD failed, proceed to GET
    except Exception as e:
        logger.warning(f"HEAD request failed for {url}: {e}")

    # Fetch Content
    details_log = {"url": url, "status": "failed", "attempts": 0}
    
    for attempt in range(config.MAX_RETRIES):
        details_log["attempts"] = attempt + 1
        try:
            async with session.get(url, headers={'User-Agent': config.USER_AGENT}, timeout=15) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    # 3. Check Content Hash
                    # We compute hash of raw HTML to detect any change.
                    content_hash = hashlib.md5(html.encode('utf-8')).hexdigest()
                    
                    if not state_manager.is_modified(url, new_content_hash=content_hash):
                         logger.info(f"Skipping {url} (Content hash unchanged).")
                         return None

                    soup = BeautifulSoup(html, 'html.parser')
                    content_area = get_main_content(soup)

                    # Process Images (Vision-Language) - Modifies content_area in-place
                    await process_images_with_vlm(session, content_area, url)
                    
                    # Extract Text & Tables
                    text = extract_content_with_tables(content_area)
                    metadata = extract_metadata(url, soup)
                    
                    # Update State (Save the new hash/headers)
                    last_modified = response.headers.get('Last-Modified')
                    etag = response.headers.get('ETag')
                    state_manager.update_url_state(url, last_modified=last_modified, etag=etag, content_hash=content_hash)
                    
                    details_log["status"] = "success"
                    return Document(page_content=text, metadata=metadata)
                else:
                    logger.warning(f"Failed to fetch {url} (Status: {response.status})")
        except Exception as e:
            logger.warning(f"Error fetching {url}: {e}")
        
        sleep_time = config.BACKOFF_FACTOR ** attempt
        await asyncio.sleep(sleep_time)

    logger.error(f"Skipped {url} after {config.MAX_RETRIES} attempts.")
    return None

async def ingest_documents(state_manager: IngestionStateManager):
    logger.info("Starting Document Ingestion...")
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url, state_manager) for url in config.URLS]
        results = await tqdm_asyncio.gather(*tasks, desc="Fetching URLs")
    
    documents = [doc for doc in results if doc is not None]
    
    if not documents:
        logger.error("No documents were successfully ingested.")
        return []
    
    logger.info(f"Successfully collected {len(documents)} documents.")
    return documents

def chunk_and_enrich_documents(documents):
    logger.info("Initializing Semantic Chunker...")
    embedding_function = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={'trust_remote_code': True}
    )
    text_splitter = SemanticChunker(embedding_function, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=80)
    
    logger.info("Splitting documents (Semantic Chunking)...")
    splits = text_splitter.split_documents(documents)
    logger.info(f"Generated {len(splits)} raw chunks from {len(documents)} new/modified docs.")

    if not splits:
        return []

    # Check for duplicates in Chroma to avoid re-enriching known chunks.
    # We use a hash of the content as the ID.
    
    # Initialize Chroma for checking existence
    vectorstore = Chroma(
        persist_directory=config.CHROMA_PATH,
        embedding_function=embedding_function
    )
    
    unique_splits = []
    seen_ids = set()
    skipped_count = 0
    
    # We assign IDs now
    for chunk in splits:
        # Create a deterministic ID based on content
        chunk_content_hash = hashlib.md5(chunk.page_content.encode('utf-8')).hexdigest()
        
        #Check if we've already seen this ID in this batch (internal duplication)
        if chunk_content_hash in seen_ids:
            skipped_count += 1
            continue

        chunk.metadata["id"] = chunk_content_hash 
        chunk.id = chunk_content_hash
        
        # 2. Check if exists in DB
        # .get(ids=...) returns dict with 'ids' list. If list is empty, ID not found.
        existing = vectorstore.get(ids=[chunk_content_hash])
        if existing and existing['ids']:
            skipped_count += 1
            continue
        else:
            seen_ids.add(chunk_content_hash)
            unique_splits.append(chunk)
            
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} chunks that already exist in DB.")
        
    if not unique_splits:
        logger.info("No new chunks to process after deduplication.")
        return []

    # Context Enrichment for remaining unique chunks
    logger.info(f"Enriching {len(unique_splits)} new chunks with context...")
    try:
        llm = ChatOllama(model=config.MODEL_ID, temperature=0.7)
    except Exception as e:
        logger.error(f"Failed to initialize Ollama: {e}")
        return unique_splits # Return raw if LLM fails

    context_prompt = ChatPromptTemplate.from_template(config.CONTEXT_ENRICHMENT_PROMPT_TEMPLATE)
    chain = context_prompt | llm | StrOutputParser()

    enriched_count = 0
    
    for i, chunk in enumerate(tqdm(unique_splits, desc="Enriching Chunks")):
        title = chunk.metadata.get("title", "Unknown Title")
        category = chunk.metadata.get("category", "")
        subject = chunk.metadata.get("subject", "")
        section = f"{category} - {subject}"
        original_content = chunk.page_content

        explanation = ""
        for attempt in range(2): 
            try:
                explanation = chain.invoke({
                    "title": title,
                    "section": section,
                    "content": original_content
                })
                break
            except Exception as e:
                 time.sleep(1)
        
        if explanation:
            new_content = f"Document Context: {title}\nSection Context: {section}\nContextual Summary: {explanation}\n\n{original_content}"
            chunk.page_content = new_content
            enriched_count += 1
    
    logger.info(f"Enrichment complete. enriched {enriched_count}/{len(unique_splits)} chunks.")
    return unique_splits

def main():
    parser = argparse.ArgumentParser(description="Ingest HDB data.")
    parser.add_argument("--clean", action="store_true", help="Wipe the existing vector database before ingestion.")
    args = parser.parse_args()

    # Clean DB if requested
    if args.clean:
        if os.path.exists(config.CHROMA_PATH):
            logger.info(f"Cleaning existing vector store at {config.CHROMA_PATH}...")
            try:
                shutil.rmtree(config.CHROMA_PATH)
            except Exception as e:
                logger.error(f"Error cleaning {config.CHROMA_PATH}: {e}")
        
        # Also clean the state file
        if os.path.exists(STATE_FILE):
             logger.info(f"Removing ingestion state file at {STATE_FILE}...")
             try:
                 os.remove(STATE_FILE)
             except Exception as e:
                 logger.error(f"Error cleaning state file: {e}")

    # Initialize State Manager
    state_manager = IngestionStateManager(STATE_FILE)

    # Ingest
    documents = asyncio.run(ingest_documents(state_manager))
    
    if not documents:
        logger.info("No documents to ingest.")
        return

    # Chunk & Enrich
    splits = chunk_and_enrich_documents(documents)
    
    if not splits:
        logger.info("No new chunks to index.")
        return

    logger.info(f"Ingesting {len(splits)} new chunks into Chroma at {config.CHROMA_PATH}...")
    try:
        embedding_function = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={'trust_remote_code': True}
        )
        # Pass IDs explicitly
        ids = [chunk.id for chunk in splits]
        
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_function,
            ids=ids, # Crucial for deduplication consistency
            persist_directory=config.CHROMA_PATH
        )
        logger.info("Ingestion to ChromaDB complete.")
    except Exception as e:
        logger.error(f"Failed to persist to ChromaDB: {e}")

if __name__ == "__main__":
    main()
