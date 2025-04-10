import streamlit as st # for user interface
import asyncio # for crawling
import os
import ollama
from duckduckgo_search import DDGS
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

# crawler modules:
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.models import CrawlResult

#for vector database (to place data in db):
import chromadb
import tempfile
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

system_prompt = ""

# invoke the llm
def call_llm(prompt: str, with_context: bool = True, context: str | None = None):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"Context: {context}, Question: {prompt}",
        },
    ]

    # if not performing websearch:
    if not with_context:
        messages.pop(0)
        messages[0]["content"] = prompt

    response = ollama.chat(model="llama3.2:1b", stream=True, messages=messages)

    for chunk in response:
        if chunk["done"] is False:           # create chunk if not created
            yield chunk["message"]["content"]
        else:
            break

# func. to return db collection
def get_vector_collection() -> tuple[chromadb.Collection, chromadb.Client]:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_client = chromadb.PersistentClient(
        path="./web-search-llm-db", settings=Settings(anonymized_telemetry=False)
    )

    # database attributes
    return (
        chroma_client.get_or_create_collection(
            name="web_llm",
            embedding_function=ollama_ef,
            metadata={"hnsw:space": "cosine"},
        ),
        chroma_client,
    )

# func. to normalise url/ make it more readable
def normalize_url(url):
    normalized_url = (
        url.replace("https://", "")
        .replace("www.", "")
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
    )
    print("Normalized URL", normalized_url)
    return normalized_url

# func. to add crawling data to database
def add_to_vector_database(results: list[CrawlResult]):
    collection, _ = get_vector_collection()

    for result in results:
        documents, metadatas, ids = [], [], []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""], # to split text in case of large paragraphs(chunk)
        )
        if result.markdown_v2:
            markdown_result = result.markdown_v2.fit_markdown
        else:
            continue

        temp_file = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) # add contents to temporary file
        temp_file.write(markdown_result)
        temp_file.flush() # to clear buffer

        loader = UnstructuredMarkdownLoader(temp_file.name, mode="single") # langchain to load temp file into doc
        docs = loader.load()
        all_splits = text_splitter.split_documents(docs) # split/divide the doc into chunks/parts
        os.unlink(temp_file.name)  # Delete the temporary file

        normalized_url = normalize_url(result.url)

        if all_splits:
            for idx, split in enumerate(all_splits):
                documents.append(split.page_content)
                metadatas.append({"source": result.url})
                ids.append(f"{normalized_url}_{idx}")

            print("Upsert collection: ", id(collection)) # push data to db
            collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )

# func. to crawl multiple webpages
async def crawl_webpages(urls: list[str], prompt: str) -> CrawlResult:
    bm25_filter = BM25ContentFilter(user_query=prompt, bm25_threshold=1.2) # filter web contents
    md_generator = DefaultMarkdownGenerator(content_filter=bm25_filter) # for Markdown (formatting plain text in a simple and readable way.)

    crawler_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        excluded_tags=["nav", "footer", "header", "form", "img", "a"], # exclude html parts
        only_text=True,
        exclude_social_media_links=True,
        keep_data_attributes=False,
        cache_mode=CacheMode.BYPASS, # without cache, fresh crawling
        remove_overlay_elements=True,
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
        page_timeout=20000,  # in milliseconds: 20 seconds
    )
    browser_config = BrowserConfig(headless=True, text_mode=True, light_mode=True) 

    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun_many(urls, config=crawler_config) # method to run multiple urls/websites.
        return results

# for ethical web crawling
def check_robots_txt(urls: list[str]) -> list[str]: # take a bunch of urls as input
    allowed_urls = []

    for url in urls:
        try:
            robots_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}/robot.txt"
            rp = RobotFileParser(robots_url)
            rp.read()

            if rp.can_fetch("*", url): # if url is valid and fetchable
                allowed_urls.append(url)
        except Exception:
            allowed_urls.append(url) # if there is no robots.txt file
    return allowed_urls

# function to display the webpage content and url.
def get_web_urls(search_term: str, num_results: int = 10) -> list[str]: # function to give about 10 results/webpages in a list.
    try:
        discard_urls = ["youtube.com", "britannica.com", "vimeo.com"] # exclude these urls.
        for url in discard_urls:
            search_term += f" -site:{url}" # to discard urls.
        

        results = DDGS().text(search_term, max_results=num_results)
        results = [result["href"] for result in results] # pust url onto list.

        st.write(results)
        return check_robots_txt(results)
    
    except Exception as e:
        error_msg = f"Failed to fetch results: {str(e)}"
        st.error(error_msg)
        print(error_msg)
        st.stop()

async def run():
    st.set_page_config(page_title = "A Chatbot") # page title
    st.header("**A Chatbot.**")
    prompt = st.text_area(
        label = "Put your query here",
        placeholder = "Add your query...",
        label_visibility = 'hidden', # to hide label
    )

    is_web_search = st.toggle("Enable web search?" ,value=False, key="enable_web_search") # invoke the crawler if websearch is activated.
    go = st.button(
        "Go!",
    )

    collection, chroma_client = get_vector_collection() # to perform db operations 

    if prompt and go: # if user press go and activates web search then display urls.
        if is_web_search:
            web_urls = get_web_urls(search_term=prompt)
            if not web_urls: 
                st.write("No results found.")
                st.stop()

            results = await crawl_webpages(urls=web_urls, prompt=prompt) # crawl the webpages and
            add_to_vector_database(results)                              # add the results to DB

            qresults = collection.query(query_texts=[prompt], n_results=10) # query results
            context = qresults.get("documents")[0] # passing context

            chroma_client.delete_collection(
                name="web_llm"
            )  # Delete collection after use

            llm_response = call_llm(
                context=context, prompt=prompt, with_context=is_web_search
            )
            st.write_stream(llm_response)
        
        # if websearch is disabed
        else:
            llm_response = call_llm(prompt=prompt, with_context=is_web_search)
            st.write_stream(llm_response)

if __name__ == "__main__":
    asyncio.run(run())