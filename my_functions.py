from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.utilities import WikipediaAPIWrapper
from ddgs import DDGS
from bs4 import BeautifulSoup
import re


def search_wikipedia(inputs):
    keyword = inputs["keyword"]
    wikipedia = WikipediaAPIWrapper()
    return wikipedia.run(keyword)


def search_duckduckgo(inputs):
    keyword = inputs["keyword"]
    ddgs = DDGS()
    return list(ddgs._search(category="text", query=keyword))


def _parse_page(doc):
    soup = BeautifulSoup(doc.page_content, "html.parser")

    if soup.header:
        soup.header.decompose()
    if soup.footer:
        soup.footer.decompose()

    for tag in soup.select("nav, aside, .sidebar, .ads, .cookie-banner, link, script"):
        tag.decompose()

    text = re.sub(
        r"\n+",
        " ",
        str(soup.get_text()).replace("\xa0", " "),
    )

    doc.page_content = text
    return doc


def retrieve_content(inputs):
    url = inputs["url"]
    html2TextTransformer = Html2TextTransformer()
    loader = AsyncChromiumLoader([url])
    docs = [_parse_page(doc) for doc in loader.load()]
    transformed = html2TextTransformer.transform_documents(docs)
    return transformed[0].page_content.replace("\n", " ")


def save_as_file(inputs):
    output = inputs["output"]
    with open("result.txt", "w", encoding="utf-8") as f:
        f.write(output)
    return {"success": True}


functions_map = {
    "search_wikipedia": search_wikipedia,
    "search_duckduckgo": search_duckduckgo,
    "retrieve_content": retrieve_content,
    "save_as_file": save_as_file,
}

functions = [
    {
        "type": "function",
        "name": "search_wikipedia",
        "description": "Given the keyword returns the search results from Wikipedia",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "The main keyword of the user’s query",
                }
            },
            "required": ["keyword"],
        },
    },
    {
        "type": "function",
        "name": "search_duckduckgo",
        "description": "Given a keyword returns the search results from DuckDuckGo.",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "The main keyword of the user’s query",
                },
            },
            "required": ["keyword"],
        },
    },
    {
        "type": "function",
        "name": "retrieve_content",
        "description": "Given a URL returns the content from the page.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The website URL",
                },
            },
            "required": ["url"],
        },
    },
    {
        "type": "function",
        "name": "save_as_file",
        "description": "Given a output you generated export to a file and returns whether its successful or not",
        "parameters": {
            "type": "object",
            "properties": {
                "output": {
                    "type": "string",
                    "description": "The output you generated",
                },
            },
            "required": ["output"],
        },
    },
]
