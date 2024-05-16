from openai import OpenAI 
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from pprint import pprint

CHUNK_SIZE = 256
CHUNK_OVERLAP = 64
PERSIST_PATH = "./.chroma_db"
COLLECTION_NAME = "langchain"


# CSVファイルからドキュメントを読み込む
loader = CSVLoader(
    file_path="../../app/data/example.csv",
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ["id", "name", "long", "lat", "area", "category"],
    },
    encoding="utf-8",
)

# 環境変数の読み込み
load_dotenv()
api_key = os.getenv("API_KEY")

# 変数docsにドキュメントを読み込む
docs = loader.load()

# OpenAIのテキスト埋め込みモデルを使用
embedding = OpenAIEmbeddings(openai_api_key = api_key ,model="text-embedding-ada-002")

# Chromaデータベースを生成するディレクトリを指定
persist_directory = "../../app/vectordb"

# ドキュメントをChromaデータベースに格納
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    collection_name="faqs",
    persist_directory=persist_directory,
)

query = "一人でゆっくり飲みたいなあ"

# 類似度検索
docs = vectordb.similarity_search_with_relevance_scores(query, k=1)

# ページコンテンツを取得
page_content = docs[0][0].page_content

# 類似度スコアとページ名を取得
similarity = docs[0][1]

# ページ名を取得
name = page_content.split("\nname: ")[1]

pprint(name)
