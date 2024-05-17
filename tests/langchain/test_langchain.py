from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from pprint import pprint

CHUNK_SIZE = 256
CHUNK_OVERLAP = 64
PERSIST_PATH = "./.chroma_db"
COLLECTION_NAME = "langchain"


# CSVファイルからドキュメントを読み込む
print("CSVファイルからドキュメントを読み込む")
try:
    loader = CSVLoader(
        file_path="../../app/data/example.csv",
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
            "fieldnames": ["id", "name", "long", "lat", "area", "category"],
        },
        encoding="utf-8",
    )

    # 変数docsにドキュメントを読み込む
    docs = loader.load()

except Exception as e:
    print(f"エラー: {e}")

print("埋め込み表現生成用モデルをHugging Face Hubから取得")
# 埋め込み表現生成用モデルをHugging Face Hubから取得
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v2")

# Chromaデータベースを生成するディレクトリを指定
persist_directory = "../../app/vectordb"

print("ドキュメントをChromaデータベースに格納")

# ドキュメントをChromaデータベースに格納
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    collection_name="faqs",
    persist_directory=persist_directory,
    collection_metadata={"hnsw:space": "cosine"}
)

print("ベクトル情報のデータベースを作成しました")

query = "一人でゆっくり飲みたいなあ"

# 類似度検索
docs = vectordb.similarity_search_with_relevance_scores(query, k=1)

# ページコンテンツを取得
page_content = docs[0][0].page_content

# 類似度スコアとページ名を取得
similarity = docs[0][1]

# ページ名を取得
shop_information = page_content.replace("\n", ", ")

pprint(shop_information)
pprint(similarity)
