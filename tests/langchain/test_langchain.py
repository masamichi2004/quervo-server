from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from pprint import pprint
import os

PERSIST_PATH = "./app/vectordb"

# Chromaデータベースを生成するディレクトリを指定
persist_directory = "./app/vectordb"



if not os.path.exists(PERSIST_PATH):
    # CSVファイルからドキュメントを読み込む
    print("ベクトル情報のデータベースが存在しません\nベクトルデータベースを作成します")
    print("CSVファイルからドキュメントを読み込む")
    
    try:
        loader = CSVLoader(
            file_path="./app/data/example.csv",
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

    print("ドキュメントをChromaデータベースに格納")

    # ドキュメントをChromaデータベースに格納
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name="faqs",
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    # データベースを永続化
    vectordb.persist()

    print("ベクトル情報のデータベースを作成しました")

else:    
    print("ベクトル情報のデータベースを読み込む")
    
    # データベースを読み込む
    vectordb = Chroma(persist_directory=persist_directory, collection_name="faqs", embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v2"))

    print("ベクトル情報のデータベースを読み込みました")

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
