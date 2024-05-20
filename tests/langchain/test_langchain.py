from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from pprint import pprint
import os
import traceback


EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v2")

persist_directory = "./app/vectordb"

csv_filepath = "./app/data/example.csv"

fieldlist = ["id", "name", "long", "lat", "area", "category"]


def csv_loader(filepath: str, fieldlist: list):

    loader = CSVLoader(
        file_path=filepath,
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
            "fieldnames": fieldlist,
        },
        encoding="utf-8",
    )

    return loader.load()


def main(query: str):
    if os.path.exists(persist_directory): 
        vectordb = Chroma(persist_directory=persist_directory, collection_name="pubs", embedding_function=EMBEDDING_MODEL)

    else:
        if os.path.exists(csv_filepath) == False:
            print(f"{csv_filepath} のパスでCSVファイルが見つかりませんでした")
            return

        try:
            docs = csv_loader(csv_filepath, fieldlist)

        except RuntimeError as e:
            print("ランタイムエラーが発生しました\n以下にエラー内容を出力します\n", traceback.format_exc())
            return

        except Exception as e:
            print("予期せぬエラーが発生しました\n以下にエラー内容を出力します\n", traceback.format_exc())
            return
        
        if len(docs) == 0:
            print("CSVファイルにデータがありません")
            return

        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=EMBEDDING_MODEL,
            collection_name="pubs",
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )
        vectordb.persist()

    docs = vectordb.similarity_search_with_relevance_scores(query, k=2)

    page_content = docs[0][0].page_content

    similarity = docs[0][1]

    shop_information = page_content.replace("\n", ", ")

    pprint(shop_information)
    pprint(similarity)


main(query="茨木の飲み屋")
