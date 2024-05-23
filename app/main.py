from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import traceback

EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v2")

persist_directory = "./app/vectordb"

csv_filepath = "./app/data/example.csv"

fieldlist = ["id", "name", "long", "lat", "area", "category"]



class TestSearchPubs:
    def csv_loader(csv_filepath: str, fieldlist: list):
        loader = CSVLoader(
            file_path=csv_filepath,
            csv_args={
                "delimiter": ",",
                "quotechar": '"',
                "fieldnames": fieldlist,
            },
            encoding="utf-8",
        )

        return loader.load()

    def search_by_vectorDB(location: str, prompt: str, csv_filepath: str, fieldlist: list):
        if os.path.exists(persist_directory): 
            vectordb = Chroma(persist_directory=persist_directory, collection_name="pubs", embedding_function=EMBEDDING_MODEL)

        else:
            try:
                docs = TestSearchPubs.csv_loader(csv_filepath, fieldlist)

            except FileNotFoundError as e:
                print(f"ファイルが見つかりませんでした: {csv_filepath}")

            except RuntimeError as e:
                print("CSVファイルに空のフィールドが存在しています\n", traceback.format_exc())
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

        docs = vectordb.similarity_search_with_relevance_scores(prompt, k=1)

        page_content = docs[0][0].page_content

        similarity = docs[0][1]

        shop_information = page_content.replace("\n", ", ")

        return{"shop_information": shop_information, "similarity": similarity}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Request(BaseModel):
    location: str
    prompt:  str


@app.get("/")
async def hello():
    return {"message": "Hello World"}

@app.post("/api")
async def search_pub(request: Request):
    location = request.location 
    prompt = request.prompt
    result = TestSearchPubs.search_by_vectorDB(location, prompt, csv_filepath, fieldlist)

    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
