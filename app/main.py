from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
import traceback
import csv
from app.models.api_models import Prompt
from app.models.izakaya import Izakaya
from app.models.coordinate import Coordinate
from geographiclib.geodesic import Geodesic
import os
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict

OPENAI_API_KEY =  os.getenv('OPENAI_API_KEY')

EMBEDDING_MODEL = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

csv_filepath = "./app/data/test.csv"

def calculate_destination_distance(currnet_coodinate: Coordinate, izakaya_coordinate: Coordinate) -> float:
    geod = Geodesic.WGS84
    g = geod.Inverse(*currnet_coodinate.coordinate, *izakaya_coordinate.coordinate)
    destination_distance_meters = g['s12']
    return destination_distance_meters

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def hello():
    return {"message": "Hello World"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api")
async def search_izakaya(izakaya_search_request: Prompt) -> List[Izakaya] | Dict[str, str]:
    izakaya_info_list = []         # type: List[Izakaya]
    fieldlist = []              # type: list[str]

    current_location = izakaya_search_request.current_coodinate     # type: Coordinate

    prompt = izakaya_search_request.prompt      # type: str

    try:
        f = open(csv_filepath, "r", encoding="utf-8")
        reader = csv.reader(f, delimiter=",", skipinitialspace=True)
        # 1行目はヘッダーなのでfieldlistに格納
        fieldlist = next(reader)         # type: list[str]
        f.close()
    except FileNotFoundError as e:
        print(f"ファイルが見つかりませんでした: {csv_filepath}")
        return {"error": "File not found"}

    except Exception as e:
        print("予期せぬエラーが発生しました\n以下にエラー内容を出力します\n", traceback.format_exc())
        return {"error": "Unexpected error"}

    try:
        loader = CSVLoader(
            file_path=csv_filepath,
            csv_args={
                "delimiter": ",",
                "quotechar": '"',
                "fieldnames": fieldlist,
            },
            encoding="utf-8",
        )

        izakaya_info_list = loader.load()

    except FileNotFoundError as e:
        print(f"ファイルが見つかりませんでした: {csv_filepath}")
        return {"error": "File not found"}

    except RuntimeError as e:
        print("CSVファイルに空のフィールドが存在しています\n", traceback.format_exc())
        return {"error": "Empty field found"}

    except Exception as e:
        print("予期せぬエラーが発生しました\n以下にエラー内容を出力します\n", traceback.format_exc())
        return {"error": "Unexpected error"}
    
    if len(izakaya_info_list) == 0:
        print("CSVファイルにデータがありません")
        return {"error": "No data found"}

    if len(izakaya_info_list) == 0:
        return {"error": "No data found"}

    vectordb = Chroma.from_documents(
        documents=izakaya_info_list,
        embedding=EMBEDDING_MODEL,
        collection_name="izakaya",
        collection_metadata={"hnsw:space": "cosine"}
    )

    docs = vectordb.similarity_search_with_relevance_scores(prompt, k=10)

    vectordb.delete_collection()

    if len(docs) == 0:
        return {"error": "No result found"}

    re_ranked_izakaya_list = []
    current_destination_distance = None
    for row in range(len(docs)):
        content = docs[row][0].page_content.split("\n")          # content = [id, name, lat, lng, area, category]

        # ヘッダー行はスキップ(あってもいらない)
        if content == ['id: id', 'name: name', 'lat: lat', 'lng: lng', 'area: area', 'category: category', 'prompt: prompt', 'photo_url: photo_url']:
            continue

        izakaya_coordinate = Coordinate(
            coordinate=(float(content[2].replace("lat: ", "")), float(content[3].replace("lng: ", "")))
        )

        if current_location is not None:
            current_destination_distance = calculate_destination_distance(current_location, izakaya_coordinate)

        izakaya_info = Izakaya(
            id=int(content[0].replace("id: ", "")),
            name=content[1].replace("name: ", ""),
            lat=float(content[2].replace("lat: ", "")),
            lng=float(content[3].replace("lng: ", "")),
            area=content[4].replace("area: ", ""),
            distance=current_destination_distance,
            category=content[5].replace("category: ", ""),
        )
        re_ranked_izakaya_list.append(izakaya_info)

    return re_ranked_izakaya_list