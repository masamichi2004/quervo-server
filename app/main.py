from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import traceback
import requests
import csv
import math
from models.api_models import Request
from models.izakaya import Izakaya
from models.coodinate import Coordinate
from geographiclib.geodesic import Geodesic

EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v2")

GOOGLE_MAP_GEOCODING_API_KEY = os.getenv("GOOGLE_MAP_GEOCODING_API_KEY")

persist_directory = "./app/vectordb"

csv_filepath = "./app/data/example.csv"

distance_limit = 1000

(LAT_COLUMUN, LONG_COLUMUN) = (2, 3)

def calculate_destination_distance(search_point_coodinate: Coordinate, izakaya_coordinate: Coordinate) -> float:
    geod = Geodesic.WGS84
    g = geod.Inverse(search_point_coodinate.lat, search_point_coodinate.lng, izakaya_coordinate.lat, izakaya_coordinate.lng)
    destination_distance_meters = g['s12']
    return destination_distance_meters

def get_search_point_coordinates(place_name: str)-> Coordinate|None:
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={place_name}&key={GOOGLE_MAP_GEOCODING_API_KEY}"
    response = requests.get(url)
    data = response.json()
    if data['status'] != 'OK':
        return None
    else:
        search_point_coordinate = Coordinate(
            lat=data['results'][0]['geometry']['location']['lat'],
            lng=data['results'][0]['geometry']['location']['lng'],
        )

        return search_point_coordinate

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

@app.post("/api")
async def search_izakaya(request: Request) -> list[Izakaya]:
    csvlist_by_chroma = [] # embedding_izakaya_info_list
    distant_elements_number_list = []
    fieldlist = []

    location = request.location
    prompt = request.prompt

    search_point_coordinate = get_search_point_coordinates(location)

    if search_point_coordinate is None:
        return {"error": "Invalid location"}
    try:
        with open(csv_filepath, encoding="utf-8", newline="") as f:
            reader=csv.reader(f)
            for list_row_number, csvfile_element_row in enumerate(reader):
                # 1行目はヘッダーなのでfieldlistに格納
                if  list_row_number == 0:
                    fieldlist = csvfile_element_row         # csvfile_element_row: list[str]
                    continue

                csvfile_element_row[LAT_COLUMUN], csvfile_element_row[LONG_COLUMUN] = float(csvfile_element_row[LAT_COLUMUN]), float(csvfile_element_row[LONG_COLUMUN])

                izakaya_coordinate = Coordinate(
                    lat=csvfile_element_row[LAT_COLUMUN], 
                    lng=csvfile_element_row[LONG_COLUMUN]
                )

                # distant_elements_number_listにpopしたい要素をメモ
                destination_distance_meters = calculate_destination_distance(search_point_coordinate, izakaya_coordinate)

                if destination_distance_meters > distance_limit:
                    distant_elements_number_list.append(list_row_number)

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

        csvlist_by_chroma = loader.load()

    except FileNotFoundError as e:
        print(f"ファイルが見つかりませんでした: {csv_filepath}")
        return {"error": "File not found"}

    except RuntimeError as e:
        print("CSVファイルに空のフィールドが存在しています\n", traceback.format_exc())
        return {"error": "Empty field found"}

    except Exception as e:
        print("予期せぬエラーが発生しました\n以下にエラー内容を出力します\n", traceback.format_exc())
        return {"error": "Unexpected error"}
    
    if len(csvlist_by_chroma) == 0:
        print("CSVファイルにデータがありません")
        return {"error": "No data found"}
    

    for index in sorted(distant_elements_number_list, reverse=True):
        csvlist_by_chroma.pop(index)

    csvlist_by_chroma.pop(0)        # 1行目はヘッダーなのでスキップ

    if len(csvlist_by_chroma) == 0:
        return {"error": "No data found"}

    vectordb = Chroma.from_documents(
        documents=csvlist_by_chroma,
        embedding=EMBEDDING_MODEL,
        collection_name="izakaya",
        collection_metadata={"hnsw:space": "cosine"}
    )

    docs = vectordb.similarity_search_with_relevance_scores(prompt, k=1)

    vectordb.delete_collection()

    if len(docs) == 0:
        return {"error": "No result found"}

    responselist = []
    for row in range(len(docs)):
        content = docs[row][0].page_content.split("\n")          # shop_information = [id, name, long, lat, area, category]
        izakaya_info = Izakaya(
            id=int(content[0].replace("id: ", "")),
            name=content[1].replace("name: ", ""),
            long=float(content[2].replace("long: ", "")),
            lat=float(content[3].replace("lat: ", "")),
            area=content[4].replace("area: ", ""),
            category=content[5].replace("category: ", ""),
        )
        responselist.append(izakaya_info)

    return responselist

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)