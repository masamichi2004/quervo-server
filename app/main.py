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
from models.api_models import Prompt
from models.izakaya import Izakaya
from app.models.coordinate import Coordinate
from geographiclib.geodesic import Geodesic
from typing import List

EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v2")

GOOGLE_MAP_GEOCODING_API_KEY = os.getenv("GOOGLE_MAP_GEOCODING_API_KEY")

csv_filepath = "./app/data/example.csv"

distance_limit = 1000

(LAT_COLUMUN, LNG_COLUMUN) = (2, 3)

def calculate_destination_distance(search_point_coodinate: Coordinate, izakaya_coordinate: Coordinate) -> float:
    geod = Geodesic.WGS84
    g = geod.Inverse(*search_point_coodinate.coordinate, *izakaya_coordinate.coordinate)
    destination_distance_meters = g['s12']
    return destination_distance_meters

def get_search_point_coordinates(place_name: str) -> Coordinate|None:
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={place_name}&key={GOOGLE_MAP_GEOCODING_API_KEY}"
    response = requests.get(url)
    data = response.json()
    if data['status'] != 'OK':
        return None
    else:
        search_point_coordinate = Coordinate(
            coordinate=(data['results'][0]['geometry']['location']['lat'], data['results'][0]['geometry']['location']['lng'])
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
async def search_izakaya(izakaya_search_request: Prompt):
    izakaya_info_list = []         # type: List[Izakaya]
    distant_elements_num_list = []     # type: List[int]
    fieldlist = []

    location = izakaya_search_request.location
    prompt = izakaya_search_request.prompt

    search_point_coordinate = get_search_point_coordinates(location)

    if search_point_coordinate is None:
        return {"error": "Invalid location"}
    try:
        f = open(csv_filepath, "r", encoding="utf-8")
        reader = csv.reader(f, delimiter=",", lineterminator="\r\n", skipinitialspace=True)
        for list_row_number, csvfile_element_row in enumerate(reader):
            # 1行目はヘッダーなのでfieldlistに格納
            if  list_row_number == 0:
                fieldlist = csvfile_element_row         # csvfile_element_row: list[str]
                continue

            csvfile_element_row[LAT_COLUMUN], csvfile_element_row[LNG_COLUMUN] = float(csvfile_element_row[LAT_COLUMUN]), float(csvfile_element_row[LNG_COLUMUN])

            izakaya_coordinate = Coordinate(
                coordinate=(csvfile_element_row[LAT_COLUMUN], csvfile_element_row[LNG_COLUMUN])
            )

            # distant_elements_num_listにpopしたい要素をメモ
            destination_distance_meters = calculate_destination_distance(search_point_coordinate, izakaya_coordinate)

            if destination_distance_meters > distance_limit:
                distant_elements_num_list.append(list_row_number)
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
    

    for distant_elements_num in sorted(distant_elements_num_list, reverse=True):
        izakaya_info_list.pop(distant_elements_num)

    izakaya_info_list.pop(0)        # 1行目はヘッダーなのでスキップ

    if len(izakaya_info_list) == 0:
        return {"error": "No data found"}

    vectordb = Chroma.from_documents(
        documents=izakaya_info_list,
        embedding=EMBEDDING_MODEL,
        collection_name="izakaya",
        collection_metadata={"hnsw:space": "cosine"}
    )

    docs = vectordb.similarity_search_with_relevance_scores(prompt, k=1)

    vectordb.delete_collection()

    if len(docs) == 0:
        return {"error": "No result found"}

    re_ranked_izakaya_list = []
    for row in range(len(docs)):
        content = docs[row][0].page_content.split("\n")          # izakaya_information = [id, name, lng, lat, area, category]
        izakaya_info = Izakaya(
            id=int(content[0].replace("id: ", "")),
            name=content[1].replace("name: ", ""),
            lng=float(content[2].replace("lng: ", "")),
            lat=float(content[3].replace("lat: ", "")),
            area=content[4].replace("area: ", ""),
            category=content[5].replace("category: ", ""),
        )
        re_ranked_izakaya_list.append(izakaya_info)

    return re_ranked_izakaya_list

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)