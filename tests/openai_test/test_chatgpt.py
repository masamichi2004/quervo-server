from fastapi import FastAPI
import uvicorn
from openai import OpenAI 
import os
from pydantic import BaseModel
from dotenv import load_dotenv

app = FastAPI()

# 環境変数の読み込み
load_dotenv()

# リクエストボディの定義
class Message(BaseModel):
    prompt: str

# openAIのクライアントを作成
client = OpenAI(
    api_key = os.getenv('API_KEY')
)

@app.post("/gpt")
async def gpt(message: Message):

    # プロンプトを受け取り、GPT-3で応答を生成
    prompt = message.prompt
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "user", "content": prompt},
        ],
    )

    # 応答を取得
    text = response['choices'][0]['message']['content']

    return {"response": text}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
