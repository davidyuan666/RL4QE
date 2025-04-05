import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class DeepseekAPI:
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        
    def get_response(self, query: str) -> str:
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a assiant to generete enhance query according to the origin query."},
                {"role": "user", "content": query}
            ],
            stream=False
        )
        return response.choices[0].message.content