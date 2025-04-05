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
                {"role": "system", "content": """You are a coding assistant that helps generate code based on user queries.
For each query, you should:
1. Think through the solution step by step
2. Generate appropriate code
3. Format your response with the following structure:
   - Thinking steps wrapped in <think></think> tags
   - Code solution wrapped in <answer></answer> tags

Example format:
<think>
1. First, we need to...
2. Then, we should...
3. Finally, we...
</think>
<answer>
[Your code solution here]
</answer>"""},
                {"role": "user", "content": query}
            ],
            stream=False
        )
        return response.choices[0].message.content