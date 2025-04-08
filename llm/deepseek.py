import os
import time
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
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
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
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
            except RateLimitError:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    return "<think>Rate limit exceeded</think><answer>Error: API rate limit exceeded</answer>"
            except (APIError, APIConnectionError) as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return f"<think>API error: {str(e)}</think><answer>Error: Failed to get response from API</answer>"
            except Exception as e:
                return f"<think>Unexpected error: {str(e)}</think><answer>Error: Failed to process query</answer>"