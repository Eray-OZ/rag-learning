from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os


load_dotenv()

client = InferenceClient(
    provider="featherless-ai",
    api_key=os.environ["HUGGING_FACE_TOKEN"],
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct-1M",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ],
)

print(response)
