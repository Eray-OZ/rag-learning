from dotenv import load_dotenv
import os
from huggingface_hub import InferenceClient


load_dotenv()

token = os.getenv("HUGGING_FACE_TOKEN")


    
client = InferenceClient(
    provider="hf-inference",
    api_key= token,
)


def get_embedding(text):
    
    result = client.feature_extraction(
    text,
    model="sentence-transformers/all-MiniLM-L6-v2",
)
    
    
    return result

embedding = get_embedding("This is an example sentece")

print(embedding)  # 384 boyutlu vektör