import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_KEY"],
)

response = client.responses.create(
    model="gpt-4.1-mini",
    input="Responde solo con: OK",
)
print(response.output_text)