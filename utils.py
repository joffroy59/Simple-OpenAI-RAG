from dotenv import load_dotenv
import os

def load_api_keys():
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")