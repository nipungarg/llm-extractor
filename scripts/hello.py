from llm_extractor.config import settings
from openai import OpenAI
from google import genai

# OpenAI: messages list, answer at resp.choices[0].message.content
oai = OpenAI(api_key=settings.openai_api_key)  # reads OPENAI_API_KEY from the environment
r1 = oai.chat.completions.create(
    model=settings.openai_model,
    messages=[{"role": "user", "content": "In one sentence, what is an AI Engineer?"}],
)
print("OpenAI:", r1.choices[0].message.content)

# Gemini: 'contents' string, answer at resp.text  (note how different the shape is)
gem = genai.Client(api_key=settings.gemini_api_key)  # reads GEMINI_API_KEY
r2 = gem.models.generate_content(
    model=settings.gemini_model,
    contents="In one sentence, what is an AI Engineer?",
)
print("Gemini:", r2.text)