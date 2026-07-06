from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from groq import Groq
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

RETRYABLE_ERRORS = (
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
    openai.UnprocessableEntityError,
)

NONRETRYABLE_ERRORS = (
    openai.AuthenticationError,
    openai.BadRequestError,
    openai.NotFoundError,
    openai.PermissionDeniedError,
)

def load_env():
    try:
        load_dotenv(find_dotenv(usecwd=True))
    except Exception as e:
        print(f"Error loading .env file: {e}")

load_env()

def test_groq(user_prompt, temperature=0.7, max_tokens=100):
    groq_client = Groq()
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

@retry(
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(RETRYABLE_ERRORS),
    reraise=True,
)
def test_openai_errors(user_prompt, model_p):
    print("Testing OpenAI model...")
    client = OpenAI()
    result = client.responses.create(
        model=model_p,
        temperature=0,
        input=user_prompt,
    )
    return result.output_text

try:
    resu = test_openai_errors("hello, tell me 5 types of retriable errors", "gpt-4.1-mini")
    print(f"Result from OpenAI: {resu}")
except NONRETRYABLE_ERRORS as e:
    print(f"Non-retriable error: {e} --- Please check your model.")
    print("Calling fallback model provider")
    resu = test_groq("hello, tell me 5 types of retriable errors")
    print(f"Result from Groq: {resu}")