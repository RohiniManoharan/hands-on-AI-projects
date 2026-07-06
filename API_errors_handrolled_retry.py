from dotenv import load_dotenv,find_dotenv
from openai import OpenAI
from groq import Groq
import openai
import random
import time

RETRYABLE_ERRORS=(
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
    openai.UnprocessableEntityError,
)

NONRETRYABLE_ERRORS=(
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

def test_groq(user_prompt,temperature=0.7,max_tokens=100):
    groq_client=Groq()
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",       # Llama 3.3 70B — Groq's flagship open-source model
        messages=[{"role": "system", "content":"You are a helpful assistant."},
                  {"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_tokens=max_tokens)
    return response.choices[0].message.content    

def test_openai_errors(user_prompt,model_p,base_delay=1,max_retries=5): 
    for attempt in range(max_retries +1 ):
        try:
            print("Testing OpenAI model...")
            Client=OpenAI()
            result=Client.responses.create(
                model=model_p,
                temperature=0,
                input=user_prompt
            )
            return result.output_text
        except NONRETRYABLE_ERRORS as e:
            print(f"Non Retriable Error : {e}---Please check your model.")
            print("calling fallback model provider")
            result=test_groq(user_prompt)
            return result
        except RETRYABLE_ERRORS as e:
            if attempt < max_retries:
                print(f"Retriable Error : {e} Retrying...attempt{attempt+1} of {max_retries}")
                delay=base_delay * ( 2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(delay)
            else:
              print(f"Max retries reached.{e}")



resu= test_openai_errors("hello, tell me 5 types of retriable errors", "gpt-4.01-mini")
print(f"Result from OpenAI: {resu}")
#test_openai_api_key()
#test_openai_Bad_model()

#def test_groq_api_key():