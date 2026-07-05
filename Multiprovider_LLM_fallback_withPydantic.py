from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from groq import Groq
from google import genai
from google.genai import types
import os
import time
from pydantic import BaseModel,Field
from typing import List,Literal

class Customer_outputschema(BaseModel):
    classification: Literal["Support","billing","refund"]=Field(description="categories the tickets")
    sentiment: Literal["Positive","Neutral","Negative"]
    emergency: str = Field(description= "Shows how fast should respond")



Models={
        "openai_reason": "gpt-5-mini",           
        "gemini":        "gemini-2.5-flash",      
        "groq":          "llama-3.3-70b-versatile",
    }

providers=["gemini","openai_reason","groq"]
load_dotenv(find_dotenv(usecwd=True))
def load_keys():

    try:
       load_dotenv(find_dotenv(usecwd=True))
    except ImportError:
        pass

import os

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set")

gemini_client = genai.Client(api_key=api_key)

        
openai_client=OpenAI()
groq_client= Groq(api_key=os.environ["GROQ_API_KEY"])
#gemini_client=genai.Client(api_key=os.getenv["GEMINI_API_KEY"])
        
           

def call_provider(model_p,system,user,temperature=0,max_tokens=500):
    #try:
            print(f"MODEL used:{model_p}")
            if model_p=="openai_reason":
             
                    t0=time.time()
                    openai_client=OpenAI()
                    response= openai_client.responses.parse(
                        model =Models[model_p],
                        instructions= system,
                        input=user,
                        reasoning={"effort": "low"},
                        text_format=Customer_outputschema,
                        )
                    latency= round(time.time()-t0,2)
                    return {"Model":model_p, "Output":response.output_parsed, "latency":latency}
               

        
            if model_p=="groq":
                try:
                   
                    t0=time.time()
            
                    response= groq_client.chat.completions.create(
                    messages=[
                    
                   {
                    "role" : "user",
                    "content": user,
                   }
                    ],
                    model =Models[model_p],
                    )
                    latency= round(time.time()-t0,2)
                    return{"Model":model_p, "Output":response.choices[0].message.content, "latency":latency}
                except Exception as e:
                    raise RuntimeError(f"Error from model {model_p} :{e}")


            if model_p=="gemini":
               
                    t0=time.time()
           
                    response= gemini_client.models.generate_content(
                    
                    model =Models[model_p],
                    contents= user,
                    config= types.GenerateContentConfig(
                        system_instruction= system,
                        response_mime_type="application/json",
                        response_schema=Customer_outputschema),

                    )
                    
                    latency= round(time.time()-t0,2)
                    return{"Model":model_p, "Output":Customer_outputschema.model_validate_json(response.text), "latency":latency}
                

print("Loading the keys..........")       
load_keys()

def withfallback():
    for m in providers:
        try:

            print(f"providers {providers} Model called: {Models[m]}")
            result1=call_provider(m,system="You are AI gym user support assistant who analyses user feedback and classifies based on its emergency",user="i didnt use my account for this month, can i use this for next month without paying. I am contacting from last week,didnt get any one reply. Answer it immediately")    
            return result1
        except Exception as e:
            print(f"Failed model {Models[m]}")    

result_from_model=withfallback()
print(f"result.....{result_from_model}")