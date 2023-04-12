import os

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from flask import Flask, request, make_response
import uvicorn
import openai
from openai_model import Openai_Chat, first_single_response

import pandas as pd 

from flask import Flask, g, request, make_response

from chat import create_llama_index, get_answer_from_llama_index, check_llama_index_exists
from file import get_index_name_without_json_extension
from file import get_index_path, get_index_name_from_file_name, check_index_file_exists, \
    get_index_name_with_json_extension

def init_chat_model():
    global my_chat_model
    system_setting = "I want you to act as a Senior Solidity Developer. I will provide some code about solidity smart contract, and it will be your job to audit provided solidity smart contract code, and refine provided smart contract code, also explain the code after the change"
    my_chat_model = Openai_Chat(model = "gpt-3.5-turbo",
                        system_setting = system_setting,
                        temperature = 0.7,
                        max_length = 4100, 
                        top_p = 1,
                        frequency_penalty = 0.1,
                        presence_penalty = 0.1,
                        init_prompt= True    # system setting
                    )


def get_global_variable():
    if my_chat_model is None:
        init_chat_model()
    return my_chat_model

app = FastAPI()

# middleware
app.add_middleware(
    CORSMiddleware, 
    allow_credentials=True, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)


@app.get('/query_llama')
async def query_from_llama_index(user_query: str):
    try:

        index_name = "./documents/data.json"

        if check_index_file_exists(index_name) is False:
            return "Index file does not exist", 404

        answer = get_answer_from_llama_index(user_query, index_name)
        return {"respond" : answer}
    
    except Exception as e:
        return "Error: {}".format(str(e)), 500

@app.get("/query_keyword")
async def query_from_keyword(user_query: str, my_chat_model = Depends(get_global_variable)):
    try:
        df = pd.read_csv("./documents/data.csv") # read external data

    except Exception as e:
        return "File doesn't exists !"

    contexts = list(df["content"])
    answer = my_chat_model(user_query, contexts)
    return {"respond" : answer}
    
if __name__ == "__main__":
    if not os.path.exists('./documents'):
        os.makedirs('./documents')
    uvicorn.run(app, host="0.0.0.0", port=8000)