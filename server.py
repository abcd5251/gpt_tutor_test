import os

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from openai_model import first_send, refine_reply
import uvicorn

import pandas as pd 


app = FastAPI()

# middleware
app.add_middleware(
    CORSMiddleware, 
    allow_credentials=True, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

class single_respond(BaseModel):
    question : str
    code_context : str
    program_language : str

class second_respond(BaseModel):
    question : str
    code_context : str
    previous_answer : str
    program_language : str


"""
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
@app.get('/query_llama')
async def query_from_llama_index(user_query: str):
    try:
        index_name = "./documents/data.json"
        if os.path.isfile(index_name) is False:
            return "Index file does not exist", 404
        
        index = GPTSimpleVectorIndex.load_from_disk(index_name)
        answer = index.query(user_query)
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
"""

"""
@app.get("/extract_keyword")
async def single_respond(question):
    respond = first_send(question, code_context, program_language)
    return respond
"""

@app.post("/single_respond/")
async def single_respond(item: single_respond):
    respond = first_send(item.question, item.code_context, item.program_language)
    print(respond)
    return {"code":0, "data":respond}


@app.post("/second_respond/")
# refine_reply(selected_code, full_code, contents : list, previous_answer, program_language)
async def second_respond(item: second_respond):
    index_name = "data.csv"
    df = pd.read(index_name)
    respond = refine_reply(item.question, item.code_context, list(df["content"]), item.previous_answer, item.program_language)
    return {"code" : 0, "data" : respond}


"""
@app.get("/refine_answer")
async def refine_answer(user_query, full_code, my_chat_model = Depends(get_global_variable)):
    temp_answer = simple_reply(user_query, full_code)
    return {"respond" : temp_answer}
    
    try:
        index_name = "./documents/data.json"
        if os.path.isfile(index_name) is False:
            return "Index file does not exist", 404
        
        index = GPTSimpleVectorIndex.load_from_disk(index_name)
        answer = index.query(user_query)
        return {"respond" : answer}
    
    except Exception as e:
        return "Error: {}".format(str(e)), 500
"""
    

if __name__ == "__main__":
    if not os.path.exists('./documents'):
        os.makedirs('./documents')
    uvicorn.run(app, host="0.0.0.0", port=8000)