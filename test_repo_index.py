# This is due to the fact that we use asyncio.loop_until_complete in
# the DiscordReader. Since the Jupyter kernel itself runs on
# an event loop, we need to add some help with nesting

import nest_asyncio
from llama_index import GPTSimpleVectorIndex, GithubRepositoryReader
import os 
import openai
from dotenv import load_dotenv
load_dotenv()

#openai.api_key = os.getenv("OPENAI_API_KEY")

import os
nest_asyncio.apply()



github_token = "aaaa"
owner = "SunWeb3Sec"
repo = "DeFiHackLabs"
branch = "main"

documents = GithubRepositoryReader(
    github_token=github_token,
    owner=owner,
    repo=repo,
    use_parser=False,
    verbose=False,
).load_data(branch=branch)
print("read")

index = GPTSimpleVectorIndex.from_documents(documents)
index.save_to_disk("github_index.json")
print("save")

response = index.query("Can you tell me 20230411 Paribus - Reentrancy code", verbose=True)

print(str(response))