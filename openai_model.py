import openai
import numpy as np
import os
from openai.error import RateLimitError, ServiceUnavailableError
import logging
import sys
from pathlib import Path
import time
import copy

from collections import Counter
from utils import calculate_cosine, calculate_cosine_code

"""
# extract keyword
def keyword_matching(query_str):
    extract_prompt = [{"role":"system","content": f"You are good at Extract keywords, for example : 我想學習solidity的智能合約與區塊鏈, you will split to : solidity,智能合約,區塊鏈"}]
    extract_prompt.append({"role":"user","content": f"Extract keywords from this text: {query_str}"})
    while True:
        try : 
            response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                temperature = 0,
                                top_p = 1,
                                messages = extract_prompt
                                )
            break
        except (RateLimitError, ServiceUnavailableError):
                time.sleep(0.1)

    # remove unuseful stuff
    temping = response.choices[0].message['content']

        
    replace_words = {"1.": "", "2.": "", "3.": "", " " : "", ".":"" ,"。":"", "\n" : ",", "、":",", "Keywords:":","}

    for old_word, new_word in replace_words.items():
        temping = temping.replace(old_word, new_word)

    key_words = temping.split(',')
    print("key_word:",key_words)
    return key_words

# answer single respone 
def first_single_response(question, context, messages :list):
    messages.append({"role":"system","content": f"You are an expert in 繁體中文 and you are good at finding answers in a given context"})
    messages.append({"role":"user", "content": f"Context information is below \n {context} \n Given the context information and previous conversation, answer the question : {question} in #zh-hant \n Respond : Unsure about answer if not sure about the answer."})
    print("run query")
    while True:
                try : 
                    response = openai.ChatCompletion.create(
                            model = "gpt-3.5-turbo",
                            messages = messages,
                            temperature = 0.5,
                            max_tokens = 2000
                        )
                    break
                except (RateLimitError, ServiceUnavailableError):
                        time.sleep(0.1)
    answer = response.choices[0].message['content']
    return answer

# chatgpt answer single reply
def simple_reply(question, code_context, program_language):
    messages = []
    messages.append({"role":"system","content": f"I want you to act as a Senior {program_language} Developer. I will provide some code about solidity smart contract, and it will be your job to audit provided solidity smart contract code, refine provided smart contract code, and explain the code after the change"})
    messages.append({"role":"user", "content": f"Here are solidity code : {question}, if there is a problem with this solidity code or if there is a security concern, modify this solidity code and explain why, Here is full code {code_context} if needed"})
    print("run query")
    while True:
                try : 
                    response = openai.ChatCompletion.create(
                            model = "gpt-3.5-turbo",
                            messages = messages,
                            temperature = 0.5,
                            max_tokens = 4000
                        )
                    break
                except (RateLimitError, ServiceUnavailableError):
                        time.sleep(0.1)
    answer = response.choices[0].message['content']
    return answer
"""

def first_send(selected_code, full_code, program_language):
    messages = []
    messages.append({"role":"system","content": f"I want you to act as a Senior {program_language} Developer. I will provide some code about solidity smart contract, and it will be your job to audit provided solidity smart contract code, refine provided smart contract code, and explain the code after the change"})
    messages.append({"role":"user", "content": f"Here are solidity code : {selected_code}, if there is a problem with this solidity code or if there is a security concern, modify this solidity code and explain why, Here is full code {full_code} if needed"})
    return messages

def refine_reply(selected_code, full_code, contents : list, previous_answer, program_language):

    print("Calculate similarity !")
    
    selected_contents = calculate_cosine_code(full_code, contents)
    
    print("finish query")
    
    refine_template=  f"The original ask code is as follows: {selected_code}\n We have provided an existing answer: {previous_answer}\n We have the opportunity to refine the existing answer (only if needed) with some more context below.\n \
    {selected_contents} \n Given the new context, refine the original answer to better \n answer the question. If the context isn't useful, return the original answer."
    
    messages = []
    messages.append({"role" : "system","content": f"I want you to act as a Senior {program_language} Developer. I will provide some code about solidity smart contract, and it will be your job to audit provided solidity smart contract code, refine provided smart contract code, and explain the code after the change"})
    messages.append({"role" : "user", "content": f"Here are solidity code : {selected_code}, if there is a problem with this solidity code or if there is a security concern, modify this solidity code and explain why, Here is full code {full_code} if needed"})
    messages.append({"role" : "assistant", "content" : previous_answer})
    messages.append({"role" : "user", "content": refine_template})

    return messages


"""
class Openai_Chat:
    def __init__(self, model, system_setting, temperature, max_length, top_p, frequency_penalty, presence_penalty, init_prompt, n = 1):

        #openai.api_key = os.getenv("OPENAI_API_KEY")

        # parameter settting
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_length
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.n = n
        
        self.conversation_list = []
        if init_prompt:
            self.conversation_list.append(
                {"role":"system", "content": system_setting}
            )

    # show current conversation
    def show_conversation(self):
        conversation_list = self.conversation_list
        print(conversation_list)

        show_talk = []
        for msg in conversation_list:
            content = msg['content']
            
            if msg['role'] == 'user':
                show_talk.append({"role" : 'user', "content" : f"\U0001F47B: {content}"})
                print(f"\U0001F47B: {content}\n")

            elif msg['role'] == 'assistant':
                show_talk.append({"role" : 'assistant', "content" : f"\U0001F4BB: {content}"})
                print(f"\U0001F4BB: {content}\n")

        return show_talk

    # openai keyword matching
    def external_keyword(self, user_text, contexts):
"""
       # user_text: str, the user's text to query for
       # contexts : list, external data for search
"""
        self.conversation_list.append({"role":"user", "content" : user_text})

        # get key words
        key_words = keyword_matching(user_text)

        # define whether key word in contexts
        context_index = []
        for word in key_words:
            for idx, context in enumerate(contexts):
                if word in context:
                    context_index.append(idx)

        # if no key word involve, then select every context to calculate cosine similarity        
        if len(context_index) == 0:
            print("no keyword !")

            # Calculate cosine similarity between question and every contexts
            print("Calculate similarity !")
            context_index = list(range(0, len(contexts)))

            similarity_scores = {} 
        
            for idx in context_index:
                score = calculate_cosine(user_text, contexts[idx])
                similarity_scores[idx] = score

            max_key = max(similarity_scores, key=lambda k: similarity_scores[k])
            print(contexts[max_key])

            input_context = contexts[max_key] # final input context

        # select involve key word paragraph
        else:   
            counted = Counter(context_index)
            most_common_element = counted.most_common(1)[0][0]

            print("most common: ",most_common_element)
            input_context = contexts[most_common_element] # final input context

        # memory past answer
        input_prompt = copy.deepcopy(self.conversation_list)
        answer = first_single_response(user_text, input_context, input_prompt)


        self.conversation_list.append({"role" : 'assistant', "content" : f"{answer}"})
        
        return answer
        
    def get_multiple_response(self,prompts):
        pass
"""   