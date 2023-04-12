from llama_index.prompts.prompts import QuestionAnswerPrompt, SummaryPrompt, RefinePrompt

DEFAULT_QA_PROMPT = (
    "We have provided context information below: \n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given this information, Please answer my question in the same language that I used to ask you.\n"
    "Please answer the question: {query_str}\n"
)

DEFAULT_SUMMARY_PROMPT_TMPL = (
    "Write a summary of the following. Try to use only the "
    "information provided. "
    "Try to include as many key details as possible.\n"
    "\n"
    "\n"
    "{context_str}\n"
    "\n"
    "\n"
    'SUMMARY:"""\n'
)

DEFAULT_REFINE_PROMPT_TMPL = (
    "The original question is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question. "
    "If the context isn't useful, return the original answer."
)



def get_QuestionAnswer_prompt():
    return QuestionAnswerPrompt(DEFAULT_QA_PROMPT)

def get_Summary_prompt():
    return SummaryPrompt(DEFAULT_SUMMARY_PROMPT_TMPL)

def get_refine_prompt():
    return RefinePrompt(DEFAULT_REFINE_PROMPT_TMPL)
