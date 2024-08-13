from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain.chains.combine_documents import create_stuff_documents_chain
from milleLangChain.utils.helpers import load_prompt
from milleLangChain.utils import print_helpers as pp

import time


class Chat():
   """
    Chat class for creating and managing a conversational bot.

    This class initializes a chatbot by loading a context template and setting up a 
    language model for the bot's responses. It supports debug mode for additional logging.
   ----------------------------------------------------------------------------------------
    Attributes:
    template_contextBot (str): The loaded context template for the bot, sourced from a specified file. The template 
        must include the variable `{messages}`. If the bot is to perform a Retrieval-Augmented 
        Generation (RAG) process, the template must also contain a `<context>{context}</context>` 
        section.
    debug (bool) : A flag to indicate whether debug information should be printed. Defaults to False.
    """
   def __init__(self, file_context: str, llm, debug:bool = False) -> None:
      self.template_contextBot = load_prompt(file = file_context)
      self.debug = debug

      self.chain = self.create_bot(llm)

      if self.debug:
         pp.printy("---- Chat Validate -----")
         print(f"Context Bot: {file_context}")
         print("------------------------")


   def create_bot(self, llm):
      try: 
         question_answering_prompt = ChatPromptTemplate.from_messages([("system", self.template_contextBot),
                                                                       MessagesPlaceholder(variable_name="messages")])
         
         chain = create_stuff_documents_chain(llm, question_answering_prompt)

         pp.printg("** Create Chat Retrieval **")
         return chain

         
      except Exception as e:
         print(e)
         raise Exception("Fail in create Chat Validate model")



   def invoke_rag(self, docs, messages) -> tuple:
      """
      Invokes a Retrieval-Augmented Generation (RAG) process using the provided documents and messages.

      This method integrates documents with the language model's response, effectively combining 
      retrieval with generation. It also tracks the time taken for the model to respond and, if 
      debug mode is enabled, prints relevant debugging information.
      --------------------------------------------------------------------------------------------
      Args:
      docs (list): A list of documents to be used as context for the RAG process. Each document should 
                  contain metadata and content.
      messages (str): The messages or prompts to be processed by the language model.

      Returns (tuple):
         - result (str): The generated response from the language model.
         - time_llm (float): The time taken by the language model to generate the response.
      """
      try:
         if self.debug:
            print("\n****** Start CHAT LLM model ******")
         
         t1_llm = time.time()
         result = self.document_chain.invoke({
                      "context": docs,
                      "messages": messages,
                        })
         t2_llm = time.time()

         time_llm = round(t2_llm - t1_llm,3)
         if self.debug:
            print("****** Finish CHAT LLM model ******")
            print(f"Time: \n-LLM: {time_llm}")

         
         return result, time_llm

      except Exception as e:
         print(e)
         raise Exception("Error in apply Chat Validate model.")
      

   def invoke_simple(self, messages):
      """
      Invokes the language model with a simple prompt.

      This method sends a prompt to the language model and returns the generated response along 
      with the time taken for the model to respond. If debug mode is enabled, it prints relevant 
      debugging information.
      --------------------------------------------------------------------------
      Args:
      messages (str): The input messages or prompts to be processed by the language model.

      Returns (tuple):
         - result (str): The generated response from the language model.
         - time_llm (float): The time taken by the language model to generate the response.
      """
      try:
         if self.debug:
            print("\n****** Start CHAT LLM model ******")

         t1_llm = time.time()
         result = self.document_chain.invoke({
                      "messages": messages,
                        })
         t2_llm = time.time()

         time_llm = round(t2_llm - t1_llm,3)
         if self.debug:
            print("****** Finish CHAT LLM model ******")
            print(f"Times: \n-LLM: {time_llm}")
         return result, time_llm

      except Exception as e:
         print(e)
         raise Exception("Error in apply Chat Validate model.")