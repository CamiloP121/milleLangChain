from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama

from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain.chains.combine_documents import create_stuff_documents_chain
from milleLangChain.utils.helpers import load_prompt
from milleLangChain.utils import print_helpers as pp


class ChatLLM:
   def __init__(self, llm_type:str = "open_ia", key: str = None, model: str = None ,temperature:int = 0.7, debug:bool = False, url: str = None, 
                # Chat Variables
                prompt_system: str = None,
                file_system: str = None, 
                rag: bool = False) -> None:
      """
      LLM class for initializing and managing different types of language models.

      This class supports various language models such as Ollama, OpenAI, and Groq. It allows 
      the user to specify the type of language model, its API key, model name, temperature, 
      and other relevant parameters. It also supports a debug mode for additional logging.
      ----------------------------------------------------------------------------
      Attributes:
      
      model (object): The initialized language model based on the specified type and parameters.
      name_model (str): The name/type of the language model initialized.
      """
      if debug: 
            pp.printy("--- LLMs ---")
            print("set: ", llm_type)

      self.name_model = llm_type; self.flag_rag = rag

      allowed_types = ["ollama", "openia", "groq"]

      if "ollama" in llm_type:
         if not model: model = "llama3:8b"
         self.model = ChatOllama(model=model, base_url=url)
      elif "openia" in llm_type:
         assert key is not None and key != "", "OpenAI API key is required for 'open_ia' llm type."
         if not model: model = "gpt-3.5-turbo-16k"
         self.model = ChatOpenAI(openai_api_key=key, model_name=model, temperature=temperature)
      elif "groq" in llm_type:
         if not model: model = "Llama3-8B-8k"
         self.model = ChatGroq(api_key=key, model=model, temperature=temperature)
      elif "anthropic" in llm_type:
         assert key is not None and key != "", "Anthropic API key is required for 'anthropic' llm type."
         raise Exception("Not implemented yet Anthropic LLM")

      else: 
         raise Exception("LLMs type must be one of: " + " or ".join(allowed_types))

      

      assert (prompt_system and not file_system) or (not prompt_system and file_system), "Only one of the system prompts should be assigned"
      
      if file_system: self.template_contextBot = load_prompt(file = file_system)
      else:  self.template_contextBot = prompt_system

      assert "{messages}" in self.template_contextBot, "Missing '{messages}' at the system prompt"

      if self.flag_rag:
         assert "<context>{context}</context>" in self.template_contextBot, "Missing '{context}' at the system prompt"

      self.chain = self.bot_setup(rag = self.flag_rag,
                                  system = self.template_contextBot, 
                                  llm = self.model)

      if debug:
         pp.printy("----------------")

   def bot_setup(system:str, rag:bool, llm):
      
      prompt = ChatPromptTemplate.from_messages(
         [("system", system),
         MessagesPlaceholder(variable_name="messages")])
      
      if rag: chain = create_stuff_documents_chain(llm, prompt)
      else: chain = prompt | llm

      return chain
   
   def invoke(self, messages, docs = None):
      
      input = {"messages": messages}
      
      if self.flag_rag:
         assert docs is None, "Don't have docs"
         input["context"] = docs

      try:
         if self.debug:
            print("\n****** Start CHAT LLM model ******")
         
         result = self.chain.invoke( input )

         if self.debug:
            print("****** Finish CHAT LLM model ******")
         return result

      except Exception as e:
         print(e)
         raise Exception("Error in apply Chat Validate model.")
      