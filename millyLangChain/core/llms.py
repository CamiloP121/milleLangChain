from langchain_community.chat_models import ChatOllama
# from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from millyLangChain.utils.CustomOllama import Ollama

from millyLangChain.utils import print_helpers as pp


class LLM():
   def __init__(self, llm_type:str = "open_ia", key: str = None, model: str = None ,temperature:int = 0.7, debug:bool = False, url: str = None) -> None:
      
      if debug: 
            pp.printy("--- LLMs ---")
            print("set: ", llm_type)

      name_model = llm_type
      llm_type = llm_type.split("-")[0]

      allowed_types = ["ollama", "open_ia", "groq"]
      assert llm_type in allowed_types, "LLMs type must be one of: " + " or ".join(allowed_types)

      if llm_type == "ollama":
         if not model: model = "llama3:8b"
         self.model = Ollama(model = model, base_url = url)
         #raise Exception("Not implemented yet Ollama LLM")
      elif llm_type == "open_ia":
         assert key is not None and key != "", "OpenAI API key is required for 'open_ia' llm type."
         if not model: model = "gpt-3.5-turbo-16k"
         self.model = ChatOpenAI(openai_api_key = key, model_name= model, temperature = temperature)
      elif llm_type == "groq":
         if not model: model = "Llama3-8B-8k"
         self.model = ChatGroq(api_key = key, model= model, temperature = temperature)

      elif llm_type == "anthropic":
         assert key is not None and key != "", "Anthropic API key is required for 'anthropic' llm type."
         #self.model = ChatAnthropic(openai_api_key = openai_api_key, model_name="gpt-3.5-turbo-16k", temperature=0)
         raise Exception("Not implemented yet Anthropic LLM")


      self.name_model = name_model

      if debug:
         pp.printy("----------------")