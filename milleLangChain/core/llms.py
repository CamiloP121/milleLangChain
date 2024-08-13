from langchain_community.chat_models import ChatOllama
# from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from milleLangChain.utils.CustomOllama import Ollama

from milleLangChain.utils import print_helpers as pp


class LLM():
   def __init__(self, llm_type:str = "open_ia", key: str = None, model: str = None ,temperature:int = 0.7, debug:bool = False, url: str = None) -> None:
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

      allowed_types = ["ollama", "open_ia", "groq"]
      assert llm_type in allowed_types, "LLMs type must be one of: " + " or ".join(allowed_types)

      if "ollama" in llm_type:
         if not model: model = "llama3:8b"
         self.model = Ollama(model=model, base_url=url)
      elif "open_ia" in llm_type:
         assert key is not None and key != "", "OpenAI API key is required for 'open_ia' llm type."
         if not model: model = "gpt-3.5-turbo-16k"
         self.model = ChatOpenAI(openai_api_key=key, model_name=model, temperature=temperature)
      elif "groq" in llm_type:
         if not model: model = "Llama3-8B-8k"
         self.model = ChatGroq(api_key=key, model=model, temperature=temperature)
      elif "anthropic" in llm_type:
         assert key is not None and key != "", "Anthropic API key is required for 'anthropic' llm type."
         raise Exception("Not implemented yet Anthropic LLM")

      self.name_model = llm_type

      if debug:
         pp.printy("----------------")