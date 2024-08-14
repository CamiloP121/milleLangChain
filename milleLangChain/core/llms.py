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

      allowed_types = ["ollama", "openia", "groq"]

      if "ollama" in llm_type:
         if not model: model = "llama3:8b"
         self.model = Ollama(model=model, base_url=url)
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

      self.name_model = llm_type

      if debug:
         pp.printy("----------------")

   def invoke(self, text, return_text_only=True):
      """
      Invokes the language model with the provided text and returns the response.
      This method sends a text prompt to the language model and returns either the full response 
      or just the content, depending on the `return_text_only` flag.
      --------------------------------------------------------------------------
      Args:
      ----
      text (str): The input text or prompt to be processed by the language model.
      return_text_only (bool, optional): If True, only the text content of the response is returned. 
                                          If False, the entire response object is returned. Defaults to True.

      Returns:
      -------
      str or object: The text content of the response if `return_text_only` is True, otherwise the full 
                     response object.
      """
      try:
         response = self.model.invoke( text )
      except Exception as e:
            print(e)
            raise Exception("Error in invoke model")
      

      if return_text_only:
         if "ollama" in self.name_model: r = response
         else: r = response.content
      else: 
         if "ollama" in self.name_model: pp.printy("Warning: Ollama only have text mode")
         r = response
      
      return r
      