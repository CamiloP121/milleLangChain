import tiktoken
from milleLangChain.utils import print_helpers as pp

class Token():
    def __init__(self, model: str = "gpt-3.5-turbo", debug: bool = False) -> None:
      self.name_model = model
      self.encoding = tiktoken.encoding_for_model(model)

      if debug: 
         pp.printy("--- Token ---")
         print("set: ", model)

    def calculate_tokens(self, string: str) -> int:
      """Returns the number of tokens in a text string."""
      num_tokens = len(self.encoding.encode(string))
      return num_tokens

      