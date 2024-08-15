from langchain_community.embeddings.spacy_embeddings import Spacyembeddings
from langchain_openai import OpenAIembeddings
from langchain_community.embeddings import GPT4Allembeddings

from milleLangChain.utils import print_helpers as pp

class embedding():
    def __init__(self, model: str, embedding_type: str = "spacy", key: str = None , debug: bool = False, ) -> None:
        """
        Initializes the embedding class.

        Parameters:
            embedding_type (str): Type of embedding to use, either "spacy" or "openai".
            key (str): OpenAI API key required if embedding_type is "openai".
            debug (bool): Flag to enable debug mode. Default is False.

        Raises:
            AssertionError: If embedding_type is not one of the allowed types.
        """

        if debug: 
            pp.printy("--- embedding ---")
            print("set: ", embedding_type)

        allowed_types = ["spacy", "openia", "gpt4"]

        if "spacy" in embedding_type:
            self.model = Spacyembeddings(model_name=model)
        elif "openia" in embedding_type:
            assert key is not None and key != "", "OpenAI API key is required for 'openia' embedding type."
            self.model = OpenAIembeddings(openai_api_key=key, model=model)
        elif "gpt4" in embedding_type:
            self.model = GPT4Allembeddings(model_name=model)
        
        else:
            raise Exception("embedding type must be one of: " + " or ".join(allowed_types))

        self.name_model = embedding_type

        if debug:
            pp.printy("----------------")

        