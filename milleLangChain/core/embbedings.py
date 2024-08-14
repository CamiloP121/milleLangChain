from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings

from milleLangChain.utils import print_helpers as pp

class Embedding():
    def __init__(self, model: str, embedding_type: str = "spacy", key: str = None , debug: bool = False, ) -> None:
        """
        Initializes the Embedding class.

        Parameters:
            embedding_type (str): Type of embedding to use, either "spacy" or "openai".
            key (str): OpenAI API key required if embedding_type is "openai".
            debug (bool): Flag to enable debug mode. Default is False.

        Raises:
            AssertionError: If embedding_type is not one of the allowed types.
        """

        if debug: 
            pp.printy("--- Embedding ---")
            print("set: ", embedding_type)

        allowed_types = ["spacy", "openia", "gpt4"]
        for allowed in allowed_types:
            assert allowed in embedding_type, "Embedding type must be one of: " + " or ".join(allowed_types)

        if "spacy" in embedding_type:
            self.model = SpacyEmbeddings(model_name=model)
        elif "openia" in embedding_type:
            assert key is not None and key != "", "OpenAI API key is required for 'openia' embedding type."
            self.model = OpenAIEmbeddings(openai_api_key=key, model=model)
        elif "gpt4" in embedding_type:
            self.model = GPT4AllEmbeddings(model_name=model)

        self.name_model = embedding_type

        if debug:
            pp.printy("----------------")

        