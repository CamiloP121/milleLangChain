from langchain_community.embbedings.spacy_embbedings import Spacyembbedings
from langchain_openai import OpenAIembbedings
from langchain_community.embbedings import GPT4Allembbedings

from milleLangChain.utils import print_helpers as pp

class embbeding():
    def __init__(self, model: str, embbeding_type: str = "spacy", key: str = None , debug: bool = False, ) -> None:
        """
        Initializes the embbeding class.

        Parameters:
            embbeding_type (str): Type of embbeding to use, either "spacy" or "openai".
            key (str): OpenAI API key required if embbeding_type is "openai".
            debug (bool): Flag to enable debug mode. Default is False.

        Raises:
            AssertionError: If embbeding_type is not one of the allowed types.
        """

        if debug: 
            pp.printy("--- embbeding ---")
            print("set: ", embbeding_type)

        allowed_types = ["spacy", "openia", "gpt4"]

        if "spacy" in embbeding_type:
            self.model = Spacyembbedings(model_name=model)
        elif "openia" in embbeding_type:
            assert key is not None and key != "", "OpenAI API key is required for 'openia' embbeding type."
            self.model = OpenAIembbedings(openai_api_key=key, model=model)
        elif "gpt4" in embbeding_type:
            self.model = GPT4Allembbedings(model_name=model)
        
        else:
            raise Exception("embbeding type must be one of: " + " or ".join(allowed_types))

        self.name_model = embbeding_type

        if debug:
            pp.printy("----------------")

        