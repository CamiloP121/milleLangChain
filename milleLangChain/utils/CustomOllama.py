from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
import requests
from ollama import Client


class Ollama(LLM):
    """
    Custom Ollama LLM (Large Language Model) class.
    This class serves as a custom implementation for interacting with the Ollama model.
    ---------------------------------------------------------------------------
    Args:
    model (str): The name or identifier of the Ollama model to be used.
    base_url (str): The base URL for the Ollama model API.
    debug (bool, optional): An optional parameter that, if True, enables the printing of debug information. Default is False.
    
    Methods:
    invoke(prompt: str, verbose: bool = False) -> str
        Sends a prompt to the model and returns either the full JSON response or just the generated text.
    
    """

    model: str
    base_url: str
    client: Any = None
    debug: bool = False
    """The number of characters from the last message of the prompt to be echoed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Instantiating the ChatBot class
        # check api
        assert self.verificar_api(), f"Don´t has connection with {self.base_url}"

        self.client = Client( host= self.base_url)
        

    def verificar_api(self):
        try:
            response = requests.get(self.base_url)
            if response.status_code == 200:
                Flag = True
            else:
                if self.debug: print(f"The API responded with the status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print("Error. Could not connect to the API:", e)
            Flag = False

        return Flag

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
        ):

        response = self.client.generate(model = self.model,
                                    prompt = prompt)
        
        if verbose: r = response["response"]
        else: r = str(response["response"])
        
        return r 

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the LLM on the given prompt.

        This method should be overridden by subclasses that support streaming.

        If not implemented, the default behavior of calls to stream will be to
        fallback to the non-streaming version of the model and return
        the output as a single chunk.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An iterator of GenerationChunks.
        """
        for char in prompt[: self.n]:
            chunk = GenerationChunk(text=char)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

            yield chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": "CustomChatModel",
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "custom"