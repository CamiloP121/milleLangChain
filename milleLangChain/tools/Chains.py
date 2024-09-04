from milleLangChain.utils.helpers import load_prompt
from milleLangChain.utils import print_helpers as pp
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, create_model
import time


class LLM_Clasify:
    def __init__(self, llm, file_context:str, context: str, intentions:list, debug:bool = False) -> None:
        
        if file_context: self.template_contextBot = PromptTemplate(load_prompt(file = file_context)) + " {text}"
        if context: self.template_contextBot = PromptTemplate(context) + " {text}"
        self.intentions = intentions

        self.chain = llm | self.template_contextBot

        if self.debug:
            pp.printy("---- Chat Validate -----")
            print(f"Context Bot: {file_context}")
            print("------------------------")

    def invoke(self, text):
        if self.debug:
            print("\n****** Start CHAT LLM model ******")
        
        t1_llm = time.time()
        result = self.chain.invoke( text )
        t2_llm = time.time()

        time_llm = round(t2_llm - t1_llm,3)
        if self.debug:
            print("****** Finish CHAT LLM model ******")
            print(f"Time: \n-LLM: {time_llm}")
        """
        if not result in self.intentions:
            r = "indefinido"
        """
        print(result)

class StructuredLLM:
    def __init__(self, model:object, outputs: dict, name_parser:str = "StructureLLM"):
        self.name = name_parser
        try: 
            self.parser = create_model(name_parser, **outputs)
        except Exception as e:
            print(e)
            print("Error in generate Structured LLM - Load BaseModel")

        self.chain = model.with_structured_output(self.parser)
        
    def apply(self, text):
        try:
            response = self.chain.invoke( text )
        except Exception as e:
            print(e)
            print(f"Error in Apply {self.name} Structured LLM")

        return response
