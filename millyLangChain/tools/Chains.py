from millyLangChain.utils.helpers import load_prompt
from millyLangChain.utils import print_helpers as pp

import time


class LLM_Clasify:
    def __init__(self, llm, file_context:str, intentions:list, debug:bool = False) -> None:
        
        self.template_contextBot = load_prompt(file = file_context)
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
