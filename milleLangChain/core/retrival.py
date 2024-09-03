from langchain.retrievers.document_compressors import DocumentCompressorPipeline # Forma de aplicar los metodos
from langchain.retrievers.document_compressors import LLMChainFilter # Verificador de documentos relevantes
from langchain.retrievers.document_compressors import LLMChainExtractor # Extractor de información relevante
from langchain.retrievers import ContextualCompressionRetriever

class Retriever():
   def __init__(self):
      self.retriever = None
   
   def create_simple_retriver(self, vectorstore: object, search_type:str = "similarity", search_kwargs:dict = {}):
      if "faiss" in vectorstore._type:
         self.retriever = vectorstore.db.as_retriever(search_type = search_type,
                                 search_kwargs = search_kwargs, sources = True)
      print("---------RAG Created---------")

   def create_rerank(self, model):
      
      pipeline_compressor = DocumentCompressorPipeline(
         transformers=[LLMChainExtractor.from_llm(model),
                     LLMChainFilter.from_llm(model)]
      )

      self.retriever = compression_retriever = ContextualCompressionRetriever(
         base_compressor=pipeline_compressor, base_retriever= self.retriever
      )
   def invoke(self, text):
      print("\n****** Start RAG model ******")
      try:
         result = self.retriever.invoke(text)
      except Exception as e:
         print(e)
         raise Exception("Error apply retriever search")
      print("\n****** Finish RAG model ******")
      return result
      
