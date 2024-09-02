class Retriever():
   def __init__(self, vectorstore: object, search_type:str = "similarity", search_kwargs:dict = {}):
      if "faiss" in vectorstore._type:
         self.retriever = vectorstore.db.as_retriever(search_type = search_type,
                                 search_kwargs = search_kwargs)
         
   def invoke(self, text):
      try:
         result = self.retriever.invoke(text)
      except Exception as e:
         print(e)
         raise Exception("Error apply retriever search")
      
      return result
      
