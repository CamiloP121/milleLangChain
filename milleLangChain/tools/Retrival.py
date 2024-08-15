class SimpleRetriever():
   def __init__(self, vectorstore: object, search_type:str = "similarity", search_kwargs:dict = {}):
      if "faiss" in vectorstore._type:
         self.retriever = vectorstore.db.as_retriever(search_type = search_type,
                                 search_kwargs = search_kwargs)
         
class MultiRetriever():
   pass
      
