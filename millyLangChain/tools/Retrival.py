from langchain_community.vectorstores.faiss import (
    FAISS,
)
import os
import joblib

from utils import print_helpers as pp


class VectorStore:
   def __init__(self, debug: bool = False, type_store: str = "faiss", load_path: str = None) -> None:
    """
    Initializes a VectorStore object.
    
    Args:
     debug (bool, optional): Whether to print debug information. Defaults to False.
     type_store (str, optional): The type of vector store. Currently, only "faiss" is supported. Defaults to "faiss".
    
    Raises:
     Exception: If no files are found in the Vector Store directory.
    
    Returns:
     None
    """
    self.debug = debug
    self.path = "store"
    self._type = type_store
    self.load_path = load_path
    self.db = None
    
    if debug: pp.printy("----------------")


   def load_vector_store(self, embedding):
      """
      Loads the vector store.

      Raises:
         AssertionError: If the specified type_store is not supported.
         Exception: If an error occurs while loading the vector store.

      Returns:
         object: The loaded vector store.
      """
      if self.debug:
         pp.printy("--- Load Vector Store ---")
         print("set: ", self._type)
      
      allowed_types = ["faiss"]
      assert self._type in allowed_types, "Vector Store type must be one of: " + " or ".join(allowed_types)

      if self._type == "faiss":
         try:
            if self.load_path:
                db = FAISS.deserialize_from_bytes(embeddings=embedding, 
                                                  serialized= joblib.load(self.load_path)
                                                 ).as_retriever(search_type="similarity",
                                                                search_kwargs={'k': 5, 'lambda_mult': 0.5})
            else:
                files_path = os.listdir(self.path)
                if not files_path:
                    raise Exception("Don't have files in Vector Store")
                db = FAISS.load_local(folder_path=self.path, 
                                      embeddings=embedding
                                      ).as_retriever(search_type="similarity",
                                                                search_kwargs={'k': 5, 'lambda_mult': 0.5})
         except Exception as e: 
            print(e)
            raise Exception("Error saving vector store")
         
      pp.printg(f"Completed load vector store: {self.path}")

      return db
    


   def generate_vector_store(self, docs: object, embedding: object):
      """
      Generates and saves a vector store.

      Args:
         docs (object): The documents to generate vectors from.
         embedding (object): The embedding object used to generate vectors.
         path (str): The path to save the vector store.
         type_store (str, optional): The type of vector store. Currently, only "faiss" is supported. Defaults to "faiss".
         debug (bool, optional): Whether to print debug information. Defaults to False.

      Raises:
         AssertionError: If the specified type_store is not supported.

      Returns:
         None
      """
      if self.debug:
         pp.printy("--- Vector Store ---")
         print("set: ", self._type)
      
      allowed_types = ["faiss"]
      assert self._type in allowed_types, "Vector Store type must be one of: " + " or ".join(allowed_types)

      if self._type == "faiss":
         try: 
            db = FAISS.from_documents(documents = docs, embedding = embedding)   
            db.save_local(self.path)
         except Exception as e: 
            print(e)
            raise Exception("Error saving vector store")
         
      