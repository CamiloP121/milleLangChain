from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

from milleLangChain.utils import print_helpers as pp
from milleLangChain.utils.helpers import load_pickle
from pathlib import Path


class VectorStore():
    def __init__(self, embbeding: object, vectorstore_type: str = "faiss" , output_path: str = "store" , debug: bool = False, ) -> None:
        """
        Initializes the VectorStore class.
        ----------------------------------------------------
        Parameters:
            embedding (object): Embedding model or method to use.
            vectorstore_type (str): Type of vector store, either "faiss" or "chroma". Default is "faiss".
            output_path (str): Path to store the vector data. Default is "store".
            debug (bool): Flag to enable debug mode. Default is False.

        Raises:
            Exception: If vectorstore_type is not one of the allowed types.
        """

        if debug: 
            pp.printy("--- Vector Store ---")
            print("set: ", vectorstore_type)

        allowed_types = ["faiss", "chroma"]

        if "faiss" in vectorstore_type:
            self.class_vector = FAISS
        elif "chroma" in vectorstore_type:
            # self.class_vector = Chroma
            raise Exception("Not implemented yet Chroma")
        else:
            raise Exception("embbeding type must be one of: " + " or ".join(allowed_types))
        
        Path(output_path).mkdir(parents=True, exist_ok=True)

        self._type = vectorstore_type
        self.output_path = output_path
        self.embbeding = embbeding
        self.db = None

        if debug:
            pp.printy("----------------")

    def load_docs(self, docs):
        """
        Loads a list of documents into the vector store.
        ---------------------------------------------------------------
        Parameters:
            docs (list): A list of documents to be loaded into the vector store.

        Raises:
            ValueError: If 'docs' is not a list.
            Exception: If there is an error during the loading process.

        This method creates a vector store from the given documents using the specified embedding model.
        """
        
        if self.debug: pp.printy("Load Data base....")
        try:
            if "faiss" in self.vectorstore_type:
                self.db = self.class_vector.from_documents(documents = docs, embbeding = self.embbeding)
        except Exception as e:
            print(e)
            raise Exception("Error load Data base vector store")

        if self.debug: pp.printg("Completed load Data base")

    def load_pkl(self, pkl_path):
        """
        Loads vector store data from a .pkl file.

        Parameters:
            pkl_path (str): The path to the .pkl file that contains serialized vector store data.

        Raises:
            ValueError: If the file is not a .pkl file.
            Exception: If there is an error during the loading process.

        This method deserializes and loads the vector store from a .pkl file using the specified embedding model.
        """
        
        if self.debug: pp.printy("Load Data base....")
        try:
            if "faiss" in self.vectorstore_type:
                self.db = self.class_vector.deserialize_from_bytes(serialized = load_pickle(pkl_path), embbeding = self.embbeding)
        except Exception as e:
            print(e)
            raise Exception("Error load Data base vector store")

        if self.debug: pp.printg("Completed load Data base")
