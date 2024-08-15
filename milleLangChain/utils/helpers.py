import os
import joblib

def load_prompt(file: str):
    """
    Loads the content of a .txt file into a string.
    ------------------------------------------------
    Args:
     file (str): The path to the .txt file to be loaded.
    
    Raises:
     AssertionError: If the file extension is not .txt.
     Exception: If the file cannot be read, with an additional message indicating the specific file.
    
    Returns:
     str: The content of the file as a string.
    """

    _, ext = os.path.splitext(file)

    assert ext == ".txt", "Extension not allowed. Valid extention .txt" 

    template  = ""
    try:
        with open(file, 'r') as f:
            template += str(f.read())+"\n"
    except Exception as e:
        print(e)
        raise Exception (f"Could not read file {file}")
    return template


def load_pickle(file: str):
    """
    Loads the content of a .pkl file into an object.
    ------------------------------------------------
    Args:
     file (str): The path to the .pkl file to be loaded.
    
    Raises:
     AssertionError: If the file extension is not .pkl.
     Exception: If the file cannot be read, with an additional message indicating the specific file.
    
    Returns:
     object: The content of the file as a Python object.
    """
    
    # Check if the file has the correct .pkl extension
    _, ext = os.path.splitext(file)
    assert ext == ".pkl", "Extension not allowed. Valid extension is .pkl"
    
    # Try to load the .pkl file
    try:
        data = joblib.load(file)
    except Exception as e:
        print(e)
        raise Exception(f"Could not read file {file}")
    
    return data

def save_pickle(data: object, file: str):
    """
    Saves the content into a .pkl file.
    -----------------------------------
    Args:
     data (object): The Python object to be saved.
     file (str): The path to the .pkl file where the data will be saved.
    
    Raises:
     AssertionError: If the file extension is not .pkl.
     Exception: If the file cannot be saved, with an additional message indicating the specific file.
    """
    
    # Check if the file has the correct .pkl extension
    _, ext = os.path.splitext(file)
    assert ext == ".pkl", "Extension not allowed. Valid extension is .pkl"
    
    # Try to save the data into the .pkl file
    try:
        joblib.dump(data, file)
    except Exception as e:
        print(e)
        raise Exception(f"Could not save file {file}")