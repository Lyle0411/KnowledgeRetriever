import os
from dotenv import load_dotenv
load_dotenv()

def getPath():
    res = os.path.abspath(os.path.dirname(__file__))
    rootPath = res[:res.find(os.getenv('PROJECT_NAME')) + len(os.getenv('PROJECT_NAME'))]
    return rootPath

