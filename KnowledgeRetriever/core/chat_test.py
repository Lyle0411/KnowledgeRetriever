from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

class LLM_CHAT:
    def __init__(self, large_language_model):
        self.llm = ChatOpenAI(model=large_language_model, temperature=1)

        self.chain = (PromptTemplate(template="""{query}""",
                                     input_variables=["query"])
                      | self.llm)

    def chat(self, query: str):
        res = self.chain.invoke({"query": query}).content
        return res