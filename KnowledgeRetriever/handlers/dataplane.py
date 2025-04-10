import io
import json
import os

from ..settings import Settings
from ..schemas import input_schemas, output_schemas
from ..errors import ServerError

from ..core.knowledge_retriever import KnowledgeRetriever
from ..core.knowledgeGraphExtractor import KnowledgeGraphExtractor
class DataPlane(object):
    """
    Internal implementation of handlers, used by REST servers.
    """
    def __init__(self, settings: Settings):

        if os.getenv("OPENAI_API_KEY") == None:
            os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
        if os.getenv("OPENAI_API_BASE") == None:
            os.environ["OPENAI_API_BASE"] = settings.OPENAI_API_BASE

        self._settings = settings
        self.kr = KnowledgeRetriever(topic = self._settings.DEFAULT_KR_TOPIC)

    async def knowledgeRetriever(
            self, Input: input_schemas.TemplateIn1
    ) -> output_schemas.TemplateOut1:

        query = Input.query
        res = self.kr.fast_retrieval(query, "Standard")
        res = self.kr.resultChek(query, res)
        return output_schemas.TemplateOut1(result=res)

    async def create_graph(self, subject, title, topic, save_path: str):
        kg = KnowledgeGraphExtractor(
            Subject=subject,
            title=title,
            topic=topic
        )
        try:
            kg.process_data(input_file=save_path)
            return {"message": "ok"}
        except Exception as e:
            return {"error": str(e)}











