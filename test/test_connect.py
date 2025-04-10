from py2neo import Graph
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件（如果使用）
import os
NEO4J_BOLT = os.getenv("NEO4J_BOLT")  # 检查变量名是否一致
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
print(NEO4J_BOLT, NEO4J_USERNAME, NEO4J_PASSWORD)  # 调试输出



graph = Graph("bolt://localhost:7687", auth=("neo4j", "wjx123456"), name="neo4j")
print(graph.run("RETURN 1").data())

