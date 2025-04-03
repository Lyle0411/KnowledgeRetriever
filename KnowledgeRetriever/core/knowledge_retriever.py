from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import json
import re
from .KG import KG
from ..utils import get_logger
logger = get_logger()

import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import FAISS
load_dotenv()

from ..utils import getPath
llm = ChatOpenAI(model=os.getenv('DEFAULT_LARGE_MODEL'), temperature=0.0)
chain = (PromptTemplate(template="""{query}""",
                        input_variables=["query"])
         | llm)
embeddings = OpenAIEmbeddings(model = os.getenv('DEFAULT_EMBEDDING_MODEL'),
                              disallowed_special=())
def qa(query:str) -> str: return chain.invoke({'query': query}).content
class RetrievalMode(Enum):
    FAST = "1"  # 快速检索：直接检索最相关的内容
    ASSOCIATE = "2"  # 联想检索：基于初始检索结果进行联想
    RELATION = "3"  # 关系检索：关注实体间的关系网络
    COMMUNITY = "4"  # 社区检索：检索社区内的相关讨论

class KnowledgeRetriever:
    def __init__(self, topic = "煤矿生产监管检查"):
        """初始化知识检索服务"""
        self.retrieval_cache: Dict[str, int] = {}  # 检索结果缓存
        self.initial_cache_rounds = 5  # 缓存初始轮数
        self.topic = topic
        with open(f"{getPath()}/KnowledgeRetriever/core/prompt/kw_extraction.txt", "r", encoding="utf-8") as file:
            self.getKw_prompt = file.read()
        with open(f"{getPath()}/KnowledgeRetriever/core/prompt/judge.txt", "r", encoding="utf-8") as file:
            self.resjudge_prompt = file.read()

        try:
            logger.info(f"正在加载知识图谱...")
            self.kg = KG()
            self.RetrieveLocalVectorStore()
            logger.info(f"知识图谱加载成功！")
        except Exception as e:
            logger.error(f"加载知识图谱时出错: {str(e)}")
            self.kg = None

    def resultChek(self, query, res):
        result = {}
        for item in res:
            if not item[1][0]['name'] in result:
                if qa(self.resjudge_prompt.format(describe=query, regulation=item[1][0]['content'])) == "yes":
                    result[item[1][0]['name']] = item[1][0]['content']
        return result
    def RetrieveLocalVectorStore(self):
        labels = [item for item in self.kg.node_labels()]
        self.Retriever = {}
        try:
            logger.info(f"载入本地向量数据库...")
            for label in labels:
                vectorstore = FAISS.load_local(f"{getPath()}/KnowledgeRetriever/core/faiss/{label}", embeddings, allow_dangerous_deserialization=True)
                self.Retriever[label] = vectorstore.as_retriever()
            logger.info(f"载入本地向量数据库完成")
        except:
            logger.info(f"未发现本地向量数据库，正在创建...")
            for label in labels:
                res = [item['n']['name'] for item in self.kg.neo4j(f"match (n:`{label}`) return n")]
                vectorstore = FAISS.from_texts(res, embeddings)
                vectorstore.save_local(f"{getPath()}/KnowledgeRetriever/core/faiss/{label}")
                self.Retriever[label] = vectorstore.as_retriever()
            logger.info(f"本地向量数据库，创建完成")

    def getEdge(self, entity_id1: str, entity_id2: str, entity_type1: str="", entity_type2:str="") -> str:
        if entity_type1 and entity_type2:
            edges_data = [str(type(item['r']))[20:-2] for item in self.kg.neo4j(
                f"""MATCH (s:`{entity_type1}`)-[r]->(t:`{entity_type2}`) WHERE s.name = '{entity_id1}' AND t.name = '{entity_id2}' RETURN r""")]
        elif entity_type1 and not entity_type2:
            edges_data = [str(type(item['r']))[20:-2] for item in self.kg.neo4j(
                f"""MATCH (s:`{entity_type1}`)-[r]->(t) WHERE s.name = '{entity_id1}' AND t.name = '{entity_id2}' RETURN r""")]
        elif not entity_type1 and entity_type2:
            edges_data = [str(type(item['r']))[20:-2] for item in self.kg.neo4j(
                f"""MATCH (s)-[r]->(t:`{entity_type2}`) WHERE s.name = '{entity_id1}' AND t.name = '{entity_id2}' RETURN r""")]
        else:
            edges_data = [str(type(item['r']))[20:-2] for item in self.kg.neo4j(
                f"""MATCH (s)-[r]->(t) WHERE s.name = '{entity_id1}' AND t.name = '{entity_id2}' RETURN r""")]
        return '\n'.join(edges_data)

    def getNode(self, entity:str, entity_type: str=""):
        if entity_type:
            return [item['n'] for item in self.kg.neo4j(f"MATCH (n:`{entity_type}`) where n.name = '{entity}' return n ")]
        else:
            return [item['n'] for item in self.kg.neo4j(f"MATCH (n) where n.name = '{entity}' return n ")]

    def getKw(self, case):
        messages = f"{self.getKw_prompt}\n请从以下文本中提取与{self.topic}相关的核心关键词：\n\n{case}"
        response = qa(messages)
        return self.parse_ai_response(response)["keywords"]

    def parse_ai_response(self, response):
        """解析AI响应"""
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.error(f"解析JSON失败: {json_str}")
            return {}

    def getSimilarEntities(self, kws: list[str], EntityLabel: str = "Qualitative_kw", k=5):
        with open(f"{getPath()}/KnowledgeRetriever/core/prompt/kw_judge.txt", "r", encoding="utf-8") as file:
            self.judge_prompt = file.read()
        entities = []
        for kw in kws:
            similar_entities = [item.page_content for item in self.Retriever[EntityLabel].invoke(kw, k=k)]
            for entity in similar_entities:
                res = qa(self.judge_prompt.format(kw1=kw, kw2=entity)) == "yes"
                if res:
                    similarity = cosine_similarity([embeddings.embed_query(kw)], [embeddings.embed_query(entity)])[0][0]
                    entities.append((kw, entity, similarity))
        if entities:
            entities.sort(key=lambda x: x[2], reverse=True)
            return entities[:min(k,len(entities))]

    def fast_retrieval(self, query: str, EntityLabel: str = "Qualitative"):
        """快速检索：直接在向量存储中搜索"""
        kws = self.getKw(query)
        try:
            entities_similarity = self.getSimilarEntities(kws, f"{EntityLabel}_kw")
            if entities_similarity:
                find_result = []
                for item in entities_similarity:
                    entity = item[1]
                    content = [item['n'] for item in self.kg.neo4j(f"""MATCH (n:`{EntityLabel}_kw`) where n.name = "{entity}" RETURN n""")]
                    scene = [item['t'] for item in self.kg.neo4j(f"""MATCH (n:`{EntityLabel}_kw`)-[r:`相关场景`]->(t) where n.name = "{entity}" RETURN t""")]
                    find_result.append((content, scene))
                return find_result

        except Exception as e:
            logger.error(f"快速检索时发生错误: {str(e)}")
        return None

    def associate_retrieval(self, query: str, entities: List[str]) -> Optional[str]:
        """
        联想检索：基于实体和关系网络进行扩展检索
        - 如果在指定实体向量库中找到结果，继续进行关联搜索
        - 如果在指定实体向量库中没有结果，则在全局向量库中搜索
        """
        try:
            retrieval_results = []
            retrieved_contents = set()  # 用于记录已检索的内容，避免重复
            found_in_entity_store = False  # 标记是否在实体向量库中找到结果

            for entity in entities:
                # 查找相似实体
                similar_entities = self.kg.search_similar_entities(
                    entity, top_n=1, threshold=0.8
                )
                if not similar_entities:
                    continue

                main_entity = similar_entities[0][0]

                # 在主实体的向量存储中搜索
                entity_results = self.kg.search_vector_store(
                    query, entity_id=main_entity, k=5
                )
                filtered_results = [
                    (doc, score) for doc, score in entity_results if score >= 0.5
                ]

                if filtered_results:
                    found_in_entity_store = True
                    selected_contents = self._get_cached_results(filtered_results)
                    for content in selected_contents:
                        # 检查内容是否已存在
                        if content not in retrieved_contents:
                            retrieved_contents.add(content)
                            retrieval_results.append(
                                f"[{main_entity}]相关内容：\n{content}"
                            )

            # 如果在实体向量库中没有找到内容，进行全局搜索并返回
            if not found_in_entity_store:
                global_results = self.kg.search_vector_store(query, k=5)
                filtered_global_results = [
                    (doc, score) for doc, score in global_results if score >= 0.5
                ]

                if filtered_global_results:
                    selected_contents = self._get_cached_results(
                        filtered_global_results
                    )
                    for content in selected_contents:
                        if content not in retrieved_contents:
                            retrieved_contents.add(content)
                            retrieval_results.append(f"[全局搜索]相关内容：\n{content}")

                return "\n\n".join(retrieval_results) if retrieval_results else None

            # 如果在实体向量库中找到了结果，继续进行关联实体搜索
            for entity in entities:
                similar_entities = self.kg.search_similar_entities(
                    entity, top_n=1, threshold=0.85
                )
                if not similar_entities:
                    continue

                main_entity = similar_entities[0][0]

                # 获取相关关系
                relationships = self.kg.search_similar_relationships(
                    query, main_entity, k=3
                )
                if not relationships:
                    continue

                # 在关联实体中搜索
                related_entities = set()  # 记录关联实体
                relations_added = set()  # 记录已添加的关系描述
                entity_relations = {}  # 记录实体对应的关系描述

                for source, relation, target, score in relationships:
                    # 构建完整的关系查询语句
                    relation_query = f"{source} 与 {target} 的关系是：{relation}"

                    # 记录关系信息
                    if relation_query not in relations_added:
                        relations_added.add(relation_query)
                        retrieval_results.append(f"[关联关系]：\n- {relation_query}")

                    # 将关系中的实体加入集合（排除主实体）并记录对应的关系描述
                    if source != main_entity:
                        related_entities.add(source)
                        entity_relations[source] = relation_query
                    if target != main_entity:
                        related_entities.add(target)
                        entity_relations[target] = relation_query

                # 在关联实体中搜索，使用关系查询语句
                for related_entity in related_entities:
                    relation_query = entity_relations[related_entity]
                    results = self.kg.search_vector_store(
                        query=relation_query, entity_id=related_entity, k=5
                    )
                    filtered_results = [
                        (doc, score) for doc, score in results if score >= 0.5
                    ]

                    if filtered_results:
                        selected_contents = self._get_cached_results(filtered_results)
                        for content in selected_contents:
                            # 检查内容是否与已有内容重复
                            content_is_unique = True
                            normalized_content = "".join(content.split())

                            for existing_content in retrieved_contents:
                                normalized_existing = "".join(existing_content.split())
                                if (
                                    normalized_content in normalized_existing
                                    or normalized_existing in normalized_content
                                ):
                                    content_is_unique = False
                                    break

                            if content_is_unique:
                                retrieved_contents.add(content)
                                retrieval_results.append(
                                    f"[关联实体 - {related_entity}]：\n{content}"
                                )

            return "\n\n".join(retrieval_results) if retrieval_results else None

        except Exception as e:
            logger.error(f"联想检索时发生错误: {str(e)}")
        return None

    def relation_retrieval(self, entities: List[str]) -> Optional[str]:
        """
        检索实体列表中每对实体之间的路径关系，对第一条路径进行向量检索
        """
        try:
            if len(entities) < 2:
                return None

            # 1. 实体匹配
            main_entities = []
            matched_indices = []  # 记录成功匹配的原始实体索引

            for i, entity in enumerate(entities):
                similar_entities = self.kg.search_similar_entities(
                    entity, top_n=1, threshold=0.85
                )
                if similar_entities:
                    main_entities.append(similar_entities[0][0])
                    matched_indices.append(i)

            if len(main_entities) < 2:
                return None

            result_parts = []
            seen_paths = set()  # 用于去重路径
            seen_contents = set()  # 用于去重内容

            # 2. 对匹配成功的实体对进行路径搜索
            for i in range(len(main_entities)):
                for j in range(i + 1, len(main_entities)):
                    # 使用匹配后的实体和对应的原始实体索引
                    entity1 = main_entities[i]
                    entity2 = main_entities[j]
                    original_entity1 = entities[matched_indices[i]]
                    original_entity2 = entities[matched_indices[j]]

                    # 避免重复路径
                    path_key = f"{min(entity1, entity2)}-{max(entity1, entity2)}"
                    if path_key in seen_paths:
                        continue
                    seen_paths.add(path_key)

                    # 搜索两个实体之间的所有路径
                    paths = self.kg.search_all_paths(entity1, entity2, max_depth=5)

                    if paths:
                        result_parts.append(
                            f"\n{original_entity1} - {original_entity2} 的关系:"
                        )

                        # 显示所有路径
                        for path_idx, path_info in enumerate(paths, 1):
                            result_parts.append(f"路径 {path_idx}:")
                            result_parts.append(
                                f"实体路径: {' -> '.join(path_info['path'])}"
                            )
                            result_parts.append("关系链:")
                            result_parts.extend(
                                f"  {rel}" for rel in path_info["relationships"]
                            )

                            # 只对第一条路径进行向量检索
                            if path_idx == 1:
                                for relationship in path_info["relationships"]:
                                    try:
                                        start_end = relationship.split("->")
                                        if len(start_end) == 2:
                                            start_part = start_end[0].strip()
                                            end_entity = start_end[1].strip()

                                            start_relation = start_part.split("-", 1)
                                            if len(start_relation) == 2:
                                                start_entity = start_relation[0].strip()
                                                relation = start_relation[1].strip()

                                                relation_query = f"{start_entity} 与 {end_entity} 的关系是：{relation}"

                                                entity_results = (
                                                    self.kg.search_vector_store(
                                                        query=relation_query,
                                                        entity_id=start_entity,
                                                        k=3,
                                                    )
                                                )

                                                filtered_results = [
                                                    (doc, score)
                                                    for doc, score in entity_results
                                                    if score >= 0.5
                                                ]
                                                if filtered_results:
                                                    selected_contents = (
                                                        self._get_cached_results(
                                                            filtered_results
                                                        )
                                                    )
                                                    for content in selected_contents:
                                                        normalized_content = "".join(
                                                            content.split()
                                                        )
                                                        if (
                                                            normalized_content
                                                            not in seen_contents
                                                        ):
                                                            seen_contents.add(
                                                                normalized_content
                                                            )
                                                            result_parts.append(
                                                                f"[{start_entity}->{end_entity}]相关内容：\n{content}\n\n"
                                                            )

                                    except Exception as e:
                                        logger.error(
                                            f"处理关系 '{relationship}' 时发生错误: {str(e)}"
                                        )
                                        continue

                        result_parts.append("-" * 50)  # 添加分隔线

            return "\n".join(result_parts) if result_parts else None

        except Exception as e:
            logger.error(f"关系检索时发生错误: {str(e)}")
        return None

    def community_retrieval(self, query: str) -> Optional[str]:
        """
        社区检索：查找相关的社区信息和全局文档中的相关表述

        Args:
            query: 用户的查询字符串

        Returns:
            Optional[str]: 返回检索结果，包括社区信息和相关表述。如果没有找到相关内容则返回 None
        """
        try:
            result_parts = []

            # 1. 社区检索
            community_results = self.kg.search_communities(query, top_n=1)
            if community_results:
                members, summary = community_results[0]
                result_parts.append("【请参考社区观点】")
                result_parts.append("相关社区成员:")
                result_parts.append(f"- {', '.join(members)}")
                result_parts.append("\n社区简介:")
                result_parts.append(summary)

            # 2. 全局文档检索
            doc_results = self.kg.search_vector_store(query, k=5)
            filtered_results = [
                (doc, score) for doc, score in doc_results if score >= 0.5
            ]

            if filtered_results:
                selected_contents = self._get_cached_results(filtered_results)
                if selected_contents:
                    if result_parts:  # 如果前面有社区结果，加一个分隔
                        result_parts.append("\n")
                    result_parts.append(
                        "【以下信息仅为表达风格参考，回答中请使用上述社区观点】"
                    )
                    result_parts.extend(selected_contents)

            # 只要有任一种检索结果就返回
            if result_parts:
                return "\n".join(result_parts)

        except Exception as e:
            logger.error(f"社区检索时发生错误: {str(e)}")
        return None

    def retrieve(
        self, mode: RetrievalMode, query: str, entities: List[str]
    ) -> Optional[str]:
        """统一的检索接口"""
        if not self.kg:
            return None

        self.update_cache_counts()

        try:
            if mode == RetrievalMode.FAST:
                return self.fast_retrieval(query)

            elif mode == RetrievalMode.ASSOCIATE:
                return self.associate_retrieval(query, entities)

            elif mode == RetrievalMode.RELATION:
                if len(entities) >= 2:
                    return self.relation_retrieval(entities)

            elif mode == RetrievalMode.COMMUNITY:
                return self.community_retrieval(query)

        except Exception as e:
            logger.error(f"检索时发生错误: {str(e)}")

        return None

if __name__ == '__main__':

    import csv
    def read_csv_standard(file_path):
        data = []
        with open(file_path, 'r', encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                data.append(row)
        return data


    data = read_csv_standard("data\历史数据样例.csv")
    with open("prompt/judge.txt", "r", encoding="utf-8") as file: judge_prompt = file.read()
    kr = KnowledgeRetriever(topic = "煤矿生产监管检查")

    score = 0
    all = 0


    for item in data:
        query = item[1]
        True_value = item[2].split("、")

        result = {}
        res = kr.fast_retrieval(query, "Standard")
        flag = False

        try:
            for item in res:
                if not item[1][0]['name'] in result:
                    if qa(judge_prompt.format(describe=query, regulation=item[1][0]['content'])) == "yes":
                        result[item[1][0]['name']] = item[1][0]['content']

            for k,v in result.items():
                if k in True_value:
                    score += 1
                    flag = True
                    break
            if not flag:
                print(f"描述:{query}")
                print(f"真值:{True_value}")
                print(f"模型:{k}")
        except:
            pass

        all += 1
        print(f"已测试样例数目: {all}, 正确率: {score / all * 100}%")