
import sys
import json
import re

from KG import KG
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()
llm = ChatOpenAI(model=os.getenv('DEFAULT_LARGE_MODEL'), temperature=0.0)
chain = (PromptTemplate(template="""{query}""",
                        input_variables=["query"])
         | llm)
embeddings = OpenAIEmbeddings(model = os.getenv('DEFAULT_EMBEDDING_MODEL'),
                              disallowed_special=())
def qa(query:str) -> str:
    return chain.invoke({'query': query}).content

class KnowledgeGraphExtractor:
    def __init__(self, Subject = "Qualitative", title = "WGSL", topic = "案件"):
        # 初始化知识图谱
        self.kg = KG()
        self.title = title

        # 目标主体type命名
        self.Subject = Subject
        self.SubjectKw = f"{self.Subject}_kw"
        self.topic = topic

        # 读取提示词模板
        self.entity_prompt = self.read_prompt("prompt/entity_extraction.txt")
        self.relation_prompt = self.read_prompt("prompt/relationship_extraction.txt")

        # 已处理文件记录
        self.progress_file = f"data/{Subject}_processed_files.txt"
        self.processed_files = self.load_progress()

    def load_progress(self):
        """加载已处理的文件列表"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, "r", encoding="utf-8") as f:
                return set(line.strip() for line in f)
        return set()

    def save_progress(self, item_id):
        """记录已处理的文件"""
        os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)
        with open(self.progress_file, "a", encoding="utf-8") as f:
            f.write(f"{item_id}\n")

    @staticmethod
    def read_prompt(file_path):
        """读取提示词模板"""
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    @staticmethod
    def read_json_file(file_path):
        """读取JSON文件"""
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

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
            print(f"解析JSON失败: {json_str}")
            return {}

    def chat_with_LLM(self, messages):
        """与LLM交互"""
        try:
            response = qa(messages)
            return response

        except Exception as e:
            print(f"LLM调用出错: {str(e)}")
            raise

    def extract_entities(self, text):
        """提取实体"""
        messages = f"{self.entity_prompt}\n请从以下文本中提取与{self.topic}相关的核心实体：\n\n{text}"
        response = self.chat_with_LLM(messages)
        return self.parse_ai_response(response)

    def extract_relations(self, text, entities):
        """提取关系"""
        entities_str = ", ".join(entities)
        messages = f"{self.relation_prompt}\n已知实体列表：{entities_str}\n\n请从以下文本中提取这些实体之间的关系：\n\n{text}"
        response = self.chat_with_LLM(messages)
        return self.parse_ai_response(response)

    def process_item(self, item_id, item_data):
        """处理单个数据项"""
        try:
            title = item_data.get("title", "")
            clusters = item_data.get("clusters", [])

            print(f"正在处理数据 {item_id}: {title}")
            print(f"该数据包含 {len(clusters)} 个线索簇")

            self.kg.create_node(self.Subject, {"name": title,
                                               "content": '\n'.join([item["comments"] for item in clusters])})

            entity_contents = {}
            all_relations = []

            for i, cluster in enumerate(clusters):
                print(f"处理第 {i+1}/{len(clusters)} 个线索簇")

                comments = [
                    comment.replace("\n", " ").strip()
                    for comment in cluster.get("comments", [])
                ]
                context = f"案件：{title}\n所有线索：\n" + "\n".join(comments)

                entities_result = self.extract_entities(context)
                entities = entities_result.get("entities", [])

                if not entities:
                    continue

                relations_result = self.extract_relations(context, entities)
                relations = relations_result.get("relations", [])

                if not relations:
                    continue

                content_unit_title = ", ".join(entities)
                content_unit = context

                for entity in entities:
                    if entity not in entity_contents:
                        entity_contents[entity] = []
                    entity_contents[entity].append((content_unit_title, content_unit))

                all_relations.extend(relations)

            if not entity_contents or not all_relations:
                print(f"数据 {item_id} 没有提取到有效的实体和关系")
                return False

            print(
                f"向知识图谱添加 {len(entity_contents)} 个实体和 {len(all_relations)} 个关系"
            )

            for entity, content_units in entity_contents.items():
                self.kg.create_node(self.SubjectKw, {"name": entity,
                                                   "trails":str(content_units)})
                self.kg.create_relationship(self.SubjectKw, {"name": entity},
                                            self.Subject, {"name": title},
                                            "相关场景")

            for relation in all_relations:
                self.kg.create_relationship(self.SubjectKw, {"name": relation['source']},
                                            self.SubjectKw, {"name": relation['target']},
                                            relation['relation'])
            return True

        except Exception as e:
            print(f"处理数据时出错: {str(e)}")
            return False

    def merge_similar_entities(self) -> None:
        """自动检查并合并相似实体"""
        print("\n开始检查和合并相似实体...")

        # 获取所有实体对的相似度
        entity_pairs = []
        entities = [item['n']['name'] for item in self.kg.neo4j(f"""MATCH (n:`{self.SubjectKw}`) RETURN n""")]

        for i, entity1 in enumerate(entities):
            embedding1 = embeddings.embed_query(entity1)
            for entity2 in entities[i + 1 :]:
                embedding2 = embeddings.embed_query(entity2)
                similarity = cosine_similarity([embedding1], [embedding2])[0][0]
                if similarity > 0.75:
                    if self._llm_merge_judgment(entity1, entity2):
                        entity_pairs.append((entity1, entity2, similarity))

        # 按相似度排序
        entity_pairs.sort(key=lambda x: x[2], reverse=True)

        # 执行合并
        merged_entities = set()
        for entity1, entity2, similarity in entity_pairs:
            if entity1 not in merged_entities and entity2 not in merged_entities:
                print(f"\n合并实体：{entity1} 和 {entity2}（相似度：{similarity:.3f}）")
                merged_id = self.merge_entities(entity1, entity2)
                merged_entities.add(entity2 if merged_id == entity1 else entity1)

    def _llm_merge_judgment(self, entity1: str, entity2: str) -> bool:
        """使用LLM判断两个实体是否应该合并"""
        try:
            with open("prompt/entity_merge.txt", "r", encoding="utf-8") as file:
                template = file.read()

            prompt = template.format(entity1=entity1, entity2=entity2)

            response = qa(prompt)
            return response == "yes"

        except Exception as e:
            print(f"LLM判断发生错误: {str(e)}")
            return False

    def merge_entities(self, entity_id1: str, entity_id2: str) -> str:
        """
        手动合并两个实体

        Args:
            entity_id1: 第一个实体ID
            entity_id2: 第二个实体ID

        Returns:
            str: 合并后的主实体ID
        """

        # 选择保留第一个实体作为主实体
        main_entity = entity_id1
        merged_entity = entity_id2 if main_entity == entity_id1 else entity_id1

        # 合并内容
        merged_attrs = {'name': merged_entity,
                        'trails': eval(self.kg.neo4j(f"""MATCH (n:`{self.SubjectKw}`) WHERE n.name = "{merged_entity}" RETURN n""")[0]['n']['trails'])}
        self._merge_entity_content(main_entity, merged_attrs)


        # 合并关系
        self._merge_entity_relationships(main_entity, merged_entity)


        # 删除被合并的实体
        merged_attrs['trails'] = str(merged_attrs['trails'])
        self.kg.delete_node(self.SubjectKw, merged_attrs)


        return main_entity

    def _merge_entity_content(self, main_id: str, attrs: dict) -> None:
        """合并实体内容"""
        existing_attrs = {'name': main_id,
                          'trails': eval(self.kg.neo4j(f"""MATCH (n:`{self.SubjectKw}`) WHERE n.name = "{main_id}" RETURN n""")[0]['n']['trails'])}

        # 使用集合去重
        existing_set = {
            (title.strip(), content.strip()) for title, content in existing_attrs['trails']
        }
        new_set = {(title.strip(), content.strip()) for title, content in attrs['trails']}
        merged_set = existing_set.union(new_set)
        existing_attrs['trails'] = str(merged_set)

        old_attrs = dict(self.kg.neo4j(f"MATCH (n:`{self.SubjectKw}`) WHERE n.name = '{main_id}' RETURN n")[0]['n'])

        # 保存合并后的内容
        self.kg.update_node_attrs(self.SubjectKw, old_attrs, existing_attrs)

    def _merge_entity_relationships(self, main_id: str, merged_id: str) -> None:
        """合并实体的关系"""
        # 处理入边
        for predecessor in [item['t']['name'] for item in self.kg.neo4j(f"""MATCH (s:`{self.SubjectKw}`)<-[r]-(t) WHERE s.name = '{merged_id}' RETURN t""")]:
            if predecessor != main_id:  # 避免自环
                edges_data = [str(type(item['r']))[20:-2] for item in self.kg.neo4j(f"""MATCH (s:`{self.SubjectKw}`)<-[r]-(t) WHERE s.name = '{merged_id}' AND t.name = '{predecessor}' RETURN r""")]
                for edge_data in edges_data:
                    # 避免添加自环
                    if predecessor != main_id:
                        predecessor_label = str(self.kg.neo4j(f"""MATCH (n) WHERE n.name = "{predecessor}" RETURN n""")[0]['n'].labels)[1:]
                        self.kg.create_relationship(predecessor_label, dict(self.kg.neo4j(f"MATCH (n:`{predecessor_label}`) WHERE n.name = '{predecessor}' RETURN n")[0]['n']),
                                                    self.SubjectKw, dict(self.kg.neo4j(f"MATCH (n:`{self.SubjectKw}`) WHERE n.name = '{main_id}' RETURN n")[0]['n']),
                                                    edge_data)

        # 处理出边
        for successor in [item['t']['name'] for item in self.kg.neo4j(f"""MATCH (s:`{self.SubjectKw}`)-[r]->(t) WHERE s.name = '{merged_id}' RETURN t""")]:
            if successor != main_id:  # 避免自环
                edges_data = [str(type(item['r']))[20:-2] for item in self.kg.neo4j(f"""MATCH (s:`{self.SubjectKw}`)-[r]->(t) WHERE s.name = '{merged_id}' AND t.name = '{successor}' RETURN r""")]
                for edge_data in edges_data:
                    # 避免添加自环
                    if successor != main_id:
                        successor_label = str(self.kg.neo4j(f"""MATCH (n) WHERE n.name = "{successor}" RETURN n""")[0]['n'].labels)[1:]
                        self.kg.create_relationship(self.SubjectKw, dict(self.kg.neo4j(f"MATCH (n:`{self.SubjectKw}`) WHERE n.name = '{main_id}' RETURN n")[0]['n']),
                                                    successor_label, dict(self.kg.neo4j(f"MATCH (n:`{successor_label}`) WHERE n.name = '{successor}' RETURN n")[0]['n']),
                                                    edge_data)

    def process_data(self, input_file="data/results.json"):
        """处理数据文件"""
        try:
            data = self.read_json_file(input_file)

            unprocessed_items = [
                (f"{self.title}_{item_id}", data)
                for item_id, data in list(data.items())
                if item_id not in self.processed_files
            ]

            if not unprocessed_items:
                print("没有新的数据需要处理")
                return self.kg

            print(f"将处理 {len(unprocessed_items)} 个数据项")

            for item_id, item_data in unprocessed_items:
                print(item_id, item_data)
                if self.process_item(item_id, item_data):
                    self.save_progress(item_id)
                    self.processed_files.add(item_id)

                    print(f"数据 {item_id} 处理完成并保存")

            self.merge_similar_entities()

            print("\n数据处理完成")
            return self.kg

        except Exception as e:
            print(f"处理文件出错: {str(e)}")
            raise

    def procss_case_data(self, data):
        """数据项第一列为案例描述, 第二列为条例列表"""
        with open("prompt/kw_extraction.txt", "r", encoding="utf-8") as file: self.getKw_prompt = file.read()

        for item in data:
            case = item[0]
            messages = f"{self.getKw_prompt}\n请从以下文本中提取与{self.topic}相关的核心关键词：\n\n{case}"
            response = qa(messages)
            kws = self.parse_ai_response(response)["keywords"]
            for kw in kws:
                for title in item[1]:
                    self.kg.create_node(self.SubjectKw, {"name": kw})
                    self.kg.create_relationship(self.SubjectKw, {"name": kw},
                                            self.Subject, {"name": title},
                                            "相关场景")

def main(input_file="data/results.json", Subject="Qualitative", title  = "WGSL"):
    try:
        print("\n开始处理数据")
        extractor = KnowledgeGraphExtractor(Subject, title)
        extractor.process_data(input_file)

    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # kg = KG()
    # kg.clear()

    extractor = KnowledgeGraphExtractor(Subject = "Standard", title = "GZPD", topic = "煤矿生产监管检查")
    extractor.process_data()