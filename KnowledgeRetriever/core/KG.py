import py2neo
from dotenv import load_dotenv
import os
load_dotenv()

class KG():
    graph: py2neo.database.Graph = (
        py2neo.Graph(os.getenv('NEO4J_BOLT'),
                     auth = (os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))))
    nodeMatcher: py2neo.NodeMatcher = py2neo.NodeMatcher(graph)
    relaMatcher: py2neo.RelationshipMatcher = py2neo.RelationshipMatcher(graph)

    def neo4j(self, cql:str):
        return self.graph.run(cql).data()

    def clear(self):
        self.neo4j('MATCH (n) DETACH DELETE n')

    def node_labels(self):
        return self.graph.schema.node_labels

    def relationship_types(self):
        return self.graph.schema.relationship_types

    def schema(self):
        return self.neo4j('CALL db.schema.visualization()')

    def create_node(self, label:str, attrs:dict):
        node = self.match_node(label, attrs)

        if node is None:
            node = py2neo.Node(label, **attrs)
            self.graph.create(node)
            return node
        else:
            print('The corresponding entity node already exists!')
            print('Merging...')
            node = self.update_node_attrs(label, dict(node), attrs)
            return node

    def create_relationship(self, label1:str, attrs1:dict, label2:str, attrs2:dict, r_name:str):
        value1 = self.match_node(label1, attrs1)
        value2 = self.match_node(label2, attrs2)
        if value1 is None or value2 is None:
            print(label1, attrs1, label2, attrs2)
            print('The entity involved in the relationship was not found!')
        r = py2neo.Relationship(value1, r_name, value2)
        self.graph.merge(r)
        return r

    def update_node_attrs(self, label:str, attrs_old:dict, attrs_new:dict):
        node = self.match_node(label, attrs_old)
        if node:
            node.update(attrs_new)
            self.graph.push(node)
            return node
        else:
            print('The corresponding entity node was not found!')

    def update_node_label(self, label_old:str, label_new:str):
        res = self.nodeMatcher.match(label_old).all()
        for item in res:
            item.graph
            item.clear_labels()
            item.add_label(label_new)
            self.graph.push(item)

    def update_rel_type(self, type_old:str, type_new:str):
        res = self.match_rela(type_old)
        for item in res:
            temp = item.nodes
            self.graph.separate(item)
            self.create_relationship(str(temp[0].labels)[1:], dict(temp[0]),
                                   str(temp[1].labels)[1:], dict(temp[1]),
                                   type_new)

    def delete_node(self, label:str, attrs:dict):
        node = self.match_node(label, attrs)
        if node:
            self.graph.delete(node)
        else:
            print('The corresponding entity node was not found!')

    def match_node(self, label:str, attrs:dict):
        n = "_.name=~" + "\"" + attrs["name"] + "\""
        return self.nodeMatcher.match(label).where(n).first()

    def match_rela(self, type:str):
        return self.relaMatcher.match(r_type=type)