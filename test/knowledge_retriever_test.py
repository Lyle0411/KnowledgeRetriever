from KnowledgeRetriever.core.knowledge_retriever import KnowledgeRetriever
import asyncio
async def kr_test(query:str):
    kr = KnowledgeRetriever(topic = "煤矿生产监管检查")
    res = kr.fast_retrieval(query, "Standard")
    print(f"案例：{query}")
    print(f"条例：")
    check_result = kr.resultChek(query,res)
    print(check_result)
async def main():
    loop = asyncio.get_event_loop()
    task = loop.create_task(kr_test(query="矿井核定生产能力90万吨/年，2024年1月生产原煤10.34万吨"))
    await task

if __name__ == '__main__':
    asyncio.run(main())