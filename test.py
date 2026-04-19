# from rag.vector_store import VectorStoreService
#
# vs = VectorStoreService()
# docs = vs.get_retriever().invoke("扫地机器人 故障")
# #
# from rag.vector_store import VectorStoreService
#
# vs = VectorStoreService()
# print("collection count =", vs.vector_store._collection.count())
#
# for i, d in enumerate(docs, 1):
#     print(f"\n--- 文档{i} ---")
#     print("source_type:", d.metadata.get("source_type"))
#     print("title:", d.metadata.get("title"))
#     print("section:", d.metadata.get("section"))
#     print("doc_id:", d.metadata.get("doc_id"))
#     print("product:", d.metadata.get("product"))
#     print("updated_at:", d.metadata.get("updated_at"))
#     print("content_preview:", d.page_content[:80])
from rag.vector_store import VectorStoreService

vs = VectorStoreService()
docs = vs.get_retriever().invoke("机器人坏了怎么办")

print("命中数量:", len(docs))
for i, d in enumerate(docs, 1):
    print(f"\n--- 命中{i} ---")
    print("source_type:", d.metadata.get("source_type"))
    print("title:", d.metadata.get("title"))
    print("section:", d.metadata.get("section"))
    print("content_preview:", d.page_content[:100])