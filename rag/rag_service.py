from collections import Counter, defaultdict
import math
import re
from typing import Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import chat_model
from rag.vector_store import VectorStoreService
from utils.config_handler import rag_conf
from utils.logger_handler import logger
from utils.prompt_loader import load_rag_prompts


def print_prompt(prompt):
    print("=" * 20)
    print(prompt.to_string())
    print("=" * 20)
    return prompt


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model

        self.query_rewrite_enabled = rag_conf.get('query_rewrite_enabled', False)
        self.multi_query_count = rag_conf.get('multi_query_count', 3)
        self.multi_query_max_docs = rag_conf.get('multi_query_max_docs', 8)

        self.hybrid_retrieval_enabled = rag_conf.get('hybrid_retrieval_enabled', False)
        self.vector_top_k = rag_conf.get('vector_top_k', 6)
        self.bm25_top_k = rag_conf.get('bm25_top_k', 6)
        self.rrf_k = rag_conf.get('rrf_k', 60)
        self.hybrid_max_candidates = rag_conf.get('hybrid_max_candidates', 12)
        self.cross_encoder_enabled = rag_conf.get('cross_encoder_enabled', False)
        self.cross_encoder_model_name = rag_conf.get('cross_encoder_model_name', 'BAAI/bge-reranker-base')
        self.final_context_docs = rag_conf.get('final_context_docs', self.multi_query_max_docs)

        self.traceable_answer_enabled = rag_conf.get('traceable_answer_enabled', False)
        self.traceable_max_sources = rag_conf.get('traceable_max_sources', 5)
        self.traceable_snippet_length = rag_conf.get('traceable_snippet_length', 120)

        self._cross_encoder: Optional[object] = None
        self._bm25_ready = False
        self._bm25_docs: list[Document] = []
        self._bm25_tf: list[Counter] = []
        self._bm25_df: Counter = Counter()
        self._bm25_doc_len: list[int] = []
        self._bm25_avgdl = 0.0
        self._bm25_k1 = 1.5
        self._bm25_b = 0.75

        self.rewrite_prompt_template = PromptTemplate.from_template(
            """
你是电商客服查询改写助手。请对用户问题进行意图识别，并生成适合知识库检索的查询。
要求：
1. 仅输出纯文本，不要输出markdown，不要解释。
2. 第一行必须是：INTENT: <意图>
3. 后续每一行必须是：QUERY: <改写查询>
4. 至少输出1条QUERY，最多输出{multi_query_count}条QUERY。
5. 查询要覆盖用户原问题、同义表达、业务术语表达。
用户问题：{input}
""".strip()
        )

        self.chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        self.rewrite_chain = self.rewrite_prompt_template | self.model | StrOutputParser()

    def _parse_rewrite_output(self, text: str) -> tuple[str, list[str]]:
        intent = "other"
        queries: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            u = line.upper()
            if u.startswith("INTENT:"):
                intent = line.split(":", 1)[1].strip() or "other"
            elif u.startswith("QUERY:"):
                q = line.split(":", 1)[1].strip()
                if q:
                    queries.append(q)

        dedup, seen = [], set()
        for q in queries:
            if q not in seen:
                seen.add(q)
                dedup.append(q)
        return intent, dedup[:self.multi_query_count]

    def rewrite_query(self, query: str) -> tuple[str, list[str]]:
        if not self.query_rewrite_enabled:
            return "disabled", [query]
        try:
            rewrite_text = self.rewrite_chain.invoke({"input": query, "multi_query_count": self.multi_query_count})
            intent, queries = self._parse_rewrite_output(rewrite_text)
            if not queries:
                logger.warning("[query_rewrite]解析失败，回退原始query")
                return "fallback", [query]
            if query not in queries:
                queries = [query] + queries
            logger.info(f"[query_rewrite]intention={intent}, queries={queries}")
            return intent, queries[:self.multi_query_count]
        except Exception as e:
            logger.warning(f"[query_rewrite]改写失败，回退原始query，错误信息：{str(e)}")
            return "fallback", [query]

    @staticmethod
    def _doc_unique_key(doc: Document):
        m = doc.metadata or {}
        return m.get('doc_id'), m.get('section'), m.get('source'), doc.page_content

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = (text or "").lower()
        words = re.findall(r"[a-z0-9]+", text)
        zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
        zh_bigrams = ["".join(zh_chars[i:i + 2]) for i in range(len(zh_chars) - 1)]
        return words + zh_chars + zh_bigrams

    def _ensure_bm25_index(self):
        if self._bm25_ready:
            return
        try:
            collection = self.vector_store.vector_store._collection
            data = collection.get(include=["documents", "metadatas"])
            documents = data.get("documents", []) or []
            metadatas = data.get("metadatas", []) or []

            self._bm25_docs, self._bm25_tf, self._bm25_doc_len = [], [], []
            self._bm25_df = Counter()

            for i, content in enumerate(documents):
                metadata = metadatas[i] if i < len(metadatas) else {}
                tokens = self._tokenize(content)
                if not tokens:
                    continue
                self._bm25_docs.append(Document(page_content=content, metadata=metadata or {}))
                tf = Counter(tokens)
                self._bm25_tf.append(tf)
                self._bm25_doc_len.append(len(tokens))
                for tk in tf.keys():
                    self._bm25_df[tk] += 1

            self._bm25_avgdl = (sum(self._bm25_doc_len) / len(self._bm25_doc_len)) if self._bm25_doc_len else 0.0
            self._bm25_ready = True
            logger.info(f"[hybrid_retrieval]BM25索引已构建，文档数：{len(self._bm25_docs)}")
        except Exception as e:
            logger.warning(f"[hybrid_retrieval]构建BM25索引失败，将降级仅向量检索，错误信息：{str(e)}")
            self._bm25_ready = False

    def _bm25_retrieve(self, query: str, top_k: int) -> list[Document]:
        self._ensure_bm25_index()
        if not self._bm25_ready or not self._bm25_docs:
            return []

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        n_docs = len(self._bm25_docs)
        scores = [0.0] * n_docs

        for q in q_tokens:
            df = self._bm25_df.get(q, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self._bm25_tf):
                f = tf.get(q, 0)
                if f == 0:
                    continue
                dl = self._bm25_doc_len[i]
                denom = f + self._bm25_k1 * (1 - self._bm25_b + self._bm25_b * dl / (self._bm25_avgdl + 1e-9))
                scores[i] += idf * (f * (self._bm25_k1 + 1)) / (denom + 1e-9)

        ranked_idx = sorted(range(n_docs), key=lambda i: scores[i], reverse=True)
        return [self._bm25_docs[i] for i in ranked_idx if scores[i] > 0][:top_k]

    def _vector_retrieve(self, query: str, top_k: int) -> list[Document]:
        retriever = self.vector_store.vector_store.as_retriever(search_kwargs={"k": top_k})
        return retriever.invoke(query)

    def _rrf_fuse(self, ranked_lists: list[list[Document]], max_candidates: int) -> list[Document]:
        score_map = defaultdict(float)
        doc_map = {}
        for docs in ranked_lists:
            for rank, doc in enumerate(docs, start=1):
                key = self._doc_unique_key(doc)
                score_map[key] += 1.0 / (self.rrf_k + rank)
                doc_map[key] = doc
        sorted_keys = sorted(score_map.keys(), key=lambda k: score_map[k], reverse=True)
        return [doc_map[k] for k in sorted_keys][:max_candidates]

    def _ensure_cross_encoder(self) -> bool:
        if not self.cross_encoder_enabled:
            return False
        if self._cross_encoder is not None:
            return True
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(self.cross_encoder_model_name)
            logger.info(f"[hybrid_retrieval]Cross-Encoder加载成功：{self.cross_encoder_model_name}")
            return True
        except Exception as e:
            logger.warning(f"[hybrid_retrieval]Cross-Encoder不可用，降级仅RRF排序，错误信息：{str(e)}")
            self._cross_encoder = None
            return False

    def _cross_encoder_rerank(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs or not self._ensure_cross_encoder():
            return docs
        try:
            pairs = [(query, d.page_content) for d in docs]
            scores = self._cross_encoder.predict(pairs)
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return [d for d, _ in ranked]
        except Exception as e:
            logger.warning(f"[hybrid_retrieval]Cross-Encoder重排失败，使用RRF结果，错误信息：{str(e)}")
            return docs

    def _hybrid_retrieve_docs(self, query: str) -> list[Document]:
        _, queries = self.rewrite_query(query)
        ranked_lists: list[list[Document]] = []

        for q in queries:
            vector_docs = self._vector_retrieve(q, self.vector_top_k)
            bm25_docs = self._bm25_retrieve(q, self.bm25_top_k)
            ranked_lists.extend([vector_docs, bm25_docs])
            logger.info(f"[hybrid_retrieval]query={q}, vector_hits={len(vector_docs)}, bm25_hits={len(bm25_docs)}")

        fused_docs = self._rrf_fuse(ranked_lists, self.hybrid_max_candidates)
        logger.info(f"[hybrid_retrieval]RRF融合命中{len(fused_docs)}条文档")

        reranked_docs = self._cross_encoder_rerank(query, fused_docs)
        final_docs = reranked_docs[:self.final_context_docs]
        logger.info(f"[hybrid_retrieval]最终上下文文档数：{len(final_docs)}")
        return final_docs

    def _vector_only_retrieve_docs(self, query: str) -> list[Document]:
        _, queries = self.rewrite_query(query)
        merged_docs, seen_keys = [], set()

        for q in queries:
            for doc in self.retriever.invoke(q):
                key = self._doc_unique_key(doc)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                merged_docs.append(doc)
                if len(merged_docs) >= self.multi_query_max_docs:
                    logger.info(f"[query_rewrite]多查询融合命中{len(merged_docs)}条文档，达到上限")
                    return merged_docs

        logger.info(f"[query_rewrite]多查询融合命中{len(merged_docs)}条文档")
        return merged_docs

    def retriever_docs(self, query: str) -> list[Document]:
        return self._hybrid_retrieve_docs(query) if self.hybrid_retrieval_enabled else self._vector_only_retrieve_docs(query)

    def _build_trace_sources(self, docs: list[Document]) -> str:
        sources, seen = [], set()
        for doc in docs:
            m = doc.metadata or {}
            title = str(m.get('title') or '未知标题')
            doc_id = str(m.get('doc_id') or 'unknown-doc')
            section = str(m.get('section') or 'unknown-section')

            key = (doc_id, section, title)
            if key in seen:
                continue
            seen.add(key)

            snippet = (doc.page_content or '').replace('\n', ' ').strip()[:self.traceable_snippet_length]
            source_id = f"S{len(sources) + 1}"
            sources.append(f"- [{source_id}] title={title} | doc_id={doc_id} | section={section} | snippet={snippet}")
            if len(sources) >= self.traceable_max_sources:
                break

        return "无可用来源" if not sources else "\n".join(sources)

    def rag_summarize(self, query: str) -> str:
        context_docs = self.retriever_docs(query)
        context = ""
        for i, doc in enumerate(context_docs, start=1):
            context += f"【参考资料{i}】:参考资料：{doc.page_content} | 参考元数据:{doc.metadata}"

        answer = self.chain.invoke({"input": query, "context": context})
        if not self.traceable_answer_enabled:
            return answer

        source_block = self._build_trace_sources(context_docs)
        return f"{answer.strip()}\n\n【参考来源】\n{source_block}"


if __name__ == '__main__':
    rag = RagSummarizeService()
    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
