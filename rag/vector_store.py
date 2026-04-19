from langchain_chroma import Chroma

from utils.config_handler import chroma_conf
from model.factory import embeddings_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import re
from datetime import datetime
from utils.path_tools import get_abs_path
from utils.file_handler import txt_loader, pdf_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from langchain_core.documents import Document


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf['collection_name'],
            persist_directory=get_abs_path(chroma_conf['persist_directory']),
            embedding_function=embeddings_model,
        )

        self.structured_conf = chroma_conf.get('structured_ingestion', {})
        self.default_chunk_size = chroma_conf['chunk_size']
        self.default_chunk_overlap = chroma_conf['chunk_overlap']
        self.default_separators = chroma_conf['separator']

        self.spliter = self._build_splitter(self.default_chunk_size, self.default_chunk_overlap)

    def _build_splitter(self, chunk_size: int, chunk_overlap: int):
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.default_separators,
            length_function=len
        )

    def _resolve_source_type(self, file_path: str) -> str:
        file_name = os.path.basename(file_path).lower()
        source_type_keywords = self.structured_conf.get('source_type_keywords', {})

        for source_type, keywords in source_type_keywords.items():
            for keyword in keywords:
                if keyword.lower() in file_name:
                    return source_type

        return 'general'

    def _get_splitter_by_source_type(self, source_type: str):
        chunk_strategy = self.structured_conf.get('chunk_strategy', {})
        strategy = chunk_strategy.get(source_type, chunk_strategy.get('general', {}))

        chunk_size = strategy.get('chunk_size', self.default_chunk_size)
        chunk_overlap = strategy.get('chunk_overlap', self.default_chunk_overlap)

        return self._build_splitter(chunk_size, chunk_overlap)

    def _infer_product_from_file_name(self, file_path: str) -> str:
        file_name = os.path.basename(file_path)

        if '扫拖一体' in file_name:
            return '扫拖一体机器人'
        if '扫地' in file_name:
            return '扫地机器人'
        return '通用'

    def _enrich_metadata(self, documents: list[Document], source_type: str, file_path: str, file_md5: str):
        title = os.path.basename(file_path)
        product = self._infer_product_from_file_name(file_path)
        doc_id = file_md5 if file_md5 else title

        try:
            updated_at = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
        except OSError:
            updated_at = datetime.now().isoformat()

        enriched_docs = []
        for idx, doc in enumerate(documents, start=1):
            metadata = dict(doc.metadata or {})
            metadata.update({
                'doc_id': doc_id,
                'source_type': source_type,
                'title': title,
                'product': product,
                'version': 'v1',
                'section': f'{source_type}_chunk_{idx}',
                'updated_at': updated_at,
            })
            enriched_docs.append(Document(page_content=doc.page_content, metadata=metadata))

        return enriched_docs

    def _split_faq_documents(self, documents: list[Document], source_type: str):
        faq_docs: list[Document] = []
        question_pattern = r'(?=(?:^|\n)\s*(?:Q[:：]|问[:：]|问题\s*\d*[:：]))'

        for doc in documents:
            content = (doc.page_content or '').strip()
            if not content:
                continue

            faq_parts = [part.strip() for part in re.split(question_pattern, content, flags=re.MULTILINE) if part.strip()]

            if len(faq_parts) > 1:
                for part in faq_parts:
                    faq_docs.append(Document(page_content=part, metadata=dict(doc.metadata or {})))
            else:
                splitter = self._get_splitter_by_source_type(source_type)
                faq_docs.extend(splitter.split_documents([doc]))

        return faq_docs

    def _split_documents_by_source_type(self, documents: list[Document], source_type: str):
        if source_type == 'faq':
            return self._split_faq_documents(documents, source_type)

        splitter = self._get_splitter_by_source_type(source_type)
        return splitter.split_documents(documents)

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf['k']})

    def load_document(self):
        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf['md5_hex_store'])):
                open(get_abs_path(chroma_conf['md5_hex_store']), 'w', encoding='utf-8').close()
                return False

            with open(get_abs_path(chroma_conf['md5_hex_store']), 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True
                return False

        def save_md5(md5_for_check: str):
            with open(get_abs_path(chroma_conf['md5_hex_store']), 'a', encoding='utf-8') as f:
                f.write(md5_for_check + '\n')

        def get_file_documents(read_path: str):
            if read_path.endswith('txt'):
                return txt_loader(read_path)

            if read_path.endswith('pdf'):
                return pdf_loader(read_path)
            return []

        allowed_files_path = listdir_with_allowed_type(get_abs_path(chroma_conf['data_path']), tuple(chroma_conf['allow_knowledge_file_type']))
        structured_enabled = self.structured_conf.get('enabled', False)

        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已存在知识库内")
                continue

            try:
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库]{path}内容为空")
                    continue

                if structured_enabled:
                    source_type = self._resolve_source_type(path)
                    split_document: list[Document] = self._split_documents_by_source_type(documents, source_type)
                    split_document = self._enrich_metadata(split_document, source_type, path, md5_hex)
                    logger.info(f"[加载知识库]{path}结构化入库类型：{source_type}，切片数量：{len(split_document)}")
                else:
                    split_document = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}切分后内容为空")
                    continue

                self.vector_store.add_documents(split_document)

                save_md5(md5_hex)
                logger.info(f"[加载知识库]{path}加载成功")
            except Exception as e:
                logger.error(f"[加载知识库]{path}失败，错误信息：{str(e)}", exc_info=True)


if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()
    res = retriever.invoke("无人机")

    for r in res:
        print(r.page_content)
        print("="*20)
