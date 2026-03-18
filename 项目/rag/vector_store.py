from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore

from utils.config_handler import chroma_conf
from model.factory import embeddings_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from utils.path_tools import get_abs_path
from utils.file_handler import txt_loader, pdf_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from langchain_core.documents import Document


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf['collection_name'],
            persist_directory=chroma_conf['persist_directory'],
            embedding_function=embeddings_model,
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf['chunk_size'],
            chunk_overlap=chroma_conf['chunk_overlap'],
            separators=chroma_conf['separator'],
            length_function=len
        )

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

                split_document: list[Document] = self.spliter.split_documents(documents)

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
