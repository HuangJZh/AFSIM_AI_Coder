import os
import glob
from typing import List
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from chromadb.config import Settings

# 初始化模型 (全局加载，避免重复初始化)
print("正在加载 Embedding 模型...")
embedding_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

print("正在加载 Rerank 模型...")
# 注意：通常 rerank 使用专门的 reranker 模型效果更好
rerank_model = CrossEncoder('BAAI/bge-reranker-base') 

# 初始化 ChromaDB
chromadb_client = chromadb.PersistentClient(path="./chroma_db")
# 获取或创建集合
chromadb_collection = chromadb_client.get_or_create_collection(name="afsim_documents")

class RagUtils:
    @staticmethod
    def load_and_index_directory(directory: str):
        """
        加载目录下的文件并建立索引（如果集合为空）
        """
        if chromadb_collection.count() > 0:
            print(f"向量库已存在，包含 {chromadb_collection.count()} 个片段。跳过初始化。")
            return

        print(f"正在初始化向量库，扫描目录: {directory} ...")
        
        file_paths = []
        for ext in ['*.txt', '*.md', '*.cpp', '*.h', '*.json']:
            file_paths.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))

        all_chunks = []
        all_ids = []
        all_metadatas = []
        
        doc_count = 0
        for file_path in file_paths:
            try:
                chunks = RagUtils.split_into_chunks(file_path)
                for idx, chunk in enumerate(chunks):
                    if not chunk.strip(): continue
                    all_chunks.append(chunk)
                    all_ids.append(f"{os.path.basename(file_path)}_{idx}")
                    all_metadatas.append({"source": os.path.basename(file_path)})
                doc_count += 1
            except Exception as e:
                print(f"读取文件错误 {file_path}: {e}")

        if all_chunks:
            # 批量计算向量
            print(f"正在生成向量 (共 {len(all_chunks)} 个片段)...")
            embeddings = embedding_model.encode(all_chunks).tolist()
            
            # 批量存入 ChromaDB (分批处理以防内存溢出)
            batch_size = 100
            for i in range(0, len(all_chunks), batch_size):
                end = min(i + batch_size, len(all_chunks))
                chromadb_collection.add(
                    ids=all_ids[i:end],
                    documents=all_chunks[i:end],
                    embeddings=embeddings[i:end],
                    metadatas=all_metadatas[i:end]
                )
            print(f"向量库初始化完成，共索引 {doc_count} 个文件。")
        else:
            print("未找到有效文档。")

    @staticmethod
    def split_into_chunks(doc_file: str) -> List[str]:
        """
        读取文件并按三换行符拆分
        """
        with open(doc_file, 'r', encoding='utf-8') as file:
            content = file.read()
        # 简单清洗并拆分
        return [chunk.strip() for chunk in content.split('\n\n\n') if chunk.strip()]

    @staticmethod
    def retrieve(query: str, top_k: int = 10) -> List[str]:
        """
        根据 Query 召回相关文档
        """
        query_embedding = embedding_model.encode(query).tolist()
        results = chromadb_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        # results['documents'] 是一个列表的列表，取第一个即可
        if results['documents']:
            return results['documents'][0]
        return []

    @staticmethod
    def rerank(query: str, retrieved_chunks: List[str], top_k: int) -> List[str]:
        """
        使用 CrossEncoder 重排文档
        """
        if not retrieved_chunks:
            return []
            
        pairs = [[query, chunk] for chunk in retrieved_chunks]
        scores = rerank_model.predict(pairs)

        chunk_with_scores_list = [(chunk, score) for chunk, score in zip(retrieved_chunks, scores)]
        # 按分数降序排列
        chunk_with_scores_list.sort(key=lambda x: x[1], reverse=True)

        return [chunk for chunk, _ in chunk_with_scores_list[:top_k]]