import os
import torch
from typing import List, Dict, Any, Optional
import numpy as np
from chromadb import PersistentClient
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from pathlib import Path
import traceback

# 导入配置管理器
try:
    from utils import ConfigManager
except ImportError:
    # 如果无法导入，创建简单的配置管理器
    class SimpleConfigManager:
        def __init__(self):
            self.config = {}
        def get(self, key, default=None):
            return default
        def get_int(self, key, default=0):
            return default
        def get_float(self, key, default=0.0):
            return default
        def get_bool(self, key, default=False):
            return default
    
    ConfigManager = SimpleConfigManager

logger = logging.getLogger(__name__)

class AFSIMRAGSystem:
    def __init__(self, 
                 model_path: Optional[str] = None,
                 embedding_model: Optional[str] = None,
                 chroma_db_path: Optional[str] = None):
        """
        初始化AFSIM RAG系统
        """
        logger.info("正在初始化AFSIM RAG系统...")
        
        # 初始化配置管理器
        self.config = ConfigManager()
        
        # 使用配置值或参数值
        self.model_path = model_path or self.config.get('model.path')
        self.embedding_model_name = embedding_model or self.config.get('embedding.model_name')
        self.chroma_db_path = chroma_db_path or self.config.get('database.chroma_path')
        
        # 检查模型路径
        if not self.model_path or not os.path.exists(self.model_path):
            logger.warning(f"模型路径不存在: {self.model_path}")
            logger.info("将尝试从HuggingFace下载或使用默认路径")
        
        # 初始化组件
        self._init_embedding_model()
        self._init_vector_db()
        self._init_llm()
        
        logger.info("系统初始化完成！")
        
    def _init_embedding_model(self):
        """初始化嵌入模型"""
        logger.info(f"加载嵌入模型: {self.embedding_model_name}")
        try:
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            
            # 从配置获取嵌入参数
            self.normalize_embeddings = self.config.get_bool('embedding.normalize_embeddings', True)
            self.embedding_batch_size = self.config.get_int('embedding.batch_size', 32)
            
            logger.info(f"嵌入维度: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {e}")
            raise
    
    def _init_vector_db(self):
        """初始化向量数据库"""
        logger.info(f"初始化Chroma数据库: {self.chroma_db_path}")
        
        try:
            # 创建数据库目录如果不存在
            os.makedirs(self.chroma_db_path, exist_ok=True)
            
            db_settings = self.config.get('database.settings', {})
            self.client = PersistentClient(
                path=self.chroma_db_path,
                settings=Settings(
                    anonymized_telemetry=db_settings.get('anonymized_telemetry', False),
                    is_persistent=True
                )
            )
            
            # 创建或获取集合
            collection_name = self.config.get('vector_db.collection_name', 'afsim_tutorials')
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "description": "AFSIM教程文档向量存储",
                    "model": self.embedding_model_name
                }
            )
            
            doc_count = self.collection.count()
            logger.info(f"数据库文档数量: {doc_count}")
            
        except Exception as e:
            logger.error(f"向量数据库初始化失败: {e}")
            raise
    
    def _init_llm(self):
        """初始化Qwen3-4B模型"""
        logger.info(f"加载Qwen3-4B模型: {self.model_path}")
        
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left"  # 对于生成任务，padding应该在左边
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.pad_token = self.tokenizer.sep_token or "<pad>"
            
            # 从配置获取加载参数
            dtype_str = self.config.get('system.dtype', 'float16')
            load_in_4bit = self.config.get_bool('system.load_in_4bit', True)
            use_quantization = self.config.get_bool('system.use_quantization', True)
            device_map = self.config.get('system.device_map', 'auto')
            
            dtype = getattr(torch, dtype_str) if hasattr(torch, dtype_str) else torch.float16
            
            # 模型加载参数
            model_kwargs = {
                "trust_remote_code": True,
                "dtype": dtype,
                "device_map": device_map,
            }
            
            # 尝试使用量化加载
            if use_quantization and load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    model_kwargs["quantization_config"] = bnb_config
                    logger.info("使用4-bit量化配置")
                except ImportError:
                    logger.warning("未安装bitsandbytes，无法使用4-bit量化")
                    load_in_4bit = False
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            # 设置为评估模式
            self.model.eval()
            
            # 从配置获取生成参数
            generation_config = self.config.get('model.generation', {})
            self.generation_config = {
                "max_new_tokens": self.config.get_int('model.generation.max_new_tokens', 1024),
                "temperature": self.config.get_float('model.generation.temperature', 0.3),
                "top_p": self.config.get_float('model.generation.top_p', 0.9),
                "do_sample": generation_config.get('do_sample', True),
                "repetition_penalty": self.config.get_float('model.generation.repetition_penalty', 1.1),
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True
            }
            
            logger.info("模型加载成功")
            logger.info(f"生成配置: {self.generation_config}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def load_documents_from_folder(self, folder_path: Optional[str] = None) -> bool:
        """
        从文件夹加载所有.md文件到向量数据库
        """
        if folder_path is None:
            folder_path = self.config.get('paths.tutorials_folder', 'tutorials')
        
        logger.info(f"开始扫描文件夹: {folder_path}")
        
        if not os.path.exists(folder_path):
            logger.error(f"文件夹不存在: {folder_path}")
            return False
        
        if not os.path.isdir(folder_path):
            logger.error(f"路径不是文件夹: {folder_path}")
            return False
        
        try:
            # 从配置获取支持的文件扩展名
            supported_extensions = self.config.get('document.supported_extensions', ['.md', '.txt'])
            
            # 扫描所有支持的文件
            supported_files = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if any(file.endswith(ext) for ext in supported_extensions):
                        full_path = os.path.join(root, file)
                        supported_files.append(full_path)
            
            logger.info(f"找到 {len(supported_files)} 个支持的文件")
            
            if not supported_files:
                logger.warning("未找到任何支持的文件")
                return False
            
            # 清空现有集合 - 修复：不能使用空的where条件
            self._clear_collection()
            
            documents = []
            metadatas = []
            ids = []
            
            # 从配置获取分块参数
            chunk_size = self.config.get_int('vector_db.chunk_size', 1500)
            chunk_overlap = self.config.get_int('vector_db.chunk_overlap', 250)
            max_file_size = self.config.get_int('document.max_file_size_mb', 10) * 1024 * 1024
            
            # 读取每个文件
            for file_idx, file_path in enumerate(supported_files, 1):
                try:
                    # 检查文件大小
                    file_size = os.path.getsize(file_path)
                    if file_size > max_file_size:
                        logger.warning(f"文件过大跳过: {os.path.basename(file_path)} ({file_size/1024/1024:.1f}MB)")
                        continue
                    
                    # 读取文件内容
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        doc_content = f.read()
                    
                    if not doc_content.strip():
                        logger.warning(f"文件内容为空: {os.path.basename(file_path)}")
                        continue
                    
                    # 分割文档
                    chunks = self._split_into_chunks(
                        doc_content, 
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        if chunk.strip():  # 跳过空块
                            doc_id = f"{os.path.basename(file_path)}_{file_idx}_{chunk_idx}"
                            documents.append(chunk)
                            metadatas.append({
                                "source": file_path,
                                "chunk": chunk_idx,
                                "filename": os.path.basename(file_path),
                                "filepath": file_path,
                                "total_chunks": len(chunks)
                            })
                            ids.append(doc_id)
                    
                    logger.info(f"已加载: {os.path.basename(file_path)} ({len(chunks)} 个块)")
                    
                except Exception as e:
                    logger.error(f"读取文件失败 {file_path}: {e}")
                    traceback.print_exc()
            
            # 批量嵌入并存储
            if documents:
                logger.info(f"正在生成 {len(documents)} 个文档块的向量...")
                
                # 分批处理
                batch_size = self.embedding_batch_size
                total_batches = (len(documents) + batch_size - 1) // batch_size
                
                for batch_idx in range(0, len(documents), batch_size):
                    end_idx = min(batch_idx + batch_size, len(documents))
                    batch_docs = documents[batch_idx:end_idx]
                    
                    # 生成嵌入
                    embeddings = self.embedding_model.encode(
                        batch_docs,
                        normalize_embeddings=self.normalize_embeddings,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    
                    # 存储到数据库
                    self.collection.add(
                        embeddings=embeddings.tolist(),
                        documents=batch_docs,
                        metadatas=metadatas[batch_idx:end_idx],
                        ids=ids[batch_idx:end_idx]
                    )
                    
                    logger.info(f"  已处理批次 {batch_idx//batch_size + 1}/{total_batches} ({end_idx}/{len(documents)})")
                
                logger.info(f"成功加载 {len(documents)} 个文档块")
                return True
            else:
                logger.warning("未找到任何文档内容")
                return False
                
        except Exception as e:
            logger.error(f"加载文档失败: {e}")
            traceback.print_exc()
            return False
    
    def _clear_collection(self):
        """清空集合中的所有文档"""
        try:
            # 先尝试获取所有文档ID
            try:
                # 尝试获取所有文档
                results = self.collection.get()
                if results and 'ids' in results and results['ids']:
                    # 如果有文档，使用ids删除
                    self.collection.delete(ids=results['ids'])
                    logger.info(f"清空了 {len(results['ids'])} 个文档")
                else:
                    logger.info("集合为空，无需清空")
            except Exception as e:
                logger.warning(f"获取文档列表失败: {e}")
                
                # 备用方法：尝试使用where条件删除
                try:
                    # 尝试删除所有文档
                    self.collection.delete(where={"filename": {"$ne": ""}})
                    logger.info("使用where条件清空集合")
                except Exception as e2:
                    logger.warning(f"使用where条件删除失败: {e2}")
                    
                    # 最后手段：删除并重新创建集合
                    collection_name = self.collection.name
                    self.client.delete_collection(collection_name)
                    logger.info(f"删除了集合: {collection_name}")
                    
                    # 重新创建集合
                    self.collection = self.client.get_or_create_collection(
                        name=collection_name,
                        metadata={
                            "description": "AFSIM教程文档向量存储",
                            "model": self.embedding_model_name
                        }
                    )
                    logger.info(f"重新创建了集合: {collection_name}")
                    
        except Exception as e:
            logger.error(f"清空集合失败: {e}")
            traceback.print_exc()
    
    def load_documents_from_list(self, file_list_path: str, base_dir: str = ".") -> bool:
        """
        从文件列表加载文档（备用方法）
        """
        logger.info(f"从文件列表加载文档: {file_list_path}")
        
        if not os.path.exists(file_list_path):
            logger.error(f"文件不存在: {file_list_path}")
            return False
        
        try:
            with open(file_list_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 过滤空行和注释
            file_paths = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    file_paths.append(line)
            
            logger.info(f"文件列表中有 {len(file_paths)} 个文件")
            
            # 清空现有集合
            self._clear_collection()
            
            documents = []
            metadatas = []
            ids = []
            
            chunk_size = self.config.get_int('vector_db.chunk_size', 1500)
            chunk_overlap = self.config.get_int('vector_db.chunk_overlap', 250)
            
            for file_idx, line in enumerate(file_paths, 1):
                try:
                    # 清理路径
                    file_path = line.replace('D:.\\', '').replace('D:.', '').strip()
                    file_path = file_path.replace('\\', '/')
                    
                    # 添加基础目录
                    if not os.path.isabs(file_path):
                        file_path = os.path.join(base_dir, file_path)
                    
                    if os.path.exists(file_path):
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            doc_content = f.read()
                        
                        chunks = self._split_into_chunks(
                            doc_content,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        
                        for chunk_idx, chunk in enumerate(chunks):
                            if chunk.strip():
                                doc_id = f"{os.path.basename(file_path)}_{file_idx}_{chunk_idx}"
                                documents.append(chunk)
                                metadatas.append({
                                    "source": file_path,
                                    "chunk": chunk_idx,
                                    "filename": os.path.basename(file_path),
                                    "filepath": file_path,
                                    "total_chunks": len(chunks)
                                })
                                ids.append(doc_id)
                        
                        logger.info(f"已加载: {os.path.basename(file_path)} ({len(chunks)} 个块)")
                        
                    else:
                        logger.warning(f"文件不存在: {file_path}")
                        
                except Exception as e:
                    logger.error(f"读取文件失败 {line}: {e}")
                    traceback.print_exc()
            
            if documents:
                # 分批嵌入
                batch_size = self.embedding_batch_size
                
                for i in range(0, len(documents), batch_size):
                    end_idx = min(i + batch_size, len(documents))
                    batch_docs = documents[i:end_idx]
                    
                    embeddings = self.embedding_model.encode(
                        batch_docs,
                        normalize_embeddings=self.normalize_embeddings,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    
                    self.collection.add(
                        embeddings=embeddings.tolist(),
                        documents=batch_docs,
                        metadatas=metadatas[i:end_idx],
                        ids=ids[i:end_idx]
                    )
                
                logger.info(f"成功加载 {len(documents)} 个文档块")
                return True
            else:
                logger.warning("未找到任何文档内容")
                return False
                
        except Exception as e:
            logger.error(f"加载文档失败: {e}")
            traceback.print_exc()
            return False
    
    def _split_into_chunks(self, text: str, chunk_size: int = 1500, chunk_overlap: int = 250) -> List[str]:
        """将文本分割成重叠的块"""
        if not text.strip():
            return []
        
        # 按段落分割
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            # 如果段落本身超过chunk_size，需要分割段落
            if para_length > chunk_size:
                # 先添加当前块
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # 分割大段落
                words = para.split()
                temp_chunk = []
                temp_length = 0
                
                for word in words:
                    word_length = len(word) + 1  # 加1是为了空格
                    if temp_length + word_length <= chunk_size:
                        temp_chunk.append(word)
                        temp_length += word_length
                    else:
                        if temp_chunk:
                            chunks.append(' '.join(temp_chunk))
                        
                        # 设置重叠
                        overlap_words = temp_chunk[-chunk_overlap//5:] if chunk_overlap > 0 else []
                        temp_chunk = overlap_words + [word]
                        temp_length = sum(len(w) + 1 for w in temp_chunk)
                
                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))
            
            # 正常情况：段落适合当前块
            elif current_length + para_length <= chunk_size:
                current_chunk.append(para)
                current_length += para_length + 2  # 加2是为了\n\n
            else:
                # 当前块已满，保存它
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                
                # 创建新块，考虑重叠
                if chunk_overlap > 0 and current_chunk:
                    # 计算重叠内容
                    overlap_text = '\n\n'.join(current_chunk)
                    overlap_words = overlap_text.split()
                    overlap_size = min(len(overlap_words), chunk_overlap // 5)
                    overlap_content = ' '.join(overlap_words[-overlap_size:]) if overlap_size > 0 else ""
                    
                    current_chunk = [overlap_content, para] if overlap_content else [para]
                    current_length = len('\n\n'.join(current_chunk))
                else:
                    current_chunk = [para]
                    current_length = para_length
        
        # 添加最后一个块
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def retrieve_relevant_docs(self, query: str, n_results: int = 5) -> List[Dict]:
        """检索相关文档"""
        doc_count = self.collection.count()
        if doc_count == 0:
            logger.warning("向量数据库为空，请先加载文档")
            return []
        
        try:
            # 生成查询向量
            query_embedding = self.embedding_model.encode(
                query,
                normalize_embeddings=True,
                show_progress_bar=False
            ).tolist()
            
            # 检索，增加结果数以提高召回率
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results * 2, doc_count),
                include=["documents", "metadatas", "distances"]
            )
            
            # 格式化结果
            retrieved_docs = []
            if results['documents'] and len(results['documents'][0]) > 0:
                for i, doc in enumerate(results['documents'][0]):
                    retrieved_docs.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else None,
                        'score': 1.0 - (results['distances'][0][i] if results['distances'] else 1.0)
                    })
            
            # 按相关性排序
            retrieved_docs.sort(key=lambda x: x['score'], reverse=True)
            
            # 只返回前n_results个
            return retrieved_docs[:n_results]
            
        except Exception as e:
            logger.error(f"检索失败: {e}")
            traceback.print_exc()
            return []
    
    def format_prompt(self, query: str, retrieved_docs: List[Dict]) -> str:
        """格式化提示词"""
        system_prompt = """你是AFSIM专家助手，专门帮助用户编写AFSIM仿真代码。
你的任务是基于提供的教程内容和上下文，**只要生成准确、完整、可运行的AFSIM代码**。

请遵循以下规则：
1. 只输出有效的AFSIM代码
2. 保持代码简洁高效

AFSIM代码："""

        if not retrieved_docs:
            context = "没有找到相关的教程内容，请基于你的知识回答。"
        else:
            context = "以下是相关的AFSIM教程内容（按相关性排序）：\n\n"
            for i, doc in enumerate(retrieved_docs, 1):
                filename = doc['metadata'].get('filename', '未知文件')
                relevance = f"相关性: {doc['score']:.2%}" if doc.get('score') else ""
                context += f"【文档{i}】{filename} {relevance}\n"
                context += f"{doc['content']}\n\n"
        
        user_query = f"用户问题：{query}"
        
        prompt = f"{system_prompt}\n\n{context}\n{user_query}\n\n请生成AFSIM代码："
        
        return prompt
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """生成回答"""
        logger.info(f"处理查询: {query[:100]}...")
        
        # 检索相关文档
        retrieved_docs = self.retrieve_relevant_docs(query, n_results=4)
        
        if retrieved_docs:
            logger.info(f"检索到 {len(retrieved_docs)} 个相关文档")
            for doc in retrieved_docs[:2]:
                logger.debug(f"文档: {doc['metadata'].get('filename', '未知')}, 分数: {doc.get('score', 0):.3f}")
        
        # 构建提示
        prompt = self.format_prompt(query, retrieved_docs)
        
        try:
            # 计算最大输入长度
            max_tokens = self.config.get_int('model.max_tokens', 4096)
            max_new_tokens = self.generation_config.get('max_new_tokens', 1024)
            max_input_tokens = max_tokens - max_new_tokens - 100  # 留出缓冲
            
            # 编码输入
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
                padding=True
            )
            
            # 移动到模型所在的设备
            device = self.model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 生成回答
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.generation_config
                )
            
            # 解码输出
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],  # 只取新生成的部分
                skip_special_tokens=True
            )
            
            # 清理响应
            response = self._clean_response(response)
            
            # 提取来源信息
            sources = []
            for doc in retrieved_docs:
                filename = doc['metadata'].get('filename')
                if filename and filename not in sources:
                    sources.append(filename)
            
            logger.info(f"回答生成完成，长度: {len(response)} 字符，来源: {len(sources)} 个文件")
            
            return {
                "response": response,
                "sources": sources,
                "raw_docs": retrieved_docs[:3]  # 只保留前3个原始文档
            }
            
        except Exception as e:
            logger.error(f"生成失败: {e}")
            traceback.print_exc()
            return {
                "response": f"生成回答时出错: {str(e)}\n请检查模型配置或尝试重新初始化系统。",
                "sources": [],
                "raw_docs": []
            }
    
    def _clean_response(self, text: str) -> str:
        """清理响应文本"""
        if not text:
            return "抱歉，我没有生成任何内容。请尝试重新提问或检查系统配置。"
        
        # 移除多余的空行
        lines = text.strip().split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.rstrip()
            if line:
                cleaned_lines.append(line)
        
        # 重新组合，确保代码格式
        cleaned_text = '\n'.join(cleaned_lines)
        
        # 如果以代码常见标点结尾，确保有换行
        code_endings = ['}', ';', ']', ')']
        if cleaned_text and cleaned_text[-1] in code_endings:
            cleaned_text += '\n'
        
        # 限制最大长度
        max_length = self.generation_config.get('max_new_tokens', 1024) * 4  # 粗略估计
        if len(cleaned_text) > max_length:
            cleaned_text = cleaned_text[:max_length] + "\n\n...(回答过长，已截断)"
        
        return cleaned_text
    
    def interactive_chat(self):
        """交互式聊天"""
        print("\n" + "="*60)
        print("AFSIM RAG 系统 - 交互模式")
        print("="*60)
        print("命令:")
        print("  'exit' 或 'quit' - 退出")
        print("  'clear' - 清空上下文")
        print("  'sources' - 显示当前来源")
        print("  'reload' - 重新加载文档")
        print("  'stats' - 显示系统统计")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n👤 用户: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("再见！")
                    break
                elif user_input.lower() == 'clear':
                    print("上下文已清空")
                    continue
                elif user_input.lower() == 'sources':
                    doc_count = self.collection.count()
                    print(f"数据库中有 {doc_count} 个文档块")
                    continue
                elif user_input.lower() == 'reload':
                    folder = self.config.get('paths.tutorials_folder', 'tutorials')
                    print(f"重新加载文档从: {folder}")
                    self.load_documents_from_folder(folder)
                    continue
                elif user_input.lower() == 'stats':
                    doc_count = self.collection.count()
                    print(f"文档块数量: {doc_count}")
                    print(f"嵌入模型: {self.embedding_model_name}")
                    print(f"LLM模型: {os.path.basename(self.model_path)}")
                    continue
                elif not user_input:
                    continue
                
                # 生成回答
                result = self.generate_response(user_input)
                
                print(f"\n🤖 AFSIM助手:")
                print("-"*60)
                print(result["response"])
                print("-"*60)
                if result["sources"]:
                    print("参考来源:")
                    for source in result["sources"][:5]:  # 最多显示5个来源
                        print(f"  • {source}")
                print("="*60)
                
            except KeyboardInterrupt:
                print("\n程序已中断")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
                traceback.print_exc()