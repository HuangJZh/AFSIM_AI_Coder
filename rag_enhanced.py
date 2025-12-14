import os
import time
import logging
import re
import json
from typing import List, Dict, Any
from functools import lru_cache

import torch
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM
)

from project_learner import AFSIMProjectLearner
from utils import ConfigManager, FileReader, setup_logging


class EnhancedRAGChatSystem:
    def __init__(self, 
                 project_root: str,
                 model_path: str = None,
                 documents_path: str = None,
                 embedding_model: str = None,
                 vector_db_dir: str = None):
        
        # 设置日志
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 加载配置
        self.config = ConfigManager()
        self.project_root = project_root
        self.documents_path = documents_path or project_root
        self.vector_db_dir = vector_db_dir or self.config.get('vector_db.persist_dir')
        self.model_path = model_path or self.config.get('model.path')
        self.embedding_model = embedding_model or self.config.get('embedding.model')
        
        self.logger.info("正在初始化AFSIM项目学习器...")
        self.project_learner = AFSIMProjectLearner(project_root)
        self.project_learner.analyze_project_structure()
        
        self.logger.info("正在加载嵌入模型...")
        self.embeddings = self._setup_embeddings()
        
        self.logger.info("正在加载大语言模型...")
        self.tokenizer, self.model = self._setup_llm()
        
        # 构建或加载向量数据库
        self.vector_db = self.build_or_load_vector_db()
        self.enhanced_qa_chain = self.create_enhanced_qa_chain()
        
        self.conversation_history = []
        
        self.logger.info("EnhancedRAGChatSystem 初始化完成")
    
    def _setup_embeddings(self):
        """设置嵌入模型"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"使用设备: {device}")
        
        try:
            return HuggingFaceBgeEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True},
                query_instruction="为这个句子生成表示以用于检索相关文章："
            )
        except Exception as e:
            self.logger.error(f"加载嵌入模型失败，回退到CPU: {e}")
            return HuggingFaceBgeEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True},
                query_instruction="为这个句子生成表示以用于检索相关文章："
            )
    
    def _setup_llm(self):
        """设置语言模型"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True,
                padding_side='left'
            )
            
            # === 关键修改：设置截断方向为左侧，保留Prompt末尾的指令 ===
            tokenizer.truncation_side = 'left' 
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 根据可用设备选择加载方式
            if torch.cuda.is_available():
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    dtype=torch.float16,
                    device_map="cuda",
                    trust_remote_code=True,
                ).eval()
            
            return tokenizer, model
                
        except Exception as e:
            self.logger.error(f"加载语言模型失败: {e}")
            raise
    
    @lru_cache(maxsize=100)
    def search_similar_documents_cached(self, query: str, top_k: int = None):
        """带缓存的文档搜索"""
        if top_k is None:
            top_k = self.config.get('vector_db.search_k', 6)
        return self.vector_db.similarity_search(query, k=top_k)
    
    def build_or_load_vector_db(self):
        """构建或加载向量数据库"""
        if os.path.exists(self.vector_db_dir):
            self.logger.info("加载已有向量数据库...")
            try:
                vector_db = Chroma(
                    persist_directory=self.vector_db_dir,
                    embedding_function=self.embeddings
                )
                # 验证数据库是否有效
                if hasattr(vector_db, '_collection') and vector_db._collection.count() > 0:
                    self.logger.info("向量数据库加载成功")
                    return vector_db
                else:
                    self.logger.warning("向量数据库为空，将重新构建")
            except Exception as e:
                self.logger.error(f"加载向量数据库失败，将重新构建: {e}")
        
        self.logger.info("构建新的向量数据库...")
        return self.build_knowledge_base()
    
    def build_knowledge_base(self):
        """构建知识库"""
        self.logger.info("开始处理文档构建知识库...")
        start_time = time.time()

        try:
            # 收集所有可读的文件
            all_txt_files = []
            for root, dirs, files in os.walk(self.documents_path):
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        if not FileReader.should_skip_file(file_path):
                            all_txt_files.append(file_path)
            
            self.logger.info(f"找到 {len(all_txt_files)} 个文本文件，开始加载...")
            
            documents = []
            skipped_files = 0
            
            # 使用进度条
            with tqdm(total=len(all_txt_files), desc="加载文件") as pbar:
                for file_path in all_txt_files:
                    try:
                        content, encoding = FileReader.read_file_safely(file_path)
                        if content and content.strip():
                            from langchain_core.documents import Document
                            doc = Document(
                                page_content=content,
                                metadata={"source": file_path, "encoding": encoding}
                            )
                            documents.append(doc)
                        else:
                            skipped_files += 1
                    except Exception as e:
                        self.logger.warning(f"跳过无法读取的文件: {file_path} - {e}")
                        skipped_files += 1
                    finally:
                        pbar.update(1)
                        pbar.set_postfix({"当前文件": os.path.basename(file_path)})
            
            if not documents:
                raise ValueError(f"在 {self.documents_path} 目录中未找到可读文档")
            
            self.logger.info(f"成功加载 {len(documents)} 个有效文档 (跳过了 {skipped_files} 个文件)")

            # 分割文档
            chunk_size = self.config.get('vector_db.chunk_size')
            chunk_overlap = self.config.get('vector_db.chunk_overlap')
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", "。", "；", " ", ""]
            )
            
            with tqdm(total=len(documents), desc="分割文档") as pbar:
                texts = []
                for doc in documents:
                    chunks = text_splitter.split_documents([doc])
                    texts.extend(chunks)
                    pbar.update(1)
            
            self.logger.info(f"文档分割完成，得到 {len(texts)} 个文本块")

            # 分批添加到向量数据库
            self.logger.info("分批创建向量数据库...")
            vector_db = self._create_vector_db_in_batches(texts)
            
            end_time = time.time()
            self.logger.info(f"知识库构建完成，耗时 {(end_time - start_time)/60:.2f} 分钟")
            return vector_db
            
        except Exception as e:
            self.logger.error(f"构建知识库失败: {e}")
            raise

    def _create_vector_db_in_batches(self, texts, batch_size=4000):
        """分批创建向量数据库以避免批量大小限制"""
        from langchain_community.vectorstores import Chroma
        
        self.logger.info(f"开始分批处理 {len(texts)} 个文档块，批次大小: {batch_size}")
        
        # 第一次批次创建数据库
        first_batch = texts[:batch_size]
        self.logger.info(f"创建初始向量数据库，包含 {len(first_batch)} 个文档...")
        
        vector_db = Chroma.from_documents(
            documents=first_batch,
            embedding=self.embeddings,
            persist_directory=self.vector_db_dir
        )
        
        # 剩余批次逐步添加
        remaining_texts = texts[batch_size:]
        if remaining_texts:
            self.logger.info(f"逐步添加剩余 {len(remaining_texts)} 个文档...")
            
            for i in range(0, len(remaining_texts), batch_size):
                batch = remaining_texts[i:i + batch_size]
                self.logger.info(f"添加批次 {i//batch_size + 1}/{(len(remaining_texts)-1)//batch_size + 1}，包含 {len(batch)} 个文档")
                
                vector_db.add_documents(batch)
                
                # 及时清理内存
                del batch
        
        # 持久化最终数据库
        vector_db.persist()
        self.logger.info("向量数据库创建完成并已持久化")
        
        return vector_db
    
    def create_enhanced_qa_chain(self):
        """创建增强的QA链"""
        
        class CustomQwenLLM:
            def __init__(self, model, tokenizer, project_learner):
                self.model = model
                self.tokenizer = tokenizer
                self.project_learner = project_learner
                self.config = ConfigManager()
                self.logger = logging.getLogger(__name__)
            
            def __call__(self, prompt):
                try:
                    # 使用配置中的最大长度，或者默认为4096 (保留更多上下文)
                    max_len = 32000 
                    
                    inputs = self.tokenizer(
                        prompt, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=max_len,
                        padding=True
                    )
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=self.config.get('model.max_tokens'),
                            do_sample=True,
                            temperature=self.config.get('model.temperature'),
                            top_p=self.config.get('model.top_p'),
                            pad_token_id=self.tokenizer.eos_token_id,
                            repetition_penalty=self.config.get('model.repetition_penalty'),
                            use_cache=True
                        )
                    
                    response = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:], 
                        skip_special_tokens=True
                    ).strip()
                    
                    return response
                    
                except Exception as e:
                    self.logger.error(f"生成回答时出错: {e}")
                    return f"生成回答时出错: {str(e)}"
        
        # 创建检索器
        search_k = self.config.get('vector_db.search_k')
        retriever = self.vector_db.as_retriever(search_kwargs={"k": search_k})
        
        class EnhancedCodeGenerationChain:
            def __init__(self, llm, retriever, project_learner):
                self.llm = llm
                self.retriever = retriever
                self.project_learner = project_learner
                self.logger = logging.getLogger(__name__)
            
            def run(self, query):
                try:
                    # 检索相关文档
                    docs = self.retriever.get_relevant_documents(query)
                    
                    # 获取项目上下文
                    project_context = self.project_learner.generate_context_prompt(query)
                    
                    # 构建增强的提示词
                    context = self._build_context_prompt(project_context, docs)
                    prompt = self._build_generation_prompt(context, query)
                    
                    # 生成回答
                    response = self.llm(prompt)
                    
                    return {
                        "result": response,
                        "source_documents": docs,
                        "project_context": project_context
                    }
                    
                except Exception as e:
                    self.logger.error(f"运行增强代码生成链时出错: {e}")
                    return {
                        "result": f"生成代码时出错: {str(e)}",
                        "source_documents": [],
                        "project_context": ""
                    }
            
            def _build_context_prompt(self, project_context, docs):
                """构建上下文提示词"""
                context = " AFSIM项目上下文信息:\n\n"
                context += project_context
                context += "\n\n 相关代码示例:\n\n"
                
                for i, doc in enumerate(docs, 1):
                    source_info = f"来源: {doc.metadata.get('source', '未知')}" if hasattr(doc, 'metadata') else ""
                    context += f"示例 {i} {source_info}:\n{doc.page_content[:500]}...\n{'='*50}\n"
                
                return context
            
            def _clean_query(self, query: str) -> str:
                """清理查询中的重复内容"""
                lines = query.split('\n')
                cleaned_lines = []
                required_keywords_seen = set()
                
                for line in lines:
                    line_clean = line.strip()
                    if not line_clean:
                        continue
                        
                    # 检测重复的"必须包含"模式
                    if "必须包含" in line_clean:
                        key_content = re.sub(r'.*必须包含', '', line_clean).strip()
                        if key_content and key_content not in required_keywords_seen:
                            cleaned_lines.append(line_clean)
                            required_keywords_seen.add(key_content)
                    else:
                        cleaned_lines.append(line_clean)
                
                # 如果清理后内容太少，返回原始查询的重要部分
                if len(cleaned_lines) < 2:
                    important_lines = []
                    for line in lines[:10]:
                        line_clean = line.strip()
                        if line_clean and "必须包含" not in line_clean:
                            important_lines.append(line_clean)
                    return "\n".join(important_lines[:5])
                
                return "\n".join(cleaned_lines[:20])  # 限制长度
            
            def _build_generation_prompt(self, context, query):
                """构建生成提示词"""
                cleaned_query = self._clean_query(query)

                return f"""你是一个AFSIM代码生成专家，熟悉整个项目结构和基础库的使用。
{context}
 用户需求: {query}
请基于以上项目结构和相关代码示例，直接生成准确、完整的AFSIM代码。
 禁止:
不要添加解释性文字
不要重复相同的内容
不要输出不完整的代码块
请生成完整的AFSIM代码:"""
        
        return EnhancedCodeGenerationChain(
            llm=CustomQwenLLM(self.model, self.tokenizer, self.project_learner),
            retriever=retriever,
            project_learner=self.project_learner
        )
    
    # === 新增代码修复方法 (增强版) ===
    def generate_code_repair_response(self, code: str, error_log: str) -> Dict[str, Any]:
        """生成代码修复建议"""
        self.logger.info("正在执行代码修复...")
        try:
            # 1. 提取错误关键信息进行RAG检索
            search_query = f"AFSIM code error: {error_log[:200]}"
            
            # 使用检索器查找相关文档
            docs = self.enhanced_qa_chain.retriever.get_relevant_documents(search_query)
            
            # 2. 构建修复提示词
            context_docs = ""
            for i, doc in enumerate(docs[:3], 1):
                context_docs += f"参考文档 {i}:\n{doc.page_content[:400]}...\n\n"

            # 转义大括号，防止 f-string 错误
            safe_code = code.replace("{", "{{").replace("}", "}}")
            safe_error = error_log.replace("{", "{{").replace("}", "}}")

            prompt = f"""
你是一个AFSIM仿真脚本代码修复专家。请根据以下错误信息修复代码。

=== 📚 参考语法文档 ===
{context_docs}

=== ❌ 错误代码 ===
{safe_code}

=== ⚠️ 报错信息 ===
{safe_error}

=== 🛠️ 修复任务 ===
1. 分析报错原因。
2. 根据参考文档修正代码中的语法错误或逻辑错误。
3. **强制使用标记包裹**：将修复后的完整代码包裹在 `<<<CODE_START>>>` 和 `<<<CODE_END>>>` 之间。
4. **严禁输出解释**：只输出修复后的代码，不要输出“我已修复”、“原因如下”等废话。

请立即输出修复后的代码：
"""
            # 3. 调用LLM
            response = self.enhanced_qa_chain.llm(prompt)
            
            # 4. 提取代码 (增强鲁棒性)
            code_match = re.search(r'<<<CODE_START>>>\s*(.*?)\s*<<<CODE_END>>>', response, re.DOTALL)
            if code_match:
                final_result = code_match.group(1)
            else:
                # 兜底1：尝试提取markdown块
                code_block = re.search(r'```(?:afsim|txt|)\s*(.*?)\s*```', response, re.DOTALL)
                if code_block:
                    final_result = code_block.group(1)
                else:
                    # 兜底2：如果提取不到，返回原始Response，并打印警告
                    self.logger.warning("未找到代码标记或Markdown块，返回原始回答。")
                    final_result = response
            
            return {
                "result": final_result.strip(),
                "source_documents": docs
            }

        except Exception as e:
            self.logger.error(f"代码修复失败: {e}")
            return {
                "result": f"修复失败: {str(e)}",
                "source_documents": []
            }
            
    def generate_enhanced_response(self, query: str) -> Dict[str, Any]:
        """生成增强的回答"""
        try:
            # 使用增强的QA链生成回答
            result = self.enhanced_qa_chain.run(query)
            
            # 更新对话历史
            self.conversation_history.append({
                'query': query,
                'response': result["result"],
                'sources': len(result["source_documents"]),
                'timestamp': time.time()
            })
            
            # 限制历史记录长度
            if len(self.conversation_history) > 8:
                self.conversation_history = self.conversation_history[-6:]
            
            return result
            
        except Exception as e:
            self.logger.error(f"生成增强响应时出错: {e}")
            return {
                "result": f"生成回答时出错: {str(e)}", 
                "source_documents": [],
                "project_context": ""
            }
    
    def get_vector_db_info(self):
        """获取向量数据库信息"""
        if hasattr(self.vector_db, '_collection'):
            count = self.vector_db._collection.count()
            return f"向量数据库包含 {count} 个文档片段"
        return "向量数据库信息不可用"
    
    def get_project_info(self):
        """获取项目信息"""
        return self.project_learner.get_project_summary()
    
    def search_project_files(self, keyword: str, max_results: int = 5):
        """在项目中搜索文件"""
        return self.project_learner.find_related_files(keyword, max_results)


class StageAwareRAGSystem:
    # ... (保持原来的 StageAwareRAGSystem 类内容不变)
    """简化的阶段感知RAG系统"""
    
    def __init__(self, project_learner: AFSIMProjectLearner, vector_db, embeddings, model, tokenizer):
        self.project_learner = project_learner
        self.vector_db = vector_db
        self.embeddings = embeddings
        self.model = model
        self.tokenizer = tokenizer
        self.config = ConfigManager()
        self.logger = logging.getLogger(__name__)
        
        # 阶段检索器缓存
        self.stage_retrievers = {}
    
    def get_stage_retriever(self, stage_name: str):
        """获取阶段感知检索器"""
        if stage_name in self.stage_retrievers:
            return self.stage_retrievers[stage_name]
        
        # 获取阶段特定的检索参数
        search_k = self.config.get(f'rag.stages.{stage_name}.search_k', 8)
        
        # 创建基础检索器
        retriever = self.vector_db.as_retriever(
            search_kwargs={"k": search_k}
        )
        
        # 创建阶段感知的检索器包装器
        stage_retriever = self._create_stage_retriever_wrapper(retriever, stage_name)
        self.stage_retrievers[stage_name] = stage_retriever
        
        return stage_retriever
    
    def _create_stage_retriever_wrapper(self, base_retriever, stage_name: str):
        """创建阶段感知检索器包装器"""
        
        def get_relevant_documents(query: str):
            """增强查询并过滤文档"""
            try:
                # 增强查询
                enhanced_query = self._enhance_query_for_stage(query, stage_name)
                
                # 执行基础检索
                docs = base_retriever.get_relevant_documents(enhanced_query)
                
                # 过滤文档
                filtered_docs = self._filter_docs_by_stage(docs, stage_name)
                
                return filtered_docs[:8]  # 限制返回数量
                
            except Exception as e:
                self.logger.error(f"检索文档失败: {e}")
                return []
        
        return get_relevant_documents
    
    def _enhance_query_for_stage(self, query: str, stage_name: str) -> str:
        """为特定阶段增强查询"""
        stage_keywords = {
            "platforms": ["platform", "aircraft", "vehicle"],
            "weapons": ["weapon", "missile", "launch"],
            "sensors": ["sensor", "radar", "detect"],
            "processors": ["processor", "algorithm", "control"],
            "scenarios": ["scenario", "mission", "environment"],
            "signatures": ["signature", "rcs", "emission"],
            "main_program": ["main", "include", "initialize"],
            "project_structure": ["project", "structure", "folder"]
        }
        
        keywords = stage_keywords.get(stage_name, [])
        enhanced = query
        
        if keywords:
            enhanced += " " + " ".join(keywords[:2])
        
        return enhanced
    
    def _filter_docs_by_stage(self, docs, stage_name: str):
        """根据阶段过滤文档"""
        if not docs:
            return []
        
        stage_keywords_map = {
            "platforms": ["platform_type", "mover"],
            "weapons": ["weapon_type", "missile"],
            "sensors": ["sensor_type", "radar"],
            "processors": ["processor_type", "tasker"],
            "scenarios": ["scenario", "mission"],
            "signatures": ["signature", "rcs"],
            "main_program": ["main", "include"],
        }
        
        keywords = stage_keywords_map.get(stage_name, [])
        
        if not keywords:
            return docs
        
        filtered = []
        for doc in docs:
            doc_text = doc.page_content.lower()
            if any(keyword in doc_text for keyword in keywords):
                filtered.append(doc)
        
        # 如果过滤后文档太少，返回原始文档
        return filtered[:5] if len(filtered) >= 3 else docs[:5]


class EnhancedStageAwareRAGChatSystem(EnhancedRAGChatSystem):
    # ... (保持原来的 EnhancedStageAwareRAGChatSystem 类内容不变)
    """增强的阶段感知RAG聊天系统"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 初始化阶段感知系统
        self.stage_aware_system = StageAwareRAGSystem(
            project_learner=self.project_learner,
            vector_db=self.vector_db,
            embeddings=self.embeddings,
            model=self.model,
            tokenizer=self.tokenizer
        )
        
        # 阶段特定的QA链缓存
        self.stage_qa_chains = {}
    
    def get_stage_aware_qa_chain(self, stage_name: str):
        """获取阶段感知的QA链"""
        if stage_name in self.stage_qa_chains:
            return self.stage_qa_chains[stage_name]
        
        self.logger.info(f"为阶段 {stage_name} 构建QA链...")
        
        # 获取阶段特定的检索器
        retriever = self.stage_aware_system.get_stage_retriever(stage_name)
        
        # 创建阶段特定的LLM
        class StageAwareLLM:
            def __init__(self, model, tokenizer, stage_name):
                self.model = model
                self.tokenizer = tokenizer
                self.stage_name = stage_name
                self.config = ConfigManager()
                self.logger = logging.getLogger(__name__)
            
            def __call__(self, prompt):
                try:
                    # 阶段特定的生成参数
                    stage_params = self.config.get(f'rag.stages.{self.stage_name}', {})
                    max_tokens = stage_params.get('max_tokens', self.config.get('model.max_tokens', 1024))
                    temperature = stage_params.get('temperature', self.config.get('model.temperature', 0.2))
                    
                    inputs = self.tokenizer(
                        prompt, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=32000,
                        padding=True
                    )
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            temperature=temperature,
                            top_p=self.config.get('model.top_p', 0.9),
                            pad_token_id=self.tokenizer.eos_token_id,
                            repetition_penalty=self.config.get('model.repetition_penalty', 1.1)
                        )
                    
                    response = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:], 
                        skip_special_tokens=True
                    ).strip()
                    
                    return response
                    
                except Exception as e:
                    self.logger.error(f"阶段 {self.stage_name} 生成回答时出错: {e}")
                    return f"生成回答时出错: {str(e)}"
        
        # 创建阶段感知的QA链
        class StageAwareQAChain:
            def __init__(self, llm, retriever, project_learner, stage_name):
                self.llm = llm
                self.retriever = retriever
                self.project_learner = project_learner
                self.stage_name = stage_name
                self.logger = logging.getLogger(__name__)
            
            def run(self, query: str, project_context: Dict = None) -> Dict[str, Any]:
                try:
                    # 获取阶段特定的学习结果
                    stage_context = self.project_learner.generate_context_prompt(query)
                    
                    # 检索相关文档
                    docs = self.retriever(query)
                    
                    # 构建增强提示词
                    prompt = self._build_stage_prompt(query, stage_context, docs, project_context)
                    
                    # 生成回答
                    response = self.llm(prompt)
                    
                    return {
                        "result": response,
                        "stage_name": self.stage_name,
                        "source_documents": docs,
                        "stage_context": stage_context
                    }
                    
                except Exception as e:
                    self.logger.error(f"阶段 {self.stage_name} QA链运行失败: {e}")
                    return {
                        "result": f"生成失败: {str(e)}",
                        "stage_name": self.stage_name,
                        "source_documents": [],
                        "stage_context": ""
                    }
            
            def _build_stage_prompt(self, query: str, stage_context: str, docs: List, project_context: Dict = None) -> str:
                """构建阶段特定的提示词"""
                # 文档内容
                doc_content = ""
                for i, doc in enumerate(docs[:4], 1):
                    doc_content += f"示例文档 {i}:\n{doc.page_content[:500]}...\n\n"
                
                # 项目上下文
                proj_context = ""
                if project_context:
                    proj_context = f"\n当前项目上下文:\n{json.dumps(project_context, indent=2, ensure_ascii=False)}\n"
                
                # 阶段特定的指令
                stage_instructions = {
                    "project_structure": "你需要分析AFSIM项目需求并规划项目结构。输出应该是一个清晰的JSON格式结构，包含必要的文件夹和文件规划。",
                    "platforms": "你需要生成AFSIM平台定义。确保包含完整的平台类型定义、物理参数、组件配置和行为定义。",
                    "weapons": "你需要生成AFSIM武器定义。包括武器类型、性能参数、制导系统和战斗部配置。",
                    "sensors": "你需要生成AFSIM传感器定义。包含传感器类型、探测参数、工作模式和数据输出格式。",
                    "processors": "你需要生成AFSIM处理器定义。包括处理器类型、输入输出接口、处理算法和配置参数。",
                    "scenarios": "你需要生成AFSIM场景定义。包含场景描述、平台配置、环境设置和事件序列。",
                    "signatures": "你需要生成AFSIM特征信号定义。包括特征类型、RCS值、角度依赖性和辐射特性。",
                    "main_program": "你需要生成AFSIM主程序。包含必要的导入、初始化、事件循环和输出配置。"
                }
                
                instruction = stage_instructions.get(self.stage_name, "请根据需求生成相应的AFSIM代码。")
                
                prompt = f"""{instruction}

阶段学习总结:
{stage_context}

相关代码示例:
{doc_content}
{proj_context}
用户需求: {query}

请基于以上信息生成完整的{self.stage_name}阶段代码。
输出要求:
1. 只输出AFSIM代码，不添加额外解释
2. 确保代码完整性和正确性
3. 遵循示例中的最佳实践

生成代码:"""
                
                return prompt
        
        # 创建LLM实例
        llm = StageAwareLLM(self.model, self.tokenizer, stage_name)
        
        # 创建QA链
        qa_chain = StageAwareQAChain(llm, retriever, self.project_learner, stage_name)
        
        self.stage_qa_chains[stage_name] = qa_chain
        return qa_chain
    
    def generate_stage_response(self, stage_name: str, query: str, project_context: Dict = None) -> Dict[str, Any]:
        """生成阶段特定的响应"""
        try:
            # 获取阶段感知的QA链
            qa_chain = self.get_stage_aware_qa_chain(stage_name)
            
            # 执行生成
            result = qa_chain.run(query, project_context)
            
            # 记录到历史
            self.conversation_history.append({
                'stage': stage_name,
                'query': query,
                'response': result["result"],
                'timestamp': time.time()
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"生成阶段 {stage_name} 响应失败: {e}")
            return {
                "result": f"生成失败: {str(e)}",
                "stage_name": stage_name,
                "error": str(e)
            }