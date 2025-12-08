import os
import glob
import re
import logging
from typing import Dict, List, Set, Tuple
from pathlib import Path
import json
from utils import FileReader, ConfigManager

class AFSIMProjectLearner:
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.config = ConfigManager()
        self.base_libraries = self.config.get('project.base_libraries', ["base_types", "base_types_nx"])
        self.code_folders = []
        self.all_files = {}
        self.import_dependencies = {}
        self.file_categories = {}
        self.logger = logging.getLogger(__name__)

    def analyze_project_structure(self):
        """分析项目结构"""
        self.logger.info("开始分析AFSIM项目结构...")

        if not os.path.exists(self.project_root):
            raise ValueError(f"项目目录不存在: {self.project_root}")

        # 获取所有文件夹
        all_folders = [f.name for f in os.scandir(self.project_root) if f.is_dir()]

        # 分离基础库和代码文件夹
        self.code_folders = [f for f in all_folders if f not in self.base_libraries]

        self.logger.info(f"发现基础库: {self.base_libraries}")
        self.logger.info(f"发现代码文件夹: {len(self.code_folders)} 个")

        # 收集所有文件
        self._collect_all_files()

        # 分析文件分类
        self._categorize_files()

        # 分析导入依赖
        self._analyze_imports()

        self.logger.info(f"项目分析完成: 总共 {len(self.all_files)} 个文件")

    def _collect_all_files(self):
        """收集所有文件内容（支持 .txt 代码和 .md 教程）"""
        all_folders = self.base_libraries + self.code_folders
        skipped_files = 0
        processed_files = 0

        # === 1. 原有逻辑：遍历特定文件夹下的 .txt 代码文件 ===
        for folder in all_folders:
            folder_path = os.path.join(self.project_root, folder)
            if not os.path.exists(folder_path):
                self.logger.warning(f"文件夹不存在: {folder_path}")
                continue

            # 查找所有.txt文件
            txt_files = glob.glob(os.path.join(folder_path, "**", "*.txt"), recursive=True)
            self._process_file_list(txt_files, folder, processed_files, skipped_files)

        # === 2. 新增逻辑：遍历整个项目下的 .md 教程文件 ===
        # 注意：这里假设md文件可能在根目录或任何子目录
        md_files = glob.glob(os.path.join(self.project_root, "**", "*.md"), recursive=True)
        if md_files:
            self.logger.info(f"发现 {len(md_files)} 个Markdown教程文件")
            # 将 md 文件视为 "documentation" 文件夹类别，或者保持其所属文件夹
            self._process_file_list(md_files, "documentation", processed_files, skipped_files)

    def _process_file_list(self, file_list, folder_category, processed_count, skipped_count):
        """辅助函数：处理文件列表读取"""
        # 注意：由于整数是不可变类型，这里只是简单处理逻辑，实际计数在self.all_files长度中体现即可
        for file_path in file_list:
            if FileReader.should_skip_file(file_path):
                continue

            relative_path = os.path.relpath(file_path, self.project_root)
            
            # 避免重复处理（如果.md文件刚好在代码文件夹里）
            if relative_path in self.all_files:
                continue

            try:
                content, encoding = FileReader.read_file_safely(file_path)
                if content.strip():  # 只保存非空文件
                    self.all_files[relative_path] = {
                        'content': content,
                        'size': len(content),
                        'folder': folder_category, # 对于MD文件，这里被标记为documentation或所属文件夹
                        'encoding': encoding,
                        'extension': os.path.splitext(file_path)[1].lower() # 记录扩展名
                    }
                else:
                    self.logger.debug(f"跳过空文件: {file_path}")
            except Exception as e:
                self.logger.error(f"处理文件 {file_path} 时出错: {e}")

    def _categorize_files(self):
        """对文件进行分类 - 包含教程分类"""
        categories = {
            'platforms': [],
            'weapons': [],
            'sensors': [],
            'processors': [],
            'scenarios': [],
            'behaviors': [],
            'signatures': [],
            'scripts': [],
            'tutorials': [], # === 新增：教程分类 ===
            'other': []
        }

        for file_path, file_info in self.all_files.items():
            # content = file_info['content'].lower() # 暂时没用到内容分类，节省性能
            
            # 首先基于路径进行分类
            self._categorize_by_path(file_path, categories)

        self.file_categories = categories

        # 记录分类统计
        self.logger.info("文件分类统计:")
        for category, files in categories.items():
            self.logger.info(f"  {category}: {len(files)} 个文件")

    def _categorize_by_path(self, file_path: str, categories: dict) -> bool:
        """基于文件路径进行分类"""
        file_path_lower = file_path.lower()

        # === 新增：优先检查 Markdown 文件 ===
        if file_path_lower.endswith('.md'):
            categories['tutorials'].append(file_path)
            return True

        # 原有的路径关键词检查
        if 'platform' in file_path_lower and 'processor' not in file_path_lower:
            categories['platforms'].append(file_path)
            return True
        elif 'weapon' in file_path_lower:
            categories['weapons'].append(file_path)
            return True
        elif 'sensor' in file_path_lower:
            categories['sensors'].append(file_path)
            return True
        elif 'processor' in file_path_lower:
            categories['processors'].append(file_path)
            return True
        elif 'scenario' in file_path_lower:
            categories['scenarios'].append(file_path)
            return True
        elif 'behavior' in file_path_lower:
            categories['behaviors'].append(file_path)
            return True
        elif 'signature' in file_path_lower:
            categories['signatures'].append(file_path)
            return True
        elif 'script' in file_path_lower:
            categories['scripts'].append(file_path)
            return True
        
        # 兜底归类
        categories['other'].append(file_path)
        return False

    def _analyze_imports(self):
        """分析文件间的导入关系（仅针对 .txt 代码文件）"""
        for file_path, file_info in self.all_files.items():
            # === 修改：跳过 .md 文件，它们没有 include 语法 ===
            if file_path.lower().endswith('.md'):
                continue
                
            content = file_info['content']
            imports = self._extract_imports(content)
            if imports:
                self.import_dependencies[file_path] = imports

    def _extract_imports(self, content: str) -> List[str]:
        """从内容中提取导入语句"""
        imports = []
        content = self._remove_cui_header(content)
        lines = content.split('\n')

        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['include', 'import', 'from', 'require', 'using']):
                line = re.sub(r'#.*$', '', line)
                line = re.sub(r'//.*$', '', line)
                line = re.sub(r'/\*.*?\*/', '', line)
                line = line.strip()
                if line:
                    imports.append(line)
        return imports

    def _remove_cui_header(self, content: str) -> str:
        """移除AFSIM CUI版权声明头部"""
        header_patterns = [
            r'# \*+\s*CUI\s*\*+.*?# \*+',
            r'# \*+\s*The Advanced Framework for Simulation.*?and LICENSE for details.*?# \*+',
        ]

        for pattern in header_patterns:
            content = re.sub(pattern, '', content, count=1, flags=re.DOTALL | re.IGNORECASE)

        cui_header = re.compile(
            r'^\s*# \*+\s*\n\s*# CUI\s*\n\s*# \*+\s*\n.*?The Advanced Framework for Simulation.*?and LICENSE for details.*?# \*+\s*',
            re.DOTALL | re.IGNORECASE | re.MULTILINE
        )
        content = cui_header.sub('', content, count=1)
        return content.strip()

    def get_file_content(self, relative_path: str) -> str:
        """获取文件内容"""
        file_info = self.all_files.get(relative_path)
        return file_info['content'] if file_info else ""

    def find_related_files(self, query: str, max_results: int = 10) -> List[str]:
        """查找与查询相关的文件"""
        related_files = []
        query_lower = query.lower()
        
        # 简单的关键词匹配
        for file_path, file_info in self.all_files.items():
            content = file_info['content'].lower()
            if query_lower in content:
                related_files.append(file_path)

        return related_files[:max_results]

    def generate_context_prompt(self, query: str) -> str:
        """生成包含项目上下文的提示词 - 增强版（包含教程）"""
        context_parts = []

        # 1. 添加项目结构信息
        context_parts.append("=== AFSIM项目结构 ===")
        context_parts.append(f"基础库: {', '.join(self.base_libraries)}")
        context_parts.append(f"代码模块: {len(self.code_folders)} 个")
        
        # 显示教程数量
        tutorial_count = len(self.file_categories.get('tutorials', []))
        if tutorial_count > 0:
            context_parts.append(f"可用教程文档: {tutorial_count} 篇")
        context_parts.append("")

        # 2. 查找相关文件（混合代码和教程）
        related_files = self.find_related_files(query, 8) # 稍微增加数量以容纳教程
        
        related_code = []
        related_tutorials = []

        if related_files:
            for f in related_files:
                if f.endswith('.md'):
                    related_tutorials.append(f)
                else:
                    related_code.append(f)

        # 3. 优先展示相关教程（概念优先）
        if related_tutorials:
            context_parts.append("=== 📚 相关教程文档 ===")
            context_parts.append("以下是可能包含相关概念解释的教程文件：")
            for file_path in related_tutorials:
                # 获取教程内容的前500个字符作为摘要，或者根据查询词截取片段
                content = self.get_file_content(file_path)
                preview = content[:500].replace('\n', ' ') + "..."
                context_parts.append(f"- 文件: {file_path}")
                context_parts.append(f"  摘要: {preview}\n")
            context_parts.append("")

        # 4. 展示相关代码文件
        if related_code:
            context_parts.append("=== 💻 相关代码文件 ===")
            for file_path in related_code:
                context_parts.append(f"- {file_path}")
            context_parts.append("")
        
        if not related_code and not related_tutorials:
            context_parts.append("(未找到直接包含查询关键词的文件，将基于通用知识回答)")
            context_parts.append("")

        # 5. 添加文件分类概览
        context_parts.append("=== 文件类型分布 ===")
        for category, files in self.file_categories.items():
            if files and category != 'tutorials': # 教程已在上面单独处理
                context_parts.append(f"{category}: {len(files)} 个文件")
        context_parts.append("")

        # 6. 添加基础库的关键内容示例 (仅在没有具体教程时详细展示，节省上下文)
        if not related_tutorials:
            context_parts.append("=== 基础库概览 ===")
            for base_lib in self.base_libraries:
                lib_files = [f for f in self.all_files.keys() if f.startswith(base_lib)]
                if lib_files:
                    platform_files = [f for f in lib_files if 'platform' in f.lower()]
                    context_parts.append(f"{base_lib} 包含: 平台定义({len(platform_files)}), 武器, 传感器等。")

        return "\n".join(context_parts)

    def get_project_summary(self) -> Dict:
        """获取项目摘要"""
        return {
            "total_files": len(self.all_files),
            "base_libraries": self.base_libraries,
            "code_modules": self.code_folders,
            "file_categories": {k: len(v) for k, v in self.file_categories.items()},
            "total_size": sum(info['size'] for info in self.all_files.values())
        }

    def save_analysis_report(self, output_path: str):
        """保存分析报告"""
        report = {
            "project_summary": self.get_project_summary(),
            "file_categories_details": self.file_categories,
            "import_dependencies": self.import_dependencies
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"分析报告已保存到: {output_path}")