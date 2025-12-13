# multi_stage_generator.py
import os
import json
import re
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from utils import ConfigManager  # 导入ConfigManager

@dataclass
class GenerationStage:
    """生成阶段定义"""
    name: str
    description: str
    max_tokens: int = 2048
    temperature: float = 0.2
    depends_on: List[str] = field(default_factory=list)
    output_patterns: List[str] = field(default_factory=list)

class AFSimProjectStructure:
    """AFSIM项目结构分析器"""
    
    def __init__(self):
        # 获取配置管理器
        self.config = ConfigManager()
        # 基础文件
        self.base_files = [
            "main.txt",
            "README.md",
            "project_structure.json"
        ]
    
    def analyze_requirements(self, query: str) -> Dict:
        """分析需求，确定项目结构"""
        query_lower = query.lower()
                
        # 检测需要的组件
        components = self._detect_components(query_lower)
        
        # 构建项目结构
        structure = self._build_project_structure(components)
        
        return {
            "components": components,
            "structure": structure,
            "stages": self._generate_stages(components)
        }
    
    def _detect_components(self, query: str) -> Dict[str, bool]:
        """检测需要的组件"""
        return {
            "platforms": any(word in query for word in [
                "平台", "导弹", "炸弹", "车", "卫星", "船", "坦克", "飞行器", "飞机", "发射车"
            ]),
            "scenarios": any(word in query for word in [
                "红", "蓝", "队", "对抗"
            ]),
            "processors": any(word in query for word in [
                "处理器", "控制", "制导", "跟踪"
            ]),
            "weapons": any(word in query for word in [
                "武器平台", "武器", "导弹", "拦截弹","火箭", "炸弹", "火炮"
            ]),
            "sensors": any(word in query for word in [
                "传感器", "雷达", "探测", "跟踪", "红外", "光学"
            ]),
            "signatures": any(word in query for word in [
                "特征", "雷达反射", "红外特征", "光学特征", "雷达截面积", "隐身"
            ]),
        }
    
    def _build_project_structure(self, components: Dict) -> Dict:
        """构建项目结构"""
        structure = {
            "files": self.base_files.copy(),
            "folders": []
        }
        
        # 根据检测到的组件添加相应的文件夹
        folder_mapping = {
            "platforms": "platforms",
            "scenarios": "scenarios",
            "processors": "processors",
            "weapons": "weapons",
            "sensors": "sensors",
            "signatures": "signatures",
        }

        # 添加检测到的组件的文件夹
        for component, has_component in components.items():
            if has_component and component in folder_mapping:
                folder_name = folder_mapping[component]
                if folder_name not in structure["folders"]:
                    structure["folders"].append(folder_name)
        
        # 确保至少有平台和场景文件夹（大部分项目都需要）
        if "platforms" not in structure["folders"] and components["platforms"]:
            structure["folders"].append("platforms")
        if "scenarios" not in structure["folders"] and components["scenarios"]:
            structure["folders"].append("scenarios")
        
        # 排序文件夹，让常用文件夹在前面
        preferred_order = ["platforms", "scenarios", "weapons", "sensors", "processors"]
        structure["folders"] = sorted(
            structure["folders"],
            key=lambda x: (preferred_order.index(x) if x in preferred_order else len(preferred_order), x)
        )
        
        return structure
    
    def _generate_stages(self, components: Dict) -> List[Dict]:
        """生成阶段计划，从config.yaml读取参数"""
        # 从配置获取阶段定义
        config_stages = self.config.get('generation.stages', [])
        
        # 创建阶段列表
        stages = []
        
        # 首先添加项目结构阶段
        project_structure_stage = next(
            (stage for stage in config_stages if stage['name'] == 'project_structure'),
            {
                "name": "project_structure",
                "description": "分析需求并规划项目结构",
                "max_tokens": 300,
                "temperature": 0.1
            }
        )
        stages.append({
            "name": project_structure_stage["name"],
            "description": project_structure_stage["description"],
            "max_tokens": project_structure_stage.get("max_tokens", 300),
            "temperature": project_structure_stage.get("temperature", 0.1),
            "depends_on": [],
            "output_patterns": ["project_structure.json"]
        })

        # 添加主程序阶段
        main_program_stage = next(
            (stage for stage in config_stages if stage['name'] == 'main_program'),
            {
                "name": "main_program",
                "description": "生成主程序文件",
                "max_tokens": 800,
                "temperature": 0.2
            }
        )
        stages.append({
            "name": main_program_stage["name"],
            "description": main_program_stage["description"],
            "max_tokens": main_program_stage.get("max_tokens", 800),
            "temperature": main_program_stage.get("temperature", 0.2),
            "depends_on": ["project_structure"],
            "output_patterns": ["main.txt"]
        })
        
        # 根据检测到的组件添加相应阶段
        component_stage_mapping = {
            "platforms": {
                "config_name": "platforms",
                "default": {
                    "name": "platforms",
                    "description": "生成平台定义文件",
                    "max_tokens": 1200,
                    "temperature": 0.15
                }
            },
            "scenarios": {
                "config_name": "scenarios",
                "default": {
                    "name": "scenarios",
                    "description": "生成场景文件",
                    "max_tokens": 1000,
                    "temperature": 0.15
                }
            },
            "processors": {
                "config_name": "processors",
                "default": {
                    "name": "processors",
                    "description": "生成处理器文件",
                    "max_tokens": 900,
                    "temperature": 0.15
                }
            },
            "sensors": {
                "config_name": "sensors",
                "default": {
                    "name": "sensors",
                    "description": "生成传感器文件",
                    "max_tokens": 700,
                    "temperature": 0.15
                }
            },
            "weapons": {
                "config_name": "weapons",
                "default": {
                    "name": "weapons",
                    "description": "生成武器文件",
                    "max_tokens": 700,
                    "temperature": 0.15
                }
            },
            "signatures": {
                "config_name": None,  # 配置中可能没有signatures阶段
                "default": {
                    "name": "signatures",
                    "description": "生成特征信号文件",
                    "max_tokens": 600,
                    "temperature": 0.1
                }
            }
        }

        # 添加检测到的组件的阶段
        for component, has_component in components.items():
            if has_component and component in component_stage_mapping:
                mapping = component_stage_mapping[component]
                
                # 从配置获取阶段参数或使用默认值
                if mapping["config_name"]:
                    stage_config = next(
                        (stage for stage in config_stages if stage['name'] == mapping["config_name"]),
                        mapping["default"]
                    )
                else:
                    stage_config = mapping["default"]
                
                # 设置依赖关系
                depends_on = ["project_structure"]
                if component == "scenarios":
                    depends_on = ["project_structure", "platforms"]
                elif component in ["processors", "sensors", "weapons"]:
                    depends_on = ["project_structure", "platforms"]
                
                # 创建阶段对象
                stage = {
                    "name": stage_config["name"],
                    "description": stage_config["description"],
                    "max_tokens": stage_config.get("max_tokens", mapping["default"]["max_tokens"]),
                    "temperature": stage_config.get("temperature", mapping["default"]["temperature"]),
                    "depends_on": depends_on,
                    "output_patterns": [f"{stage_config['name']}/*.txt"]
                }
                
                # 检查是否已存在同名阶段
                if not any(s["name"] == stage["name"] for s in stages):
                    stages.append(stage)
            
        return stages

class MultiStageGenerator:
    """多阶段生成器"""
    
    def __init__(self, chat_system, config):
        self.chat_system = chat_system
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.project_analyzer = AFSimProjectStructure()
        
        # 项目状态
        self.current_project = None
        self.generated_files = []
        self.current_stage = None
        self.project_context = {}
        self.stage_results = {}

    def _execute_stage(self, stage_info: Dict, query: str, output_dir: str) -> Dict:
        """执行单个生成阶段"""
        stage_name = stage_info["name"]
        stage_max_tokens = stage_info.get("max_tokens", 1024)
        stage_temperature = stage_info.get("temperature", 0.3)
        
        try:
            print(f"\n🔧 开始执行阶段: {stage_name}")
            print(f"   阶段描述: {stage_info.get('description', '')}")
            print(f"   生成参数: max_tokens={stage_max_tokens}, temperature={stage_temperature}")
            
            # 构建阶段特定的提示词
            prompt = self._build_stage_prompt(stage_info, query)
            
            print(f"📝 提示词长度: {len(prompt)} 字符")
            print(f"📝 提示词前200字符:\n{prompt[:200]}...")
            
            # 生成内容
            start_gen_time = time.time()
            
            # 检查是否有生成参数的方法
            if hasattr(self.chat_system, 'generate_enhanced_response_with_params'):
                print("   使用带参数的生成方法...")
                result = self.chat_system.generate_enhanced_response_with_params(
                    prompt, 
                    max_tokens=stage_max_tokens,
                    temperature=stage_temperature
                )
            elif hasattr(self.chat_system, 'generate_enhanced_response'):
                print("   使用增强响应生成方法...")
                result = self.chat_system.generate_enhanced_response(prompt)
            else:
                print("   使用默认生成方法...")
                # 尝试直接调用
                result = self.chat_system(prompt)
            
            gen_duration = time.time() - start_gen_time
            print(f"✅ 生成完成，耗时: {gen_duration:.2f}秒")
            
            if not result or "result" not in result:
                error_msg = "生成结果为空"
                print(f"❌ {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            # 解析生成的内容
            generated_content = result["result"]
            
            print(f"📝 阶段 {stage_name} 生成内容长度: {len(generated_content)} 字符")
            print(f"📝 生成内容前300字符:\n{generated_content[:300]}...")
            
            # 提取文件内容
            extract_start = time.time()
            files = self._extract_files_from_content(generated_content, stage_info, output_dir)
            extract_duration = time.time() - extract_start
            
            print(f"📄 阶段 {stage_name} 提取到 {len(files)} 个文件，耗时: {extract_duration:.2f}秒")
            
            # 保存文件
            save_start = time.time()
            output_files = self._save_generated_files(files, output_dir)
            save_duration = time.time() - save_start
            
            # 更新上下文
            context = self._extract_context_from_content(generated_content)
            
            print(f"💾 保存文件完成，耗时: {save_duration:.2f}秒")
            
            return {
                "success": True,
                "output_files": output_files,
                "context": context,
                "raw_content": generated_content[:200] + "..." if len(generated_content) > 200 else generated_content,
                "stage_name": stage_name,
                "generation_params": {
                    "max_tokens": stage_max_tokens,
                    "temperature": stage_temperature
                }
            }
            
        except Exception as e:
            self.logger.error(f"执行阶段 {stage_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }
        
    def _build_stage_query(self, stage_info: Dict, query: str) -> str:
        """构建阶段特定的查询"""
        stage_name = stage_info["name"]
        
        # 阶段特定的查询增强
        stage_queries = {
            "project_structure": f"分析以下AFSIM项目需求并生成项目结构规划:\n{query}",
            "main_program": f"根据项目需求生成主程序文件，需求:\n{query}",
            "platforms": f"生成平台定义，基于项目需求:\n{query}\n已确定平台: {self.project_context.get('platforms', [])}",
            "scenarios": f"生成场景文件，基于项目需求:\n{query}\n可用平台: {self.project_context.get('platforms', [])}",
            "processors": f"生成处理器文件，基于项目需求:\n{query}\n平台上下文: {self.project_context.get('platforms', [])}",
            "sensors": f"生成传感器文件，基于项目需求:\n{query}\n平台上下文: {self.project_context.get('platforms', [])}",
            "weapons": f"生成武器文件，基于项目需求:\n{query}\n平台上下文: {self.project_context.get('platforms', [])}",
            "signatures": f"生成特征信号文件，基于项目需求:\n{query}\n平台类型: {self.project_context.get('platforms', [])}"
        }
        
        return stage_queries.get(stage_name, f"生成{stage_info['description']}，需求:\n{query}")
        
    def generate_project(self, query: str, output_dir: str = None) -> Dict:
        """生成完整的AFSIM项目"""
        try:
            # 1. 分析需求
            self.logger.info("分析项目需求...")
            print("🔍 分析项目需求...")
            project_analysis = self.project_analyzer.analyze_requirements(query)
            
            print(f"✅ 需求分析完成:")
            print(f"   检测到组件: {project_analysis['components']}")
            print(f"   生成阶段: {len(project_analysis['stages'])} 个")
            
            # 2. 准备输出目录
            if not output_dir:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_dir = os.path.join(
                    self.config.get('generation.output.base_dir', 'generated_projects'),
                    f"afsim_project_{timestamp}"
                )
            
            os.makedirs(output_dir, exist_ok=True)
            print(f"📁 输出目录: {output_dir}")
            
            # 3. 保存项目分析
            self.current_project = {
                "analysis": project_analysis,
                "output_dir": output_dir,
                "query": query,
                "start_time": time.time(),
                "stages": {}
            }
            
            # 创建项目结构
            self._create_project_structure(output_dir, project_analysis["structure"])
            
            # 4. 按阶段生成
            stages = project_analysis["stages"]
            total_stages = len(stages)
            
            print(f"\n🚀 开始执行 {total_stages} 个生成阶段...")
            
            for idx, stage_info in enumerate(stages, 1):
                stage_name = stage_info["name"]
                stage_desc = stage_info["description"]
                stage_max_tokens = stage_info.get("max_tokens", 1024)
                stage_temperature = stage_info.get("temperature", 0.3)
                
                self.current_stage = stage_name
                print(f"\n{'='*60}")
                print(f"📋 阶段 {idx}/{total_stages}: {stage_name}")
                print(f"   描述: {stage_desc}")
                print(f"   参数: max_tokens={stage_max_tokens}, temperature={stage_temperature}")
                
                # 检查依赖
                if not self._check_stage_dependencies(stage_info):
                    self.logger.warning(f"阶段 {stage_name} 的依赖未满足，跳过")
                    print(f"⚠️  跳过阶段 {stage_name}（依赖未满足）")
                    continue
                
                # 执行阶段生成
                stage_start = time.time()
                result = self._execute_stage(stage_info, query, output_dir)
                stage_duration = time.time() - stage_start
                
                # 记录结果
                self.current_project["stages"][stage_name] = {
                    "status": "success" if result["success"] else "failed",
                    "output_files": result.get("output_files", []),
                    "error": result.get("error"),
                    "duration": stage_duration,
                    "max_tokens": stage_max_tokens,
                    "temperature": stage_temperature
                }
                
                if result["success"]:
                    # 去重添加文件
                    for file_path in result.get("output_files", []):
                        if file_path not in self.generated_files:
                            self.generated_files.append(file_path)
                    
                    self.project_context.update(result.get("context", {}))
                    self.stage_results[stage_name] = result
                    print(f"✅ 阶段 {stage_name} 完成 ({stage_duration:.1f}秒)")
                    if result.get("output_files"):
                        print(f"   生成文件: {', '.join(result['output_files'])}")
                else:
                    self.logger.error(f"阶段 {stage_name} 失败: {result.get('error')}")
                    print(f"❌ 阶段 {stage_name} 失败: {result.get('error')}")
            
            # 5. 生成项目报告
            report = self._generate_project_report()
            
            self.logger.info(f"项目生成完成: {output_dir}")
            print(f"\n{'='*60}")
            print(f"🎉 项目生成完成！位置: {output_dir}")
            print(f"📄 总共生成 {len(self.generated_files)} 个文件")
            
            return {
                "success": True,
                "project_dir": output_dir,
                "generated_files": self.generated_files,
                "report": report,
                "project_analysis": project_analysis
            }
            
        except Exception as e:
            self.logger.error(f"项目生成失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_project_structure(self, output_dir: str, structure: Dict):
        """创建项目文件夹结构"""
        self.logger.info(f"创建项目结构: {output_dir}")
        
        # 创建文件夹
        for folder in structure.get("folders", []):
            folder_path = os.path.join(output_dir, folder)
            os.makedirs(folder_path, exist_ok=True)
            self.logger.debug(f"创建文件夹: {folder_path}")
    
    def _check_stage_dependencies(self, stage_info: Dict) -> bool:
        """检查阶段依赖是否满足"""
        depends_on = stage_info.get("depends_on", [])
        if not depends_on:
            return True
        
        for dep in depends_on:
            if dep not in self.current_project["stages"]:
                return False
            if self.current_project["stages"][dep]["status"] != "success":
                return False
        
        return True
    
    def _build_stage_prompt(self, stage_info: Dict, query: str) -> str:
        """构建阶段特定的提示词"""
        stage_name = stage_info["name"]
        
        # 更简洁明确的阶段特定提示词
        stage_instructions = {
            "project_structure": f"""生成项目结构JSON。

需求：{query}

输出JSON格式：
{{
  "components": ["平台组件列表"],
  "file_structure": {{
    "folders": ["文件夹列表"],
    "files": ["文件列表"]
  }},
  "main_platform": "主要平台名称",
  "scenario_description": "场景描述"
}}

只输出JSON，不要任何其他文字。""",
            
            "main_program": f"""生成AFSIM主程序文件。

需求：{query}

输出有效的AFSIM代码，包含：
1. include语句
2. 平台定义
3. 场景定义
4. 输出配置
5. 仿真控制

只输出AFSIM代码，不要任何解释:""",
            
            "platforms": f"""生成AFSIM平台定义。

需求：{query}

只输出AFSIM代码，不要任何解释:""",
            
            "scenarios": f"""生成AFSIM场景文件。

需求：{query}

只输出AFSIM代码，不要任何解释:。"""
        }
        
        instruction = stage_instructions.get(stage_name, f"根据需求生成{stage_info['description']}。\n需求：{query}")
        
        instruction += "\n\n只输出AFSIM代码，不要任何解释:"
        
        return instruction

    def _get_platform_requirements(self) -> str:
        """获取平台需求描述"""
        if "platforms" in self.project_context:
            platforms = self.project_context["platforms"]
            return "\n".join([f"- {p}" for p in platforms])
        return "根据项目需求生成合适的平台"
    
    def _clean_generated_content(self, content: str, stage_name: str) -> str:
        """清理生成的内容 - 更严格的版本"""
        
        # 移除所有引导性和解释性文字
        patterns_to_remove = [
            r'^现在，请.*$', r'^以下是.*$', r'^该代码.*$', r'^您提供的代码.*$',
            r'^修正后的代码.*$', r'^在AFSIM中.*$', r'^注意.*$', r'^确保.*$',
            r'^禁止.*$', r'^由于.*$', r'^根据.*要求.*$',
            r'```[a-z]*\n', r'\n```',  # Markdown代码块
            r'^\[.*\]$',  # 方括号内容
            r'^输出：.*$', r'^生成：.*$',
        ]
        
        for pattern in patterns_to_remove:
            content = re.sub(pattern, '', content, flags=re.MULTILINE | re.IGNORECASE)
        
        # 移除重复的代码块
        lines = content.split('\n')
        seen_lines = set()
        unique_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
                
            # 跳过重复的行（对于平台定义特别重要）
            if line_stripped in seen_lines:
                continue
                
            seen_lines.add(line_stripped)
            unique_lines.append(line)
        
        content = '\n'.join(unique_lines)
        
        # 移除多余的空行
        content = re.sub(r'\n\s*\n+', '\n\n', content)
        
        return content.strip()
    
    def _extract_files_from_content(self, content: str, stage_info: Dict, output_dir: str) -> List[Dict]:
        """从生成的内容中提取文件"""
        files = []
        stage_name = stage_info["name"]
        
        print(f"🔍 提取阶段 {stage_name} 的内容...")
        
        if stage_name == "project_structure":
            # 直接查找并提取 JSON
            import re
            
            print(f"   查找JSON内容...")
            
            # 尝试直接提取大括号中的内容
            json_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
            matches = re.findall(json_pattern, content, re.DOTALL)
            
            if matches:
                print(f"   找到 {len(matches)} 个可能的JSON块")
                
                for i, json_str in enumerate(matches):
                    try:
                        json_data = json.loads(json_str)
                        print(f"   JSON块 {i+1} 解析成功")
                        
                        # 验证基本结构
                        if isinstance(json_data, dict):
                            # 确保必需字段存在
                            if "components" not in json_data:
                                json_data["components"] = []
                            if "file_structure" not in json_data:
                                json_data["file_structure"] = {"folders": [], "files": []}
                            if "main_platform" not in json_data:
                                json_data["main_platform"] = ""
                            if "scenario_description" not in json_data:
                                json_data["scenario_description"] = ""
                            
                            files.append({
                                "path": "project_structure.json",
                                "content": json.dumps(json_data, indent=2, ensure_ascii=False)
                            })
                            
                            # 更新上下文
                            self.project_context.update(json_data)
                            print(f"   ✅ 提取到有效JSON")
                            break  # 找到第一个有效JSON就停止
                            
                    except json.JSONDecodeError as e:
                        print(f"   JSON块 {i+1} 解析失败: {e}")
                        continue
                
        elif stage_name == "main_program":
            # 提取 main.txt 内容
            main_content = self._extract_main_program(content)
            if main_content:
                files.append({
                    "path": "main.txt",
                    "content": main_content
                })
                print(f"   ✅ 提取到 main.txt 内容")
            else:
                # 如果没提取到内容，创建默认的main.txt
                print(f"   ⚠️ 未提取到有效内容，创建默认main.txt")
                
                default_main = f"""# AFSIM 主程序文件
# 基于需求生成: {self.current_project.get('query', '')[:100]}

include_once base_types/platforms/tank_type_a.txt

platform_type Default_Platform WSF_PLATFORM
icon default
mover WSF_GROUND_MOVER

scenario default_scenario
description "默认场景"
duration 600.0 sec

output_config
enable_output true
output_frequency 10 Hz

simulation_control
max_time 60 s
time_step 0.1 s
log true"""
                
                files.append({
                    "path": "main.txt",
                    "content": default_main
                })
                
        else:
            # 对于其他阶段，使用智能文件分割
            extracted_files = self._extract_multiple_files_smart(content, stage_name)
            files.extend(extracted_files)
            print(f"   📄 提取到 {len(extracted_files)} 个文件")
        
        return files
    
    def _extract_main_program(self, content: str) -> str:
        """专门提取main.txt内容"""
        # 查找AFSIM代码的开始
        patterns = [
            r'(platform_type[\s\S]*?simulation_control[\s\S]*?log true)',
            r'(include[\s\S]*?simulation_control[\s\S]*?log true)',
            r'(platform_type[\s\S]*?scenario[\s\S]*?end_scenario)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1)
        
        # 如果没有找到完整结构，返回清理后的内容
        lines = []
        code_started = False
        
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # 检测代码开始
            if any(keyword in line.lower() for keyword in ['platform_type', 'include', 'scenario']):
                code_started = True
                
            if code_started and '禁止' not in line and '注意' not in line:
                lines.append(line)
        
        return '\n'.join(lines) if lines else content[:500]
    
    def _extract_multiple_files_smart(self, content: str, folder_name: str) -> List[Dict]:
        """智能提取多个文件"""
        files = []
        
        # 多种文件分隔模式
        patterns = [
            (r'=== (.+?\.txt) ===\n(.*?)(?=\n=== |\Z)', re.DOTALL),  # === 文件名.txt ===
            (r'// File: (.+?\.txt)\n(.*?)(?=\n// File: |\Z)', re.DOTALL),  # // File: 文件名.txt
            (r'# File: (.+?\.txt)\n(.*?)(?=\n# File: |\Z)', re.DOTALL),  # # File: 文件名.txt
            (r'文件：(.+?\.txt)\n(.*?)(?=\n文件：|\Z)', re.DOTALL),  # 文件：文件名.txt
            (r'(\w+)_platform\.txt:\n(.*?)(?=\n\w+_platform\.txt:|\Z)', re.DOTALL),  # 平台名_platform.txt:
        ]
        
        for pattern, flags in patterns:
            matches = re.findall(pattern, content, flags)
            if matches:
                for filename, file_content in matches:
                    # 清理文件名
                    filename = filename.strip()
                    if not filename.endswith('.txt'):
                        filename += '.txt'
                    
                    # 清理文件内容
                    file_content = file_content.strip()
                    
                    files.append({
                        "path": f"{folder_name}/{filename}",
                        "content": file_content
                    })
                break
        
        # 如果没有明确的分割，尝试其他方法
        if not files:
            files = self._extract_files_by_platform_pattern(content, folder_name)
        
        # 如果还是没有找到文件，创建一个默认文件
        if not files and content.strip():
            default_name = f"{folder_name}_main.txt"
            files.append({
                "path": f"{folder_name}/{default_name}",
                "content": content.strip()
            })
        
        return files
    
    def _extract_files_by_platform_pattern(self, content: str, folder_name: str) -> List[Dict]:
        """根据平台模式提取文件"""
        files = []
        
        # 查找平台定义
        platform_patterns = [
            r'platform_type\s+(\w+)',
            r'class\s+(\w+)\s*\{',
            r'(\w+)_platform\s*\{'
        ]
        
        all_platforms = []
        for pattern in platform_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            all_platforms.extend(matches)
        
        # 为每个平台提取相关内容
        for platform in set(all_platforms):
            # 查找该平台的相关内容
            platform_content = self._extract_platform_content(content, platform)
            if platform_content:
                filename = f"{platform}.txt"
                files.append({
                    "path": f"{folder_name}/{filename}",
                    "content": platform_content
                })
        
        return files
    
    def _extract_platform_content(self, content: str, platform: str) -> str:
        """提取特定平台的内容"""
        # 查找以平台名开始的部分
        patterns = [
            fr'platform_type\s+{platform}.*?\n}}(?=\n|$)' if '}' in content else fr'platform_type\s+{platform}.*?(?=\nplatform_type|\Z)',
            fr'class\s+{platform}.*?\n}}(?=\n|$)' if '}' in content else fr'class\s+{platform}.*?(?=\nclass|\Z)',
            fr'{platform}_platform.*?\n}}(?=\n|$)' if '}' in content else fr'{platform}_platform.*?(?=\n\w+_platform|\Z)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group().strip()
        
        return ""
    
    def _save_generated_files(self, files: List[Dict], output_dir: str) -> List[str]:
        """保存生成的文件"""
        saved_files = []
        
        for file_info in files:
            try:
                file_path = os.path.join(output_dir, file_info["path"])
                
                # 确保目录存在
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # 保存文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(file_info["content"])
                
                saved_files.append(file_info["path"])
                self.logger.info(f"✅ 保存文件: {file_info['path']} ({len(file_info['content'])} 字符)")
                
                # 输出调试信息
                print(f"   ✓ 保存: {file_info['path']}")
                
            except Exception as e:
                error_msg = f"保存文件失败 {file_info['path']}: {e}"
                self.logger.error(error_msg)
                print(f"   ✗ 失败: {error_msg}")
        
        return saved_files
    
    def _extract_context_from_content(self, content: str) -> Dict:
        """从内容中提取上下文信息"""
        context = {}
        
        # 提取平台名称
        platform_matches = re.findall(r'platform_type\s+(\w+)', content, re.IGNORECASE)
        if platform_matches:
            context["platforms"] = list(set(platform_matches))
        
        # 提取武器名称
        weapon_matches = re.findall(r'weapon_type\s+(\w+)', content, re.IGNORECASE)
        if weapon_matches:
            context["weapons"] = list(set(weapon_matches))
        
        # 提取传感器名称
        sensor_matches = re.findall(r'sensor_type\s+(\w+)', content, re.IGNORECASE)
        if sensor_matches:
            context["sensors"] = list(set(sensor_matches))
        
        # 提取场景名称
        scenario_matches = re.findall(r'scenario\s+(\w+)', content, re.IGNORECASE)
        if scenario_matches:
            context["scenarios"] = list(set(scenario_matches))
        
        return context
    
    def _generate_project_report(self) -> Dict:
        """生成项目报告"""
        if not self.current_project:
            return {}
        
        total_duration = time.time() - self.current_project["start_time"]
        
        return {
            "project_info": {
                "output_dir": self.current_project["output_dir"],
                "query": self.current_project["query"],
                "total_duration": total_duration,
                "generated_files_count": len(self.generated_files)
            },
            "analysis": self.current_project["analysis"],
            "stage_results": self.current_project["stages"],
            "file_list": self.generated_files,
            "summary": {
                "total_stages": len(self.current_project["stages"]),
                "successful_stages": sum(1 for s in self.current_project["stages"].values() 
                                       if s["status"] == "success"),
                "total_files": len(self.generated_files),
                "avg_stage_duration": total_duration / max(len(self.current_project["stages"]), 1),
                "stage_params": {
                    stage_name: {
                        "max_tokens": stage_info.get("max_tokens"),
                        "temperature": stage_info.get("temperature")
                    }
                    for stage_name, stage_info in self.current_project["stages"].items()
                }
            }
        }


class MultiStageChatSystem:
    """支持多阶段生成的聊天系统"""
    
    def __init__(self, project_root: str, model_path: str = None):
        from rag_enhanced import EnhancedRAGChatSystem
        from utils import setup_logging, ConfigManager
        
        # 设置日志
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 初始化基础RAG系统
        self.chat_system = EnhancedRAGChatSystem(
            project_root=project_root,
            model_path=model_path
        )
        
        # 加载配置
        self.config = ConfigManager()
        
        # 初始化多阶段生成器
        self.project_analyzer = AFSimProjectStructure()
        self.multi_stage_generator = MultiStageGenerator(self.chat_system, self.config)
    
    def generate_complete_project(self, query: str, output_dir: str = None) -> Dict:
        """生成完整的AFSIM项目"""
        self.logger.info(f"开始生成完整项目: {query[:100]}...")
        
        # 使用多阶段生成器
        result = self.multi_stage_generator.generate_project(query, output_dir)
        
        # 记录到对话历史
        self.chat_system.conversation_history.append({
            'query': query,
            'type': 'project_generation',
            'result': result,
            'timestamp': time.time()
        })
        
        return result
    
    def get_project_info(self):
        """获取项目信息"""
        return self.chat_system.get_project_info()
    
    def get_vector_db_info(self):
        """获取向量数据库信息"""
        return self.chat_system.get_vector_db_info()