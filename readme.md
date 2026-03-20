#  AFSIM 智能代码生成与调试助手（轻量级 API 版）

一个基于 DeepSeek API 的 AFSIM 代码生成与调试辅助工具，支持本地知识库检索、代码修复、多模式交互，适用于 AFSIM 仿真建模与脚本开发。

---

## 项目结构

```
.
│  main.py                    # 主程序入口
│  readme.md                  # 项目说明文档
│  tasks.txt                  # 待处理任务清单
│  task_results_summary.txt   # 任务结果摘要
│  .env                       # apikey
├─cache                       # 缓存目录（自动生成）
├─tutorials                   # 本地知识库：AFSIM 教程与规范文档
│      afsim_command_and_reports.md
│      afsim_error_collection.md
│      afsim_mover_types_routes.md
│      afsim_platform_tutorial.md
│      afsim_scripting_and_processors.md
│      afsim_sensors_and_tracking.md
│      afsim_signatures_and_parts.md
│      afsim_weapons_and_commands.md
│      afsim_results_visualization.md
└─__pycache__                 # Python 缓存文件（自动生成）
```

---

## 核心功能

### 1. **智能代码生成**
- 根据自然语言需求生成符合 AFSIM 语法的代码
- 支持多种文件格式：`.txt`, `.md`, `.cpp`, `.h`, `.json`, `.xml`
- 内置本地知识库检索机制，确保生成代码符合项目规范

### 2. **分步调试与修复**
- 支持"修复模式"：用户可依次输入任务目标、错误代码、报错信息
- 自动分析错误原因，并提供修复后的完整代码
- 历史对话支持，便于连续调试

### 3. **本地知识库支持**
- 自动加载 `./tutorials` 目录下的教程文件作为参考
- 知识库内容会作为系统提示词的一部分，提升生成准确性

### 4. **交互式多行输入**
- 支持 `//end` 作为多行输入结束标志
- 支持 `exit` 退出程序，`clear` 清空对话历史

---

## 快速开始

### 环境要求
- Python 3.8+
- 网络连接（用于调用 DeepSeek API）
- 有效的 DeepSeek API Key 或者其他可用的api提供商

### 安装依赖
```bash
pip install -r requirements.txt
```

### 配置 API Key

在根目录下新建`.env`

编辑 `.env` 文件，填入您的 API Key：
```env
   DEEPSEEK_API_KEY=sk-your-actual-api-key  # 替换为你的 API Key
   DEEPSEEK_API_BASE=https://api.deepseek.com  # 或者其他API提供商
```

### 启动程序
```bash
python main.py
```

---

## 使用指南

### 模式选择
启动后，系统提供两种模式：

#### 1. **生成模式（模式1）**
- 直接输入自然语言需求，系统生成 AFSIM 代码
- 示例输入：
```
创建一个战斗机平台，包含雷达和空对空导弹
必须包含运动模型和通信网络
//end
```

#### 2. **修复模式（模式2）**
- 分步输入：
  1. **任务目标**
  2. **之前的错误代码**
  3. **编译器报错信息**
- 系统将分析错误并提供修复建议

### 常用命令
- `//end` – 提交多行输入
- `exit` – 退出程序
- `clear` – 清空对话历史
- `1` 或 `2` – 选择操作模式

---

## 知识库管理

### 目录结构
- `./tutorials/` 存放所有教程文档
- 支持格式：`.txt`, `.md`, `.cpp`, `.h`, `.json`, `.xml`

### 添加新的教程
只需将文档放入 `tutorials/` 目录，重启程序即可自动加载。

---

## 配置说明

### 主要参数（main.py 顶部）
| 变量名 | 说明 |
|--------|------|
| `API_BASE_URL` | DeepSeek API 地址 |
| `API_KEY` | 你的 API 密钥 |
| `MODEL_NAME` | 模型名称（默认 deepseek-chat） |
| `KNOWLEDGE_DIR` | 本地知识库目录（默认 ./tutorials） |

### 系统提示词
- 包含知识库内容作为参考
- 支持生成与修复两种模式

---

## 常见问题

### 1. API 调用失败
- 检查网络连接
- 确认 API Key 有效
- 查看控制台输出的错误信息

### 2. 知识库加载失败
- 确认 `tutorials` 目录存在且包含有效文件
- 检查文件编码是否为 UTF-8

### 3. 输入不响应
- 确保输入以 `//end` 结束
- 检查是否在输入过程中误按了 Ctrl+C

---

## 📄 许可证

仅供学习和研究使用，请遵守 DeepSeek API 使用协议。

---

## 支持与反馈

如遇到问题或建议，请：
1. 检查 `tutorials` 目录内容是否完整
2. 确认 API Key 权限和额度
3. 查看程序控制台输出信息

---

**开始使用：**
```bash
python main.py
```