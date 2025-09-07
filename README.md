# Personal Agents 项目

这是一个基于 LangChain 的个人智能代理项目，支持多种模型接入和数据分析功能。

## 环境要求

- Python 3.8+ (推荐 3.12)
- 虚拟环境 (venv)

## 快速开始

### 1. 创建虚拟环境

```bash
python -m venv venv
```

### 2. 激活虚拟环境

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制 `env_template.txt` 为 `.env` 并填入你的 API 密钥：

```bash
cp env_template.txt .env
```

然后编辑 `.env` 文件，填入你的实际配置。

## 项目结构

```
personal-agents/
├── venv/                    # 虚拟环境
├── langchain/              # LangChain 相关代码
│   ├── csv-analyze/        # CSV 数据分析
│   └── basic-chain/        # 基础链式调用
├── tutorial-paper/         # 教程文档
├── requirements.txt        # 依赖列表
├── env_template.txt        # 环境变量模板
└── README.md              # 项目说明
```

## 已安装的库

- **langchain** (0.2.17) - LangChain 核心库
- **langchain-core** (0.2.43) - LangChain 核心组件
- **streamlit** (1.40.1) - Web 应用框架
- **pandas** (2.0.3) - 数据处理库
- **python-dotenv** (1.0.1) - 环境变量管理

## 支持的模型

### 1. 阿里云百炼平台 Qwen3 模型
- **qwen-plus** - 通用对话模型
- **qwen-turbo** - 快速响应模型
- **qwen-max** - 最强性能模型
- **qwen-long** - 长文本处理模型

### 2. 其他模型
- OpenAI GPT 系列
- DeepSeek 模型
- 其他兼容 OpenAI API 的模型

## 使用说明

1. 确保虚拟环境已激活
2. 配置好 `.env` 文件中的 API 密钥
3. 运行相应的 Python 脚本

## Qwen3 模型使用示例

### 基本使用

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# 设置 Qwen3 模型
llm = ChatOpenAI(
    openai_api_key="your_dashscope_api_key",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name="qwen-plus",
    temperature=0.7
)

# 发送消息
messages = [
    SystemMessage(content="你是一个有用的AI助手。"),
    HumanMessage(content="请介绍一下 LangChain。")
]

response = llm(messages)
print(response.content)
```

### 运行示例

```bash
# 测试 Qwen3 模型接入
python langchain/qwen-integration/main.py
```

## 注意事项

- 请确保你的 API 密钥安全，不要提交到版本控制系统
- 建议使用 Python 3.12 以获得最佳性能
- 如果遇到导入错误，请检查虚拟环境是否正确激活
