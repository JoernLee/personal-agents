"""
LangChain 接入百炼平台 Qwen3 模型示例
"""
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 加载环境变量
load_dotenv()

def setup_qwen_model():
    """设置 Qwen3 模型"""
    # 获取 API 密钥
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL")
    if not api_key:
        raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
    if not base_url:
        raise ValueError("请设置 DASHSCOPE_BASE_URL 环境变量")
    
    # 初始化 ChatOpenAI 模型
    llm = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base=base_url,
        model_name="qwen-plus",  # 可以选择不同的 Qwen 模型
        temperature=0.7,
        max_tokens=1024
    )
    
    return llm

def test_qwen_chat():
    """测试 Qwen3 聊天功能"""
    llm = setup_qwen_model()
    
    # 创建消息
    messages = [
        SystemMessage(content="你是一个有用的AI助手，请用中文回答问题。"),
        HumanMessage(content="请介绍一下 LangChain 框架的主要功能。")
    ]
    
    # 调用模型
    response = llm(messages)
    print("Qwen3 回复:")
    print(response.content)
    print("-" * 50)

def test_qwen_with_prompt_template():
    """使用 PromptTemplate 测试 Qwen3"""
    llm = setup_qwen_model()
    
    # 创建提示模板
    prompt = PromptTemplate(
        input_variables=["topic", "language"],
        template="请用{language}详细介绍{topic}，包括其主要特点和应用场景。"
    )
    
    # 创建链
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # 运行链
    result = chain.run(topic="人工智能", language="中文")
    print("使用 PromptTemplate 的结果:")
    print(result)
    print("-" * 50)

if __name__ == "__main__":
    print("开始测试百炼平台 Qwen3 模型接入...")
    print("=" * 60)
    
    try:
        # 测试基本聊天功能
        test_qwen_chat()
        
        # 测试 PromptTemplate
        test_qwen_with_prompt_template()
        
    except Exception as e:
        print(f"测试失败: {e}")
        print("请确保已设置 DASHSCOPE_API_KEY 环境变量")
