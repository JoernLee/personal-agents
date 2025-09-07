import os
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv 
load_dotenv(override=True)

DeepSeek_API_KEY = os.getenv("DEEPSEEK_API_KEY")
print(DeepSeek_API_KEY)  # 可以通过打印

model = init_chat_model(model="deepseek-chat", model_provider="deepseek")

basic_qa_chain = model | StrOutputParser()

question = "你好，请你介绍一下你自己。"
result = basic_qa_chain.invoke(question)
print(result)









