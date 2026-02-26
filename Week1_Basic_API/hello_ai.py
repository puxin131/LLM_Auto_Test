import os
from openai import OpenAI
from dotenv import load_dotenv 

# 加载上一级目录的 .env 文件
load_dotenv(dotenv_path="../.env") 

# 从环境变量获取 Key
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# 增加一个安全检查（可选，但推荐）
if not API_KEY:
    raise ValueError("❌ 未找到 API Key，请检查 .env 文件！")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ---------------- 业务逻辑 ----------------
def generate_test_cases(requirement):
    """
    输入一个简单的需求，让 AI 生成测试用例
    """
    print(f" 正在思考需求：{requirement} ...")
    
    response = client.chat.completions.create(
        model="deepseek-chat",  # 或者是 deepseek-reasoner (R1)
        messages=[
            {
                "role": "system", 
                "content": "你是一个资深测试工程师。请根据用户输入的简短需求，生成 3 条核心功能的测试用例。格式要求：\n1. 用例标题\n2. 前置条件\n3. 测试步骤\n4. 预期结果"
            },
            {
                "role": "user", 
                "content": requirement
            }
        ],
        temperature=0.3,  # 温度设低点，让回答更严谨、稳定
        stream=False
    )
    
    # 获取并返回 AI 的回答内容
    return response.choices[0].message.content

# ---------------- 主程序 ----------------
if __name__ == "__main__":
    # 模拟一个只有 5 个字的需求
    user_requirement = "票务退票功能"
    
    try:
        result = generate_test_cases(user_requirement)
        print("\n AI 生成结果如下：\n" + "="*30)
        print(result)
        print("="*30)
    except Exception as e:
        print(f"\n 出错了：{e}")
        print("检查一下 API Key 是否正确，或者网络是否通畅。")