import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# ---------------- 配置区域 ----------------
# 加载 .env
load_dotenv(dotenv_path="../.env")

API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def generate_json_cases(requirement):
    print(f" 正在调用 AI 生成 JSON 数据：{requirement} ...")
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        # 【关键点 1】强制开启 JSON 模式，防止 AI 废话
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                # 【关键点 2】Prompt 里必须包含 'JSON' 字样，并规定字段
                "content": """
                你是一个资深测试工程师。
                请根据用户需求生成测试用例。
                必须返回符合以下结构的 JSON 格式数据（不要包含 Markdown 代码块标记）：
                {
                    "cases": [
                        {
                            "id": "TC_001",
                            "title": "用例标题",
                            "steps": "简略步骤",
                            "expected": "预期结果"
                        }
                    ]
                }
                """
            },
            {
                "role": "user",
                "content": requirement
            }
        ],
        temperature=0.1, # JSON 生成时温度要低，越低越严谨
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    # 结合你的业务场景
    requirement = "电商平台的购物车功能"
    
    try:
        json_str = generate_json_cases(requirement)
        
        # 【关键点 3】验证它是不是真的 JSON
        data = json.loads(json_str)
        
        print("\n 原始 JSON 字符串：")
        print(json_str)
        
        print("\n 解析后的数据（可以直接存 Excel 了）：")
        for case in data['cases']:
            print(f"[{case['id']}] {case['title']} -> 预期: {case['expected']}")
            
    except json.JSONDecodeError:
        print(" AI 没返回合法的 JSON，可能是 Prompt 没写对。")
    except Exception as e:
        print(f" 出错：{e}")