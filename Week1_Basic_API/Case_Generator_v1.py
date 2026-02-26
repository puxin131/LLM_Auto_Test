import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 1. 加载环境变量 (自动找根目录下的 .env)
load_dotenv(dotenv_path="../.env")  # 指向上一级目录

class CaseGenerator:
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL")
        
        if not self.api_key:
            raise ValueError(" 未找到 API Key，请检查 .env 文件！")
            
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_cases(self, feature_name):
        """
        核心方法：传入功能名，返回 Python 字典格式的用例
        """
        print(f" [AI] 正在为 '{feature_name}' 设计测试用例...")
        
        prompt = f"""
        你是一名资深测试专家。请为功能点“{feature_name}”设计 3-5 条核心测试用例。
        
        【必须返回纯 JSON 格式】：
        {{
            "feature": "{feature_name}",
            "test_cases": [
                {{
                    "id": "TC_001",
                    "title": "用例标题",
                    "priority": "P0/P1",
                    "steps": "1. 步骤A; 2. 步骤B",
                    "expected_result": "预期结果"
                }}
            ]
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a QA Assistant. Output JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2 
            )
            content = response.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            print(f" API 调用失败: {e}")
            return None

    def save_to_file(self, data, filename="generated_cases.json"):
        """
        工具方法：将结果保存为文件
        """
        if not data:
            return
            
        # 写入文件（使用 utf-8 防止中文乱码）
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f" 文件已保存至: {os.path.abspath(filename)}")

# ---------------- 执行入口 ----------------
if __name__ == "__main__":
    # 实例化生成器
    generator = CaseGenerator()
    
    # 你的真实业务场景
    target_feature = "用户订单列表翻页查询"
    
    # 1. 生成
    result = generator.generate_cases(target_feature)
    
    # 2. 打印预览
    if result:
        print(f"\n 生成成功，共 {len(result.get('test_cases', []))} 条用例")
        
    # 3. 落盘保存
    generator.save_to_file(result)