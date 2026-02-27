import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 强制加载项目根目录下的 .env 文件
load_dotenv(dotenv_path="../.env", override=True)

class CaseGenerator:
    def __init__(self):
        # 从 .env 读取你的凭证
        self.api_key = os.getenv("DEEPSEEK_API_KEY")

        self.base_url = os.getenv("DEEPSEEK_BASE_URL")

        if not self.api_key:
            raise ValueError("未找到 API Key，请检查 .env 文件！")
            
        # 初始化客户端
        self.client = OpenAI(
            api_key=self.api_key, 
            base_url=self.base_url
        )

    def generate_cases(self, feature_name):
        print(f"[AI] 正在为 '{feature_name}' 设计测试用例...")
        
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
            print(f"API 调用失败: {e}")
            return None

    def save_to_file(self, data, filename):
        # 自动寻找项目根目录并创建 data/outputs 文件夹
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "data", "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"用例已成功保存至：{filepath}")
        except Exception as e:
            print(f"保存文件失败: {e}")

# ---------------- 执行入口 ----------------
if __name__ == "__main__":
    # 1. 实例化生成器
    generator = CaseGenerator()
    
    # 2. 你的真实业务任务列表（批量循环）
    todo_list = [
        "售后接入抖音组件",
        "增加观演人退票次数限制",
        "用户订单列表翻页查询"
    ]
    
    print(f"开始批量生成测试用例，共 {len(todo_list)} 个功能点...\n")
    print("="*50)
    
    # 3. 循环执行
    for feature in todo_list:
        # 生成用例
        result_data = generator.generate_cases(feature)
        
        if result_data:
            # 动态拼接文件名并保存 (例如: cases_售后接入抖音组件.json)
            custom_filename = f"cases_{feature}.json"
            generator.save_to_file(result_data, filename=custom_filename)
        else:
            print(f"功能点 '{feature}' 的用例生成失败，已跳过。")
            
        print("-" * 50)

    print("\n所有功能点处理完成！")