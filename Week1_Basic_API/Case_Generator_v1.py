import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

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

        # 获取此时此刻的真实时间（精确到分钟）
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        prompt = f"""
        你是一名拥有 10 年经验的资深测试架构师。请为功能点“{feature_name}”设计 3-5 条核心测试用例。

        【全局时间基准】（极其重要！）：
        当前的真实系统时间是：{current_time}。
        你在设计用例的前置条件和测试步骤时，所有涉及时间的数据（如支付时间、退票截止时间等），必须以该系统时间为基准，合理推算具体的未来时间或临界时间！严禁使用 2024 等脱离现实的历史数据！
        
        【测试设计指导原则】：
        1. 核心基石优先 (P0)：优先验证支撑该功能的最底层逻辑（如核心算法、数据状态流转、时间/时区计算）。
        2. 依赖与前置优先 (P0/P1)：如果业务依赖底层配置（如后台开关、次数限制），必须先验证配置生效。
        3. 异常与健壮性挖掘 (P1/P2)：包含防资损、边界值及第三方依赖异常等负向拦截场景。
        
        【场景化与具体化要求】（核心重点！你必须严格遵守！）：
        - 拒绝假大空抽象描述，必须基于具体场景！
        - 必须包含前置条件：明确指出需要提前准备好的具体数据状态（例如：不要写“有一个有效订单”，要写“准备一个状态为'已支付'、实付金额为 99.00 的东七区项目订单”）。
        - 具象化输入参数：步骤中必须体现具体的输入值或操作时间（例如：不要写“在截止时间后退票”，要写“在系统当前时间为 2024-01-01 12:05（超过截止时间5分钟）时点击退票”）。
        - 多维度预期结果：预期结果必须包含明确的 UI 提示语、数据库预期状态变更，或具体的拦截逻辑。

        【必须返回纯 JSON 格式，严格参考以下结构】：
        {{
            "feature": "{feature_name}",
            "test_cases": [
                {{
                    "id": "TC_001",
                    "title": "[场景化标题] 示例：东八区用户在退票截止前1分钟发起退票，验证时区换算与申请成功",
                    "priority": "P0",
                    "precondition": "前置条件：示例：1. 创建东七区项目A，设置退票截止时间为东七区 2024-01-01 10:00；2. 准备东八区普通用户账号B，拥有一笔项目A的已支付订单",
                    "steps": "1. [动作与具体数据]...; 2. [动作与具体数据]...",
                    "expected_result": "1. [UI表现]...; 2. [数据流转]..."
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