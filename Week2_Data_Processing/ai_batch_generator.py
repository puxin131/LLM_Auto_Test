import os
import sys
import pandas as pd

#把项目根目录加入系统路径
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from Week1_Basic_API.Case_Generator_v1 import CaseGenerator

csv_path = os.path.join(base_dir,"data","input","requirements.csv")
print(f"正在加载需求文件{csv_path}")

try:
    df = pd.read_csv(csv_path,encoding="utf-8")
    print(f"成功读取表格，共发现{len(df)}条测试需求\n")

except Exception as e:
    print(f"csv读取失败")
    sys.exit(1) # 如果连表格都读不出来，直接停止程序

#实例化
ai_bot = CaseGenerator()
print("================= 开始启动 AI 批量生产流水线 =================\n")

for index,row in df.iterrows():
    req_id = row['功能ID']
    feature_desc = row['需求描述']

    print(f" 正在处理第 {index + 1} 条需求：[{req_id}] {feature_desc}")
    
    # 把 Pandas 读出来的数据，喂给 AI 生成用例
    result_data = ai_bot.generate_cases(feature_desc)
    if result_data:
        # 动态拼接文件名，比如：REQ_001_cases.json
        filename = f"{req_id}_cases.json"
        # 调用 Week1 写好的保存方法，把文件存进 data/outputs
        ai_bot.save_to_file(result_data, filename)
    else:
        print(f"需求 [{req_id}] 生成失败，已跳过。")
        
    print("-" * 60)

print("\n全自动化流水线执行完毕！快去检查 data/outputs 文件夹吧！")