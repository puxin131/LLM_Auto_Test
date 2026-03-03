import pandas as pd
import os

#获取跟目录路径，构建文件路径
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#拼接测试功能点csv文件路径
file_path = os.path.join(base_dir, 'data','input', 'requirements.csv')

print(f"正在读取文件: {file_path}\n")

try:
    #使用pandas读取csv文件
    df = pd.read_csv(file_path, encoding='utf-8')
    print("成功读取表格数据：\n")
    print(df)
    
    print("\n" + "="*40)
    print(f"表格一共包含 {len(df)} 行有效数据。")

    #遍历表格每一行
    for index,row in df.iterrows():
        req_id = row['功能ID']
        feature = row['需求描述']
        priority = row['优先级']
        print(f" 第 {index + 1} 行提取成功 -> 【{priority}】{req_id}: {feature}")

except Exception as e:
    print(f"读取失败：{e}")
