import pandas as pd
import time

# 🚨 关键点 1：导入你的 LangGraph 状态机
# 根据你的项目结构，图谱应该是在 workflow.py 中编译的。
# 假设编译好的图变量名叫 app (例如：app = builder.compile())
from workflow import generate_app

print("📦 正在读取 ManimBench 测试集...")
df = pd.read_parquet("hf://datasets/SuienR/ManimBench-v1/manim_sft_dataset_test_v2.parquet")

# 🚨 关键点 2：为了测试，我们先只跑前 5 条！
df_test = df.sample(2)

experiment_results = []
# 我们先对比两种策略：纯运行时盲循环 和 你的双重反馈架构
strategies = ["Runtime Only", "Ours"] 

print(f"🚀 开始执行自动化评测，共 {len(df_test)} 个测试用例...")

for index, row in df_test.iterrows():
    instruction = row['Reviewed Description']
    
    for strategy in strategies:
        print(f"\n========================================")
        print(f"▶️ 正在处理 Task {index} | 策略: {strategy}")
        print(f"========================================")
        start_time = time.time()
        
        # 构造喂给你的系统的初始状态
        initial_state = {
            "task": instruction,
            "code": "",
            "retry_count": 0,
            "ast_error_ratio": 0.0, # 初始值
            "vlm_iou_score": 0.0,   # 初始值
            "is_success": 0,
            "strategy": strategy    # 传入策略标记
        }
        
        # 加上重试机制，防止偶发网络断开
        max_api_retries = 3
        api_retry_count = 0
        final_state = None
        
        while api_retry_count < max_api_retries:
            try:
                # 🚀 核心：触发你的 LangGraph 系统
                final_state = generate_app.invoke(initial_state)
                break # 如果成功跑通，就跳出 while 循环
                
            except Exception as e:
                api_retry_count += 1
                print(f"⚠️ 遇到报错: {e}")
                if api_retry_count < max_api_retries:
                    print(f"🔄 正在尝试重新连接 (第 {api_retry_count}/{max_api_retries} 次)... 休息 10 秒")
                    time.sleep(10) # 休息一下再试
                else:
                    print(f"❌ 任务 {index} 彻底执行崩溃，已跳过。")
                    
        # 提取数据的逻辑不变
        if final_state:
            iteration_count = final_state.get("retry_count", 5)
            
            if iteration_count < 5:
                is_success = 1
            else:
                is_success = 0
                
            ast_error_ratio = final_state.get("ast_error_ratio", 0.5) 
            vlm_iou_score = final_state.get("vlm_iou_score", 0.2)
        else:
            # 如果重试3次都失败了，就记作全部失败
            is_success = 0
            iteration_count = 5
            ast_error_ratio = 1.0
            vlm_iou_score = 1.0
            
        time_cost = time.time() - start_time
        
        # 将结果存入列表
        experiment_results.append({
            "task_id": index,
            "strategy": strategy,
            "iteration_count": iteration_count,
            "is_success": is_success,
            "ast_error_ratio": ast_error_ratio,
            "vlm_iou_score": vlm_iou_score,
            "time_cost": time_cost
        })
        
        print(f"✅ 完成！耗时 {time_cost:.1f}s | 成功: {is_success} | 迭代: {iteration_count}")

# 导出为 CSV
results_df = pd.DataFrame(experiment_results)
results_df.to_csv("experiment_results_mini.csv", index=False)
print("\n🎉 小批量跑批完成！已生成 experiment_results_mini.csv")
