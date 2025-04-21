import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# 全局变量
OUTPUT_DIR = 'visualizations'
PROCESSED_DATA_DIR = 'processed_data'
ANALYSIS_RESULTS_DIR = 'analysis_results'

# ============== 数据分析模块 ==============

def analyze_csv_file(file_path):
    """分析CSV文件并显示基本统计信息，只生成文本报告"""
    print(f"\n正在分析文件: {file_path}")
    
    # 尝试读取文件，显示加载进度
    print("正在加载数据，这可能需要一些时间...")
    df = pd.read_csv(file_path)
    
    # 显示基本信息
    print(f"\n数据集形状: {df.shape[0]} 行 x {df.shape[1]} 列")
    print("\n列名:")
    for col in df.columns:
        print(f"- {col}")
    
    # 显示前5行数据
    print("\n数据前5行:")
    print(df.head())
    
    # 数据类型和缺失值情况
    print("\n数据类型和缺失值情况:")
    dtypes_missing = pd.DataFrame({
        '数据类型': df.dtypes,
        '非空值数量': df.count(),
        '缺失值数量': df.isnull().sum(),
        '缺失值百分比': (df.isnull().sum() / len(df) * 100).round(2)
    })
    print(dtypes_missing)
    
    # 数值列的基本统计信息
    print("\n数值列的基本统计信息:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe().T)
    else:
        print("没有数值型列")
    
    # 文本列的基本统计信息
    text_cols = df.select_dtypes(include=['object']).columns
    if len(text_cols) > 0:
        print("\n文本列的基本统计信息:")
        for col in text_cols:
            unique_values = df[col].nunique()
            most_common = df[col].value_counts().head(5)
            print(f"\n列 '{col}':")
            print(f"  - 唯一值数量: {unique_values}")
            print(f"  - 前5个最常见值:")
            for val, count in most_common.items():
                print(f"    * {val}: {count} 次 ({count/len(df):.2%})")
    else:
        print("\n没有文本列")
    
    # 创建结果目录
    if not os.path.exists(ANALYSIS_RESULTS_DIR):
        os.makedirs(ANALYSIS_RESULTS_DIR)
    
    # 文件名（不带扩展名）
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    
    # 保存摘要信息到文本文件
    with open(f"{ANALYSIS_RESULTS_DIR}/{base_filename}_summary.txt", "w") as f:
        f.write(f"数据分析报告 - {file_path}\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"数据集形状: {df.shape[0]} 行 x {df.shape[1]} 列\n\n")
        f.write("列名:\n")
        for col in df.columns:
            f.write(f"- {col}\n")
        
        f.write("\n数据类型和缺失值情况:\n")
        f.write(dtypes_missing.to_string())
        
        if len(numeric_cols) > 0:
            f.write("\n\n数值列的基本统计信息:\n")
            f.write(df[numeric_cols].describe().T.to_string())
        
        if len(text_cols) > 0:
            f.write("\n\n文本列的基本统计信息:\n")
            for col in text_cols:
                unique_values = df[col].nunique()
                most_common = df[col].value_counts().head(5)
                f.write(f"\n列 '{col}':\n")
                f.write(f"  - 唯一值数量: {unique_values}\n")
                f.write(f"  - 前5个最常见值:\n")
                for val, count in most_common.items():
                    f.write(f"    * {val}: {count} 次 ({count/len(df):.2%})\n")
    
    print(f"\n摘要信息已保存到 {ANALYSIS_RESULTS_DIR}/{base_filename}_summary.txt")
    print("\n分析完成!")
    
    return df

# ============== 数据处理模块 ==============

def load_and_process_data(file_path='Results_21Mar2022.csv'):
    """
    加载并处理饮食与环境影响数据
    """
    print(f"正在加载数据: {file_path}")
    df = pd.read_csv(file_path)
    
    # 检查数据结构
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 提取主要环境影响指标列
    env_impact_cols = [col for col in df.columns if col.startswith('mean_') and not col.startswith('mean_ghgs_')]
    env_impact_cols += ['mean_ghgs_ch4', 'mean_ghgs_n2o']  # 添加特定的温室气体指标
    
    # 创建聚合的数据集，按饮食类型、性别和年龄组分组
    agg_data = {}
    
    # 1. 按饮食类型聚合
    diet_agg = df.groupby('diet_group')[env_impact_cols].mean().reset_index()
    agg_data['diet'] = diet_agg
    
    # 2. 按饮食类型和性别聚合
    diet_sex_agg = df.groupby(['diet_group', 'sex'])[env_impact_cols].mean().reset_index()
    agg_data['diet_sex'] = diet_sex_agg
    
    # 3. 按饮食类型和年龄组聚合
    diet_age_agg = df.groupby(['diet_group', 'age_group'])[env_impact_cols].mean().reset_index()
    agg_data['diet_age'] = diet_age_agg
    
    # 4. 计算饮食类型的环境影响总排名
    # 对每个环境指标进行归一化处理，使得所有指标在相同的比例下
    norm_data = diet_agg.copy()
    for col in env_impact_cols:
        max_val = norm_data[col].max()
        min_val = norm_data[col].min()
        norm_data[col] = (norm_data[col] - min_val) / (max_val - min_val)
    
    # 计算总体环境影响得分（越低越好）
    norm_data['total_env_impact'] = norm_data[env_impact_cols].sum(axis=1)
    norm_data = norm_data.sort_values('total_env_impact')
    agg_data['diet_ranked'] = norm_data
    
    # 5. 创建用于雷达图的数据
    # 计算每种饮食类型相对于其他饮食类型的百分比值
    radar_data = diet_agg.copy()
    for col in env_impact_cols:
        max_val = radar_data[col].max()
        radar_data[col] = (radar_data[col] / max_val) * 100
    
    agg_data['radar'] = radar_data
    
    # 6. 计算年龄组和性别对环境影响的交互效应
    interaction_data = df.groupby(['diet_group', 'sex', 'age_group'])[env_impact_cols].mean().reset_index()
    agg_data['interaction'] = interaction_data
    
    # 7. 环境影响指标之间的相关性分析
    correlation = df[env_impact_cols].corr()
    agg_data['correlation'] = correlation
    
    return df, agg_data

def get_env_impact_description():
    """返回环境影响指标的描述"""
    descriptions = {
        'mean_ghgs': '平均温室气体排放量 (kg)',
        'mean_land': '平均农业用地面积 (m²)',
        'mean_watscar': '平均缺水量',
        'mean_eut': '平均富营养化潜能 (gPOe)',
        'mean_ghgs_ch4': '畜牧业管理中甲烷排放的平均温室气体 (kg)',
        'mean_ghgs_n2o': '与化肥使用相关的N₂O排放产生的平均温室气体',
        'mean_bio': '平均生物多样性影响 (每天物种灭绝)',
        'mean_watuse': '平均农业用水量 (m³)',
        'mean_acid': '平均酸化潜力'
    }
    return descriptions

def get_diet_colors():
    """返回饮食类型的配色方案"""
    colors = {
        'vegan': '#2ca02c',       # 绿色
        'vegetarian': '#98df8a',  # 浅绿色
        'pescatarian': '#1f77b4', # 蓝色
        'fish': '#1f77b4',        # 蓝色 (与pescatarian相同)
        'meat50': '#ff7f0e',      # 橙色
        'meat': '#d62728',        # 红色
        'meat100': '#d62728'      # 红色 (与meat相同)
    }
    return colors

def save_processed_data(agg_data, output_dir=PROCESSED_DATA_DIR):
    """保存处理后的数据到CSV文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for key, data in agg_data.items():
        data.to_csv(f"{output_dir}/{key}_data.csv", index=False)
    
    print(f"处理后的数据已保存到 {output_dir} 目录")

# ============== 可视化模块 ==============

def create_alluvial_diagram(diet_data, output_file='diet_environmental_impact_alluvial.html'):
    """创建冲积图(桑基图)展示饮食类型与多个环境影响指标的关系"""
    # 获取环境影响指标的描述
    impact_descriptions = get_env_impact_description()
    diet_colors = get_diet_colors()
    
    # 选择主要的环境影响指标
    selected_impacts = ['mean_ghgs', 'mean_land', 'mean_watuse', 'mean_bio']
    selected_labels = [impact_descriptions[col] for col in selected_impacts]
    
    # 桑基图需要源-目标-值的格式
    sankey_data = []
    
    # 添加饮食类型到环境影响指标的连接
    for impact_col, impact_label in zip(selected_impacts, selected_labels):
        # 对当前指标按值排序
        ranked = diet_data.sort_values(impact_col)
        rank_mapper = {diet: i+1 for i, diet in enumerate(ranked['diet_group'])}
        
        for _, row in diet_data.iterrows():
            diet = row['diet_group']
            impact_value = row[impact_col]
            
            # 标准化值以便更好地展示
            normalized_value = (impact_value - diet_data[impact_col].min()) / (diet_data[impact_col].max() - diet_data[impact_col].min())
            value_weight = 0.3 + 0.7 * normalized_value  # 避免太小的值
            
            sankey_data.append({
                'source': diet,
                'target': impact_label,
                'value': value_weight,
                'rank': rank_mapper[diet]
            })
    
    # 创建源和目标的节点列表
    all_nodes = list(diet_data['diet_group'].unique()) + selected_labels
    node_indices = {node: i for i, node in enumerate(all_nodes)}
    
    # 创建链接数据
    links = []
    for item in sankey_data:
        links.append({
            'source': node_indices[item['source']],
            'target': node_indices[item['target']],
            'value': item['value'] * 5,  # 乘以5以增强可视效果
            'customdata': [item['rank']]
        })
    
    # 创建节点数据
    nodes = []
    for node in all_nodes:
        if node in diet_colors:
            color = diet_colors[node]
        else:
            color = 'lightgray'
        
        nodes.append({
            'label': node,
            'color': color
        })
    
    # 创建桑基图
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[node['label'] for node in nodes],
            color=[node['color'] for node in nodes]
        ),
        link=dict(
            source=[link['source'] for link in links],
            target=[link['target'] for link in links],
            value=[link['value'] for link in links],
            customdata=[link['customdata'] for link in links],
            hovertemplate='从 %{source.label} 到 %{target.label}<br>'
                          '影响程度: %{value:.2f}<br>'
                          '排名: %{customdata[0]}<extra></extra>'
        )
    )])
    
    # 设置布局
    fig.update_layout(
        title_text="饮食类型与环境影响的关系流图",
        font_size=12,
        height=600,
        width=900
    )
    
    # 保存为HTML文件，以支持交互
    fig.write_html(f"{OUTPUT_DIR}/{output_file}")
    
    print(f"冲积图已保存到 {OUTPUT_DIR}/{output_file}")
    return fig

# ============== 主函数 ==============

def analyze_all_csv_files():
    """分析当前目录下的所有CSV文件"""
    # 获取当前目录下的所有CSV文件
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("当前目录下没有找到CSV文件。")
        return None
    
    print(f"找到以下CSV文件: {', '.join(csv_files)}")
    print(f"默认分析所有 {len(csv_files)} 个CSV文件...")
    
    # 分析所有CSV文件
    for file in csv_files:
        analyze_csv_file(file)
    
    # 返回第一个CSV文件的路径
    return csv_files[0]

def run_analysis():
    """运行完整的数据分析、处理和可视化流程"""
    # 创建输出目录
    for directory in [ANALYSIS_RESULTS_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    try:
        # 首先分析数据
        print("\n===== 步骤1：基础数据分析 =====")
        first_csv = analyze_all_csv_files()
        
        # 然后处理数据
        print("\n===== 步骤2：高级数据处理 =====")
        if first_csv:
            # 尝试读取已处理的数据
            try:
                print("尝试读取已处理的数据...")
                diet_data = pd.read_csv(f"{PROCESSED_DATA_DIR}/diet_data.csv")
                print("成功读取处理后的数据")
            except FileNotFoundError:
                # 如果找不到处理后的数据，则重新处理
                print("找不到处理后的数据，开始处理...")
                _, processed_data = load_and_process_data(first_csv)
                save_processed_data(processed_data)
                
                # 读取处理后的数据
                diet_data = pd.read_csv(f"{PROCESSED_DATA_DIR}/diet_data.csv")
            
            # 最后创建可视化
            print("\n===== 步骤3：数据可视化 =====")
            print("创建冲积图(桑基图)...")
            create_alluvial_diagram(diet_data)
            
            print("\n所有操作已完成!")
            print(f"- 分析结果保存在: {ANALYSIS_RESULTS_DIR}")
            print(f"- 处理后的数据保存在: {PROCESSED_DATA_DIR}")
            print(f"- 可视化结果保存在: {OUTPUT_DIR}")
    
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_analysis()
    print("\n完成！请查看生成的结果。") 