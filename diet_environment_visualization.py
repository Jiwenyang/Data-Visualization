import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Global variables
OUTPUT_DIR = 'visualizations'
PROCESSED_DATA_DIR = 'processed_data'
ANALYSIS_RESULTS_DIR = 'analysis_results'

# ============== Data Analysis Module ==============

def analyze_csv_file(file_path):
    """Analyze CSV file and display basic statistics, only generate text report"""
    print(f"\nAnalyzing file: {file_path}")
    
    # Try to read file, show loading progress
    print("Loading data, this may take some time...")
    df = pd.read_csv(file_path)
    
    # Show basic info
    print(f"\nDataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumn names:")
    for col in df.columns:
        print(f"- {col}")
    
    # Show first 5 rows
    print("\nFirst 5 rows of data:")
    print(df.head())
    
    # Data types and missing values
    print("\nData types and missing values:")
    dtypes_missing = pd.DataFrame({
        'Data Type': df.dtypes,
        'Non-null Count': df.count(),
        'Missing Count': df.isnull().sum(),
        'Missing Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    print(dtypes_missing)
    
    # Basic stats for numerical columns
    print("\nBasic statistical information for numerical columns:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe().T)
    else:
        print("No numerical columns")
    
    # Basic stats for text columns
    text_cols = df.select_dtypes(include=['object']).columns
    if len(text_cols) > 0:
        print("\nBasic statistical information for text columns:")
        for col in text_cols:
            unique_values = df[col].nunique()
            most_common = df[col].value_counts().head(5)
            print(f"\nColumn '{col}':")
            print(f"  - Unique values: {unique_values}")
            print(f"  - Top 5 most common values:")
            for val, count in most_common.items():
                print(f"    * {val}: {count} times ({count/len(df):.2%})")
    else:
        print("\nNo text columns")
    
    # Create results directory
    if not os.path.exists(ANALYSIS_RESULTS_DIR):
        os.makedirs(ANALYSIS_RESULTS_DIR)
    
    # Filename (without extension)
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    
    # Save summary to text file
    with open(f"{ANALYSIS_RESULTS_DIR}/{base_filename}_summary.txt", "w") as f:
        f.write(f"Data Analysis Report - {file_path}\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n")
        f.write("Columns:\n")
        for col in df.columns:
            f.write(f"- {col}\n")
        
        f.write("\nData types and missing values:\n")
        f.write(dtypes_missing.to_string())
        
        if len(numeric_cols) > 0:
            f.write("\n\nBasic statistical information for numerical columns:\n")
            f.write(df[numeric_cols].describe().T.to_string())
        
        if len(text_cols) > 0:
            f.write("\n\nBasic statistical information for text columns:\n")
            for col in text_cols:
                unique_values = df[col].nunique()
                most_common = df[col].value_counts().head(5)
                f.write(f"\nColumn '{col}':\n")
                f.write(f"  - Unique values: {unique_values}\n")
                f.write(f"  - Top 5 most common values:\n")
                for val, count in most_common.items():
                    f.write(f"    * {val}: {count} times ({count/len(df):.2%})\n")
    
    print(f"\nSummary saved to {ANALYSIS_RESULTS_DIR}/{base_filename}_summary.txt")
    print("\nAnalysis complete!")
    
    return df

# ============== Data Processing Module ==============

def load_and_process_data(file_path='Results_21Mar2022.csv'):
    """
    Load and process diet and environmental impact data
    """
    print(f"Loading data: {file_path}")
    df = pd.read_csv(file_path)
    
    # Check data structure
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Extract main environmental impact indicator columns
    env_impact_cols = [col for col in df.columns if col.startswith('mean_') and not col.startswith('mean_ghgs_')]
    env_impact_cols += ['mean_ghgs_ch4', 'mean_ghgs_n2o']  # Add specific greenhouse gas indicators
    
    # Create aggregated dataset grouped by diet type, gender, and age group
    agg_data = {}
    
    # 1. Aggregate by diet type
    diet_agg = df.groupby('diet_group')[env_impact_cols].mean().reset_index()
    agg_data['diet'] = diet_agg
    
    # 2. Aggregate by diet type and gender
    diet_sex_agg = df.groupby(['diet_group', 'sex'])[env_impact_cols].mean().reset_index()
    agg_data['diet_sex'] = diet_sex_agg
    
    # 3. Aggregate by diet type and age group
    diet_age_agg = df.groupby(['diet_group', 'age_group'])[env_impact_cols].mean().reset_index()
    agg_data['diet_age'] = diet_age_agg
    
    # 4. Calculate overall environmental impact ranking for diet types
    # Normalize all indicators to same scale
    norm_data = diet_agg.copy()
    for col in env_impact_cols:
        max_val = norm_data[col].max()
        min_val = norm_data[col].min()
        norm_data[col] = (norm_data[col] - min_val) / (max_val - min_val)
    
    # Calculate total environmental impact score (lower is better)
    norm_data['total_env_impact'] = norm_data[env_impact_cols].sum(axis=1)
    norm_data = norm_data.sort_values('total_env_impact')
    agg_data['diet_ranked'] = norm_data
    
    # 5. Create data for radar chart
    # Calculate percentage values relative to other diet types
    radar_data = diet_agg.copy()
    for col in env_impact_cols:
        max_val = radar_data[col].max()
        radar_data[col] = (radar_data[col] / max_val) * 100
    
    agg_data['radar'] = radar_data
    
    # 6. Calculate interaction effects between age groups and gender
    interaction_data = df.groupby(['diet_group', 'sex', 'age_group'])[env_impact_cols].mean().reset_index()
    agg_data['interaction'] = interaction_data
    
    # 7. Correlation analysis between environmental impact indicators
    correlation = df[env_impact_cols].corr()
    agg_data['correlation'] = correlation
    
    return df, agg_data

def get_env_impact_description():
    """Return descriptions for environmental impact indicators"""
    descriptions = {
        'mean_ghgs': 'Average greenhouse gas emissions (kg)',
        'mean_land': 'Average agricultural land use (m²)',
        'mean_watscar': 'Average water scarcity',
        'mean_eut': 'Average eutrophication potential (gPOe)',
        'mean_ghgs_ch4': 'Average methane emissions from livestock management (kg)',
        'mean_ghgs_n2o': 'Average N₂O emissions from fertilizer use (kg)',
        'mean_bio': 'Average biodiversity impact (daily species extinction)',
        'mean_watuse': 'Average agricultural water use (m³)',
        'mean_acid': 'Average acidification potential'
    }
    return descriptions

def get_diet_colors():
    """Return color scheme for diet types"""
    colors = {
        'vegan': '#2ca02c',       # Green
        'vegetarian': '#98df8a',  # Light green
        'pescatarian': '#1f77b4', # Blue
        'fish': '#1f77b4',        # Blue (same as pescatarian)
        'meat50': '#ff7f0e',      # Orange
        'meat': '#d62728',        # Red
        'meat100': '#d62728'      # Red (same as meat)
    }
    return colors

def save_processed_data(agg_data, output_dir=PROCESSED_DATA_DIR):
    """Save processed data to CSV files"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for key, data in agg_data.items():
        data.to_csv(f"{output_dir}/{key}_data.csv", index=False)
    
    print(f"Processed data saved to {output_dir} directory")

# ============== Visualization Module ==============

def create_alluvial_diagram(diet_data, output_file='diet_environmental_impact_alluvial.html'):
    """Create alluvial diagram (Sankey diagram) showing relationships between diet types and environmental impacts"""
    # Get descriptions for environmental impact indicators
    impact_descriptions = get_env_impact_description()
    diet_colors = get_diet_colors()
    
    # Select main environmental impact indicators
    selected_impacts = ['mean_ghgs', 'mean_land', 'mean_watuse', 'mean_bio']
    selected_labels = [impact_descriptions[col] for col in selected_impacts]
    
    # Sankey diagram requires source-target-value format
    sankey_data = []
    
    # Add connections between diet types and environmental impacts
    for impact_col, impact_label in zip(selected_impacts, selected_labels):
        # Sort by current indicator value
        ranked = diet_data.sort_values(impact_col)
        rank_mapper = {diet: i+1 for i, diet in enumerate(ranked['diet_group'])}
        
        for _, row in diet_data.iterrows():
            diet = row['diet_group']
            impact_value = row[impact_col]
            
            # Normalize values for better visualization
            normalized_value = (impact_value - diet_data[impact_col].min()) / (diet_data[impact_col].max() - diet_data[impact_col].min())
            value_weight = 0.3 + 0.7 * normalized_value  # Prevent too small values
            
            sankey_data.append({
                'source': diet,
                'target': impact_label,
                'value': value_weight,
                'rank': rank_mapper[diet]
            })
    
    # Create node and target lists
    all_nodes = list(diet_data['diet_group'].unique()) + selected_labels
    node_indices = {node: i for i, node in enumerate(all_nodes)}
    
    # Create link data
    links = []
    for item in sankey_data:
        links.append({
            'source': node_indices[item['source']],
            'target': node_indices[item['target']],
            'value': item['value'] * 5,  # Scale values for better visibility
            'customdata': [item['rank']]
        })
    
    # Create node data
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
    
    # Create Sankey diagram
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
            hovertemplate='From %{source.label}<br>'
                          'To %{target.label}<br>'
                          'Impact Value: %{value:.2f}<br>'
                          'Rank: %{customdata[0]}<extra></extra>'
        )
    )])
    
    # Set layout
    fig.update_layout(
        title_text="Diet-Environmental Impact Relationship Flowchart",
        font_size=12,
        height=600,
        width=900
    )
    
    # Save as HTML file for interactivity
    fig.write_html(f"{OUTPUT_DIR}/{output_file}")
    
    print(f"Sankey diagram saved to {OUTPUT_DIR}/{output_file}")
    return fig

# ============== Main Function ==============

def analyze_all_csv_files():
    """Analyze all CSV files in current directory"""
    # Get all CSV files in current directory
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in current directory.")
        return None
    
    print(f"Found CSV files: {', '.join(csv_files)}")
    print(f"Defaulting to analyze all {len(csv_files)} files...")
    
    # Analyze all CSV files
    for file in csv_files:
        analyze_csv_file(file)
    
    # Return path to first CSV file
    return csv_files[0]

def run_analysis():
    """Run complete data analysis, processing, and visualization workflow"""
    # Create output directories
    for directory in [ANALYSIS_RESULTS_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    try:
        # Step 1: Basic data analysis
        print("\n===== Step 1: Basic Data Analysis =====")
        first_csv = analyze_all_csv_files()
        
        # Step 2: Advanced data processing
        print("\n===== Step 2: Advanced Data Processing =====")
        if first_csv:
            # Try to read existing processed data
            try:
                print("Attempting to read processed data...")
                diet_data = pd.read_csv(f"{PROCESSED_DATA_DIR}/diet_data.csv")
                print("Successfully loaded processed data")
            except FileNotFoundError:
                # Process data if not found
                print("Processed data not found. Starting processing...")
                _, processed_data = load_and_process_data(first_csv)
                save_processed_data(processed_data)
                
                # Load processed data
                diet_data = pd.read_csv(f"{PROCESSED_DATA_DIR}/diet_data.csv")
            
            # Step 3: Create visualizations
            print("\n===== Step 3: Data Visualization =====")
            print("Creating Sankey diagram...")
            create_alluvial_diagram(diet_data)
            
            print("\nAll operations completed!")
            print(f"- Analysis results saved to: {ANALYSIS_RESULTS_DIR}")
            print(f"- Processed data saved to: {PROCESSED_DATA_DIR}")
            print(f"- Visualizations saved to: {OUTPUT_DIR}")
    
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_analysis()
    print("\nComplete! Please check the generated results.")
