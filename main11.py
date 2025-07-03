import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import shutil
import cv2
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_mermaid import st_mermaid

st.set_page_config(
    page_title="Wildlife Detection Pipeline",
    page_icon="🦓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #4682B4;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🦓 Wildlife Detection Pipeline</h1>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Home", "Dataset Organization", "Model Training", "Model Comparison", "Results Analysis"]
)

# Initialize session state
if 'dataset_organized' not in st.session_state:
    st.session_state.dataset_organized = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None

def home_page():
    st.markdown("## Welcome to the Wildlife Detection Pipeline!")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Features")
        st.markdown("""
        - **Dataset Organization**: Convert your wildlife datasets to YOLO format
        - **Model Training**: Train multiple object detection models
        - **Model Comparison**: Compare YOLOv5, YOLOv8, and R-CNN performance
        - **SMOTE Integration**: Handle class imbalance with SMOTE oversampling
        - **Interactive Analysis**: Visualize results and metrics
        """)

    with col2:
        st.markdown("### 🦎 Supported Animals")
        animals = ['Sheep', 'Cattle', 'Seal', 'Camelus', 'Kiang', 'Zebra']
        for animal in animals:
            st.markdown(f"- {animal}")

    st.markdown("### 📊 Pipeline Overview")
    # Changed from st.mermaid to st_mermaid
    st_mermaid("""
    graph TD
        A[Upload Dataset] --> B[Dataset Organization]
        B --> C[Data Analysis]
        C --> D[Model Training]
        D --> E[Model Comparison]
        E --> F[Results Visualization]
    """)

def dataset_organization_page():
    st.markdown('<h2 class="section-header">📁 Dataset Organization</h2>', unsafe_allow_html=True)

    # Path configuration
    st.markdown("### Configuration")
    col1, col2, col3 = st.columns(3)

    with col1:
        base_images_dir = st.text_input(
            "Base Images Directory",
            value="/path/to/your/local/directory/WAID/images",
            help="Directory containing train/test/valid image folders"
        )

    with col2:
        base_labels_dir = st.text_input(
            "Base Labels Directory",
            value="/path/to/your/local/directory/WAID/labels",
            help="Directory containing train/test/valid label folders"
        )

    with col3:
        output_dir = st.text_input(
            "Output Directory",
            value="/path/to/output",
            help="Directory for organized dataset"
        )

    # Dataset organization
    if st.button("🔄 Organize Dataset", type="primary"):
        if all([base_images_dir, base_labels_dir, output_dir]):
            with st.spinner("Organizing dataset..."):
                try:
                    # This would use your WildlifeDatasetOrganizer class
                    # organizer = WildlifeDatasetOrganizer(base_images_dir, base_labels_dir, output_dir)
                    # organizer.organize_dataset()
                    # stats = organizer.analyze_dataset()

                    # Simulated results for demo
                    st.success("Dataset organized successfully!")
                    st.session_state.dataset_organized = True

                    # Display sample statistics
                    sample_stats = {
                        'train': {'total_images': 1200, 'total_labels': 1200},
                        'val': {'total_images': 300, 'total_labels': 300},
                        'test': {'total_images': 400, 'total_labels': 400}
                    }

                    st.markdown("### 📊 Dataset Statistics")
                    for split, stats in sample_stats.items():
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(f"{split.upper()} Images", stats['total_images'])
                        with col2:
                            st.metric(f"{split.upper()} Labels", stats['total_labels'])

                except Exception as e:
                    st.error(f"Error organizing dataset: {str(e)}")
        else:
            st.error("Please fill in all directory paths")

    # Sample images display
    if st.session_state.dataset_organized:
        st.markdown("### 🖼️ Sample Images")
        # This would display actual sample images from your dataset
        st.info("Sample images would be displayed here from your organized dataset")

def model_training_page():
    st.markdown('<h2 class="section-header">🚀 Model Training</h2>', unsafe_allow_html=True)

    if not st.session_state.dataset_organized:
        st.warning("Please organize your dataset first!")
        return

    # Training configuration
    st.markdown("### Training Configuration")
    col1, col2, col3 = st.columns(3)

    with col1:
        batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=1)

    with col2:
        num_workers = st.selectbox("Number of Workers", [2, 4, 8], index=1)

    with col3:
        epochs = st.selectbox("Epochs", [5, 10, 20, 50], index=0)

    # Model selection
    st.markdown("### Model Selection")
    models_to_train = st.multiselect(
        "Select models to train",
        ["YOLOv5", "YOLOv8", "R-CNN"],
        default=["YOLOv5", "YOLOv8"]
    )

    # Training button
    if st.button("🎯 Start Training", type="primary"):
        if models_to_train:
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, model in enumerate(models_to_train):
                status_text.text(f"Training {model}...")

                # Simulate training progress
                for j in range(epochs):
                    progress_bar.progress((i * epochs + j + 1) / (len(models_to_train) * epochs))
                    st.write(f"Epoch {j+1}/{epochs} - {model}")

            st.success("Training completed!")
            st.session_state.models_trained = True
        else:
            st.error("Please select at least one model to train")

def model_comparison_page():
    st.markdown('<h2 class="section-header">⚖️ Model Comparison</h2>', unsafe_allow_html=True)

    if not st.session_state.models_trained:
        st.warning("Please train models first!")
        return

    # Comparison options
    st.markdown("### Comparison Options")
    col1, col2 = st.columns(2)

    with col1:
        use_smote = st.checkbox("Use SMOTE for class imbalance", value=True)

    with col2:
        cross_validation = st.checkbox("Use Cross-Validation", value=True)

    # Run comparison
    if st.button("📊 Compare Models", type="primary"):
        with st.spinner("Comparing models..."):
            # Simulate comparison results
            comparison_data = {
                'Model': ['YOLOv5', 'YOLOv8', 'R-CNN'] * 2,
                'Scenario': ['Without SMOTE'] * 3 + ['With SMOTE'] * 3,
                'Precision': [0.85, 0.88, 0.82, 0.87, 0.90, 0.84],
                'Recall': [0.82, 0.85, 0.79, 0.86, 0.88, 0.83],
                'F1 Score': [0.83, 0.86, 0.80, 0.86, 0.89, 0.83],
                'mAP': [0.81, 0.84, 0.77, 0.85, 0.87, 0.82]
            }

            comparison_df = pd.DataFrame(comparison_data)
            st.session_state.comparison_results = comparison_df

            st.success("Model comparison completed!")

            # Display results table
            st.markdown("### 📈 Results Table")
            st.dataframe(comparison_df, use_container_width=True)

def results_analysis_page():
    st.markdown('<h2 class="section-header">📊 Results Analysis</h2>', unsafe_allow_html=True)

    if st.session_state.comparison_results is None:
        st.warning("Please run model comparison first!")
        return

    df = st.session_state.comparison_results

    # Metrics overview
    st.markdown("### 🎯 Performance Metrics")
    metrics = ['Precision', 'Recall', 'F1 Score', 'mAP']

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Bar Charts", "📈 Line Charts", "🎯 Radar Chart", "📋 Summary"])

    with tab1:
        # Bar charts for each metric
        for metric in metrics:
            fig = px.bar(
                df,
                x='Model',
                y=metric,
                color='Scenario',
                title=f'{metric} Comparison',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Line chart showing all metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metrics,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        for i, metric in enumerate(metrics):
            row = i // 2 + 1
            col = i % 2 + 1

            for scenario in df['Scenario'].unique():
                scenario_data = df[df['Scenario'] == scenario]
                fig.add_trace(
                    go.Scatter(
                        x=scenario_data['Model'],
                        y=scenario_data[metric],
                        name=f'{scenario} - {metric}',
                        mode='lines+markers'
                    ),
                    row=row, col=col
                )

        fig.update_layout(height=600, title_text="Performance Metrics Comparison")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Radar chart for model comparison
        st.markdown("### 🎯 Radar Chart - Model Performance")

        # Select scenario for radar chart
        scenario = st.selectbox("Select Scenario", df['Scenario'].unique())
        scenario_data = df[df['Scenario'] == scenario]

        fig = go.Figure()

        for model in scenario_data['Model'].unique():
            model_data = scenario_data[scenario_data['Model'] == model]

            fig.add_trace(go.Scatterpolar(
                r=[model_data[metric].values[0] for metric in metrics],
                theta=metrics,
                fill='toself',
                name=model
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title=f"Model Performance Radar Chart - {scenario}"
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        # Summary statistics
        st.markdown("### 📋 Summary Statistics")

        # Best performing model per metric
        st.markdown("#### 🏆 Best Performing Models")
        for metric in metrics:
            best_row = df.loc[df[metric].idxmax()]
            st.markdown(f"**{metric}**: {best_row['Model']} ({best_row['Scenario']}) - {best_row[metric]:.3f}")

        # Average performance by scenario
        st.markdown("#### 📊 Average Performance by Scenario")
        avg_by_scenario = df.groupby('Scenario')[metrics].mean()
        st.dataframe(avg_by_scenario)

        # Model ranking
        st.markdown("#### 🥇 Overall Model Ranking")
        df['Overall_Score'] = df[metrics].mean(axis=1)
        ranking = df.nlargest(6, 'Overall_Score')[['Model', 'Scenario', 'Overall_Score']]
        st.dataframe(ranking)

# Page routing
if page == "Home":
    home_page()
elif page == "Dataset Organization":
    dataset_organization_page()
elif page == "Model Training":
    model_training_page()
elif page == "Model Comparison":
    model_comparison_page()
elif page == "Results Analysis":
    results_analysis_page()

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Wildlife Detection Pipeline v1.0")

# Save this as launcher.py
import subprocess
import sys

def run_streamlit():
    subprocess.run([sys.executable, "-m", "streamlit", "run", "wildlife_app.py"])

if __name__ == "__main__":
    run_streamlit()

import subprocess
subprocess.run(["streamlit", "run", "wildlife_app.py"])
