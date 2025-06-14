#!/usr/bin/env python3
"""
Streamlit Dashboard for Fraud Job Detection
Interactive web application for data analysis and predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import pickle
import os
from main import FraudJobDetector
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fraud Job Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class DashboardApp:
    def __init__(self):
        self.detector = FraudJobDetector()
        self.train_data = None
        self.test_data = None
        self.models_loaded = False
        
    def load_session_data(self):
        """Load data into session state"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False
            
    def sidebar_navigation(self):
        """Create sidebar navigation"""
        st.sidebar.title("🔍 Navigation")
        
        pages = {
            "🏠 Home": "home",
            "📊 Data Analysis": "analysis", 
            "🤖 Model Training": "training",
            "🔮 Predictions": "predictions",
            "📈 Performance": "performance"
        }
        
        selected_page = st.sidebar.selectbox(
            "Choose a page:",
            list(pages.keys())
        )
        
        return pages[selected_page]
    
    def load_data_section(self):
        """Data loading section"""
        st.subheader("📂 Data Loading")
        
        col1, col2 = st.columns(2)
        
        with col1:
            train_file = st.file_uploader(
                "Upload Training Data (CSV)", 
                type=['csv'],
                key="train_upload"
            )
            
        with col2:
            test_file = st.file_uploader(
                "Upload Test Data (CSV)", 
                type=['csv'],
                key="test_upload"
            )
        
        if train_file is not None:
            try:
                self.train_data = pd.read_csv(train_file)
                st.success(f"✅ Training data loaded: {self.train_data.shape}")
                st.session_state.data_loaded = True
                st.session_state.train_data = self.train_data
            except Exception as e:
                st.error(f"❌ Error loading training data: {e}")
        
        if test_file is not None:
            try:
                self.test_data = pd.read_csv(test_file)
                st.success(f"✅ Test data loaded: {self.test_data.shape}")
                st.session_state.test_data = self.test_data
            except Exception as e:
                st.error(f"❌ Error loading test data: {e}")
                
        # Try to load from session state
        if 'train_data' in st.session_state:
            self.train_data = st.session_state.train_data
        if 'test_data' in st.session_state:
            self.test_data = st.session_state.test_data
    
    def home_page(self):
        """Home page content"""
        st.markdown('<h1 class="main-header">🔍 Fraud Job Detection Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        ### Welcome to the Fraud Job Detection System
        
        This dashboard provides comprehensive tools for detecting fraudulent job postings using machine learning techniques.
        
        **Features:**
        - 📊 **Data Analysis**: Explore patterns in job postings data
        - 🤖 **Model Training**: Train multiple ML models (Logistic Regression, Naive Bayes, XGBoost)
        - 🔮 **Predictions**: Make real-time fraud predictions
        - 📈 **Performance**: Analyze model performance metrics
        
        **How to use:**
        1. Upload your training and test datasets
        2. Explore the data through visualizations
        3. Train machine learning models
        4. Make predictions on new job postings
        """)
        
        # Load data section
        self.load_data_section()
        
        # Quick stats if data is loaded
        if self.train_data is not None:
            st.subheader("📋 Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Jobs", len(self.train_data))
            
            with col2:
                fraud_count = self.train_data['fraudulent'].sum() if 'fraudulent' in self.train_data.columns else 0
                st.metric("Fraudulent Jobs", fraud_count)
            
            with col3:
                fraud_rate = (fraud_count / len(self.train_data) * 100) if len(self.train_data) > 0 else 0
                st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
            
            with col4:
                st.metric("Features", len(self.train_data.columns))
    
    def analysis_page(self):
        """Data analysis page"""
        st.header("📊 Data Analysis")
        
        if self.train_data is None:
            st.warning("⚠️ Please upload training data first!")
            return
        
        # Preprocess data for analysis
        processed_data = self.detector.preprocess_data(self.train_data.copy())
        
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Fraud Distribution", "Text Analysis", "Geographic Analysis"])
        
        with tab1:
            st.subheader("Dataset Overview")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(processed_data.head())
            
            with col2:
                st.subheader("Dataset Info")
                buffer = []
                buffer.append(f"**Shape:** {processed_data.shape}")
                buffer.append(f"**Columns:** {len(processed_data.columns)}")
                buffer.append(f"**Missing Values:** {processed_data.isnull().sum().sum()}")
                st.markdown("<br>".join(buffer), unsafe_allow_html=True)
        
        with tab2:
            st.subheader("Fraud vs Legitimate Jobs")
            
            if 'fraudulent' in processed_data.columns:
                fraud_counts = processed_data['fraudulent'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    fig = px.pie(
                        values=fraud_counts.values,
                        names=['Legitimate', 'Fraudulent'],
                        title="Distribution of Job Types",
                        color_discrete_sequence=['#2E8B57', '#DC143C']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Bar chart
                    fig = px.bar(
                        x=['Legitimate', 'Fraudulent'],
                        y=fraud_counts.values,
                        title="Job Count by Type",
                        color=['Legitimate', 'Fraudulent'],
                        color_discrete_sequence=['#2E8B57', '#DC143C']
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Text Analysis")
            
            if 'text' in processed_data.columns:
                # Word clouds
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Legitimate Jobs Word Cloud**")
                    if 'fraudulent' in processed_data.columns:
                        legit_text = ' '.join(processed_data[processed_data['fraudulent'] == 0]['text'].astype(str))
                    else:
                        legit_text = ' '.join(processed_data['text'].astype(str))
                    
                    if legit_text.strip():
                        wordcloud = WordCloud(
                            width=400, height=300, 
                            background_color='white',
                            colormap='Greens'
                        ).generate(legit_text)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                
                with col2:
                    st.write("**Fraudulent Jobs Word Cloud**")
                    if 'fraudulent' in processed_data.columns and processed_data['fraudulent'].sum() > 0:
                        fraud_text = ' '.join(processed_data[processed_data['fraudulent'] == 1]['text'].astype(str))
                        
                        if fraud_text.strip():
                            wordcloud = WordCloud(
                                width=400, height=300,
                                background_color='white',
                                colormap='Reds'
                            ).generate(fraud_text)
                            
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig)
                    else:
                        st.info("No fraudulent jobs found for word cloud generation")
        
        with tab4:
            st.subheader("Geographic Analysis")
            
            if 'country' in processed_data.columns:
                # Top countries
                country_counts = processed_data['country'].value_counts().head(10)
                
                if len(country_counts) > 0:
                    fig = px.bar(
                        x=country_counts.index,
                        y=country_counts.values,
                        title="Top 10 Countries by Job Postings",
                        labels={'x': 'Country', 'y': 'Number of Jobs'}
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Fraud by country
                    if 'fraudulent' in processed_data.columns:
                        fraud_by_country = processed_data.groupby('country')['fraudulent'].agg(['count', 'sum']).reset_index()
                        fraud_by_country['fraud_rate'] = (fraud_by_country['sum'] / fraud_by_country['count'] * 100).round(2)
                        fraud_by_country = fraud_by_country.sort_values('count', ascending=False).head(10)
                        
                        fig = px.scatter(
                            fraud_by_country,
                            x='count',
                            y='fraud_rate',
                            size='sum',
                            hover_data=['country'],
                            title="Fraud Rate vs Job Count by Country",
                            labels={'count': 'Total Jobs', 'fraud_rate': 'Fraud Rate (%)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    def training_page(self):
        """Model training page"""
        st.header("🤖 Model Training")
        
        if self.train_data is None:
            st.warning("⚠️ Please upload training data first!")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Training Configuration")
            
            # Training parameters
            balance_method = st.selectbox(
                "Data Balancing Method:",
                ["undersample", "oversample (SMOTE)"]
            )
            
            test_size = st.slider(
                "Validation Split Size:",
                min_value=0.1,
                max_value=0.5,
                value=0.3,
                step=0.05
            )
            
            selected_models = st.multiselect(
                "Select Models to Train:",
                ["Logistic Regression", "Naive Bayes", "XGBoost"],
                default=["Logistic Regression", "XGBoost"]
            )
        
        with col2:
            st.subheader("Training Status")
            if st.session_state.get('models_trained', False):
                st.success("✅ Models trained successfully!")
            else:
                st.info("🔄 Ready to train models")
        
        # Training button
        if st.button("🚀 Start Training", type="primary"):
            if not selected_models:
                st.error("❌ Please select at least one model to train!")
                return
            
            with st.spinner("Training models... This may take a few minutes."):
                try:
                    # Preprocess data
                    processed_data = self.detector.preprocess_data(self.train_data.copy())
                    
                    # Balance data
                    if balance_method == "undersample":
                        balanced_data = self.detector.balance_data(processed_data, method='undersample')
                    else:
                        balanced_data = processed_data
                    
                    # Split data
                    from sklearn.model_selection import train_test_split
                    X_train, X_val, y_train, y_val = train_test_split(
                        balanced_data['text'],
                        balanced_data['fraudulent'],
                        test_size=test_size,
                        stratify=balanced_data['fraudulent'],
                        random_state=42
                    )
                    
                    # Prepare features
                    X_train_vec, _ = self.detector.prepare_features(
                        pd.DataFrame({'text': X_train})
                    )
                    X_val_vec = self.detector.vectorizer.transform(X_val)
                    
                    # Train models
                    results = self.detector.train_models(X_train_vec, y_train, X_val_vec, y_val)
                    
                    # Filter results by selected models
                    filtered_results = {k: v for k, v in results.items() if k in selected_models}
                    
                    # Store results in session state
                    st.session_state.training_results = filtered_results
                    st.session_state.models_trained = True
                    
                    # Save models
                    self.detector.save_models()
                    
                    st.success("✅ Training completed successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
        
        # Display results if available
        if st.session_state.get('models_trained', False) and 'training_results' in st.session_state:
            st.subheader("📊 Training Results")
            
            results = st.session_state.training_results
            
            # Performance metrics table
            metrics_data = []
            for model_name, result in results.items():
                cr = result['classification_report']
                metrics_data.append({
                    'Model': model_name,
                    'F1-Score': f"{result['f1_score']:.4f}",
                    'Precision': f"{cr['weighted avg']['precision']:.4f}",
                    'Recall': f"{cr['weighted avg']['recall']:.4f}",
                    'Accuracy': f"{cr['accuracy']:.4f}"
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Confusion matrices
            st.subheader("🔍 Confusion Matrices")
            
            cols = st.columns(len(results))
            for idx, (model_name, result) in enumerate(results.items()):
                with cols[idx]:
                    cm = result['confusion_matrix']
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_title(f'{model_name}')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)
    
    def predictions_page(self):
        """Predictions page"""
        st.header("🔮 Make Predictions")
        
        # Try to load deployment model (XGBoost only)
        if not self.detector.load_models(deploy_only=True):
            st.warning("⚠️ No XGBoost deployment model found. Please train models first!")
            return
        
        tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
        
        with tab1:
            st.subheader("Single Job Posting Analysis")
            
            st.info("🚀 Using XGBoost model for predictions (optimized for deployment)")
            
            # Input fields
            col1, col2 = st.columns(2)
            
            with col1:
                title = st.text_input("Job Title:", placeholder="e.g., Software Engineer")
                location = st.text_input("Location:", placeholder="e.g., New York, NY")
                company = st.text_area("Company Profile:", placeholder="Brief company description...")
            
            with col2:
                description = st.text_area("Job Description:", placeholder="Detailed job description...")
                requirements = st.text_area("Requirements:", placeholder="Job requirements...")
                benefits = st.text_area("Benefits:", placeholder="Benefits offered...")
            
            industry = st.text_input("Industry:", placeholder="e.g., Technology")
            
            if st.button("🔍 Analyze Job Posting", type="primary"):
                if not any([title, description, requirements]):
                    st.error("❌ Please provide at least a job title, description, or requirements!")
                    return
                
                # Create combined text
                text_data = f"{title} {location} {company} {description} {requirements} {benefits} {industry}"
                
                try:
                    # Make prediction using deployment model
                    predictions, probabilities = self.detector.predict_deployment([text_data])
                    
                    prediction = predictions[0]
                    probability = probabilities[0]
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if prediction == 1:
                            st.error(f"🚨 **FRAUDULENT**")
                        else:
                            st.success(f"✅ **LEGITIMATE**")
                    
                    with col2:
                        st.metric("Fraud Probability", f"{probability:.2%}")
                    
                    with col3:
                        confidence = max(probability, 1-probability)
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Risk assessment
                    st.subheader("📊 Risk Assessment")
                    
                    if probability >= 0.8:
                        st.error("🔴 **HIGH RISK** - Strong indicators of fraud detected")
                    elif probability >= 0.5:
                        st.warning("🟡 **MEDIUM RISK** - Some suspicious indicators present")
                    else:
                        st.success("🟢 **LOW RISK** - Appears to be legitimate")
                    
                    # Probability visualization
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = probability * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Fraud Probability (%)"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgreen"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 80
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
        
        with tab2:
            st.subheader("Batch Prediction")
            
            st.info("🚀 Upload a CSV file with job postings for batch analysis using XGBoost")
            
            uploaded_file = st.file_uploader(
                "Upload CSV file for batch prediction",
                type=['csv']
            )
            
            if uploaded_file is not None:
                try:
                    batch_data = pd.read_csv(uploaded_file)
                    st.success(f"✅ File uploaded: {batch_data.shape}")
                    
                    # Show sample data
                    st.subheader("📋 Data Preview")
                    st.dataframe(batch_data.head())
                    
                    if st.button("🚀 Run Batch Prediction", type="primary"):
                        with st.spinner("Processing predictions..."):
                            # Preprocess data
                            processed_batch = self.detector.preprocess_data(batch_data.copy(), is_training=False)
                            
                            # Make predictions
                            predictions, probabilities = self.detector.predict_deployment(processed_batch['text'])
                            
                            # Add results to dataframe
                            batch_data['fraud_prediction'] = predictions
                            batch_data['fraud_probability'] = probabilities
                            batch_data['risk_level'] = pd.cut(
                                probabilities, 
                                bins=[0, 0.5, 0.8, 1.0], 
                                labels=['Low', 'Medium', 'High']
                            )
                            
                            # Display results summary
                            st.subheader("📊 Batch Results Summary")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Jobs", len(batch_data))
                            
                            with col2:
                                fraud_count = sum(predictions)
                                st.metric("Predicted Fraudulent", fraud_count)
                            
                            with col3:
                                fraud_rate = (fraud_count / len(batch_data) * 100) if len(batch_data) > 0 else 0
                                st.metric("Fraud Rate", f"{fraud_rate:.1f}%")
                            
                            with col4:
                                avg_prob = np.mean(probabilities)
                                st.metric("Avg Fraud Prob", f"{avg_prob:.2%}")
                            
                            # Risk distribution
                            risk_counts = batch_data['risk_level'].value_counts()
                            fig = px.bar(
                                x=risk_counts.index,
                                y=risk_counts.values,
                                title="Risk Level Distribution",
                                color=risk_counts.index,
                                color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show results table
                            st.subheader("📋 Detailed Results")
                            results_df = batch_data[['fraud_prediction', 'fraud_probability', 'risk_level']].copy()
                            results_df['fraud_prediction'] = results_df['fraud_prediction'].map({0: 'Legitimate', 1: 'Fraudulent'})
                            results_df['fraud_probability'] = results_df['fraud_probability'].round(4)
                            st.dataframe(results_df)
                            
                            # Download results
                            csv = batch_data.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results CSV",
                                data=csv,
                                file_name="fraud_predictions.csv",
                                mime="text/csv"
                            )
                
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
    
    def performance_page(self):
        """Performance analysis page"""
        st.header("📈 Model Performance")
        
        # Load comparison models if available
        comparison_available = self.detector.load_models(filepath='models_comparison', deploy_only=False)
        
        if not comparison_available:
            st.warning("⚠️ No trained models found for performance analysis. Please train models first!")
            return
        
        st.info("📊 Performance comparison of all trained models (XGBoost is used for deployment)")
        
        # Model performance metrics (this would be loaded from training results)
        if 'training_results' in st.session_state:
            results = st.session_state.training_results
            
            # Performance metrics comparison
            st.subheader("🎯 Model Comparison")
            
            metrics_data = []
            for model_name, result in results.items():
                cr = result['classification_report']
                metrics_data.append({
                    'Model': model_name,
                    'F1-Score': result['f1_score'],
                    'Precision': cr['weighted avg']['precision'],
                    'Recall': cr['weighted avg']['recall'],
                    'Accuracy': cr['accuracy'],
                    'Deployment': '🚀' if model_name == 'XGBoost' else ''
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            
            # Highlight XGBoost row
            def highlight_deployment(row):
                return ['background-color: #e6f3ff' if row['Model'] == 'XGBoost' else '' for _ in row]
            
            st.dataframe(
                metrics_df.style.apply(highlight_deployment, axis=1),
                use_container_width=True
            )
            
            # Performance visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # F1-Score comparison
                fig = px.bar(
                    metrics_df,
                    x='Model',
                    y='F1-Score',
                    title='F1-Score Comparison',
                    color='Model'
                )
                # Highlight XGBoost
                fig.update_traces(
                    marker_color=['#ff6b6b' if model == 'XGBoost' else '#74b9ff' 
                                for model in metrics_df['Model']]
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Multi-metric radar chart
                metrics_for_radar = ['F1-Score', 'Precision', 'Recall', 'Accuracy']
                
                fig = go.Figure()
                
                for _, row in metrics_df.iterrows():
                    fig.add_trace(go.Scatterpolar(
                        r=[row[metric] for metric in metrics_for_radar],
                        theta=metrics_for_radar,
                        fill='toself',
                        name=row['Model'],
                        line=dict(width=3 if row['Model'] == 'XGBoost' else 2)
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="Multi-Metric Performance Comparison"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # XGBoost deployment info
            st.subheader("🚀 Deployment Model Information")
            st.info("""
            **XGBoost Selected for Deployment:**
            - Best overall performance balance
            - Robust handling of imbalanced data
            - Good interpretability
            - Fast inference time
            - Reliable probability estimates
            """)
            
        else:
            st.info("📊 Train models first to see performance metrics")
            
            # Show deployment model info
            st.subheader("🚀 Deployment Model")
            if 'XGBoost' in self.detector.models:
                st.success("✅ XGBoost model loaded and ready for deployment")
                
                # Model info
                model = self.detector.models['XGBoost']
                st.write(f"**Model Type:** {type(model).__name__}")
                st.write(f"**Features:** {model.n_features_in_ if hasattr(model, 'n_features_in_') else 'Unknown'}")
            else:
                st.warning("⚠️ XGBoost deployment model not loaded")