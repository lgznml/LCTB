import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import warnings
import io
from PIL import Image, ImageFile, PngImagePlugin
import requests
from io import BytesIO
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor, as_completed
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill

warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True
PngImagePlugin.MAX_TEXT_CHUNK = 10000000

# Page configuration
st.set_page_config(
    page_title="Discount Analysis System",
    page_icon="💰",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load the pre-trained models"""
    try:
        # Load your models here - you'll need to upload them to your deployment
        discount_model = tf.keras.models.load_model("discount_predictive_model_v2.keras")
        gradient_model = joblib.load("optimized_gradient_boosting_model.pkl")
        return discount_model, gradient_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    img = image.copy()
    img.thumbnail(target_size, Image.LANCZOS)
    new_img = Image.new("RGB", target_size)
    left = (target_size[0] - img.size[0]) // 2
    top = (target_size[1] - img.size[1]) // 2
    new_img.paste(img, (left, top))
    return new_img

def categorize_st(df, function_name, year_month, df_classified):
    """Categorize ST items based on function and year-month"""
    df_filtered = df[(df['Function'] == function_name) & (df['Commercial YearMonth'] == year_month)].copy()
    
    if df_filtered.empty:
        return df_classified
    
    if df_filtered.shape[0] == 1:
        df_function = df[df['Function'] == function_name]
        st_percentiles = df_function['ST item'].quantile([0.25, 0.5, 0.75])
        cluster_method = "Cluster funzione"
    else:
        st_percentiles = df_filtered['ST item'].quantile([0.25, 0.5, 0.75])
        cluster_method = "Cluster funzione/mese commerciale"
    
    def categorize(row):
        if row['ST item'] <= st_percentiles[0.25]:
            return 'Basso'
        elif row['ST item'] <= st_percentiles[0.5]:
            return 'Medio Basso'
        elif row['ST item'] <= st_percentiles[0.75]:
            return 'Medio Alto'
        else:
            return 'Alto'
    
    df_filtered['ST_Cluster'] = df_filtered.apply(categorize, axis=1)
    df_filtered['Metodo Cluster'] = cluster_method
    df_classified = pd.concat([df_classified, df_filtered], ignore_index=True)
    
    return df_classified

def process_discount_analysis(uploaded_files, week_range, selected_segments):
    """Main processing function"""
    try:
        # Load uploaded files
        st.info("Loading uploaded files...")
        
        files_dict = {}
        required_files = ['st_item', 'A', 'B', 'calendar', 'tracking', 'goals', 'segment', 'images', 'sequenza']
        
        for file in uploaded_files:
            file_key = file.name.split('.')[0].lower()
            if any(req in file_key for req in required_files):
                files_dict[file_key] = pd.read_excel(file)
        
        # Validate required files
        missing_files = [f for f in required_files if not any(f in k for k in files_dict.keys())]
        if missing_files:
            st.error(f"Missing required files: {missing_files}")
            return None
        
        # Get main dataframes
        st_item = files_dict.get('st_item')
        A = files_dict.get('a')
        B = files_dict.get('b')
        calendar = files_dict.get('calendar')
        tracking = files_dict.get('tracking')
        goals = files_dict.get('goals')
        segment = files_dict.get('segment')
        images_df = files_dict.get('images')
        df_sequenza = files_dict.get('sequenza')
        
        # Process calendar data
        calendar['YearWeek'] = calendar['YearWeek'].astype(str)
        calendar[['anno', 'settimana']] = calendar['YearWeek'].str.split('-', n=1, expand=True)
        calendar['anno'] = calendar['anno'].astype(int)
        calendar['settimana'] = calendar['settimana'].astype(int)
        calendar = calendar.sort_values(by=['anno', 'settimana']).drop(columns=['anno', 'settimana']).reset_index(drop=True)
        
        # Filter data based on user inputs
        start_week, end_week = week_range
        
        # Filter A dataframe by week range
        start_year, start_week_num = map(int, start_week.split('-'))
        end_year, end_week_num = map(int, end_week.split('-'))
        
        calendar[['anno', 'settimana']] = calendar['YearWeek'].str.split('-', n=1, expand=True)
        calendar['anno'] = calendar['anno'].astype(int)
        calendar['settimana'] = calendar['settimana'].astype(int)
        
        mask = (
            ((calendar['anno'] > start_year) | ((calendar['anno'] == start_year) & (calendar['settimana'] >= start_week_num))) &
            ((calendar['anno'] < end_year) | ((calendar['anno'] == end_year) & (calendar['settimana'] <= end_week_num)))
        )
        yearweeks = calendar[mask]['YearWeek'].drop_duplicates().tolist()
        A_filtered = A[A['First Tracking YearWeek'].astype(str).isin(yearweeks)]
        
        # Filter segment data
        segment_filtered = segment[segment["Segment"].isin(selected_segments)]
        
        # Continue with your existing processing logic...
        # This is where you'd implement all your analysis logic
        # but adapted for web deployment
        
        st.success("Processing completed!")
        
        # Return processed dataframe
        return A_filtered  # Replace with your final merged_df2
        
    except Exception as e:
        st.error(f"Error during processing: {e}")
        return None

def main():
    st.title("💰 Discount Analysis System")
    st.markdown("Upload your files and configure parameters to generate discount proposals.")
    
    # Load models
    discount_model, gradient_model = load_models()
    if discount_model is None or gradient_model is None:
        st.error("Models not loaded. Please ensure model files are available.")
        return
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # File upload section
    st.sidebar.subheader("📁 File Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Upload required Excel files:",
        accept_multiple_files=True,
        type=['xlsx', 'xls'],
        help="Upload: st_item, A, B, calendar, tracking, goals, segment, images, sequenza files"
    )
    
    if not uploaded_files:
        st.info("Please upload the required Excel files to proceed.")
        return
    
    # Week range selection
    st.sidebar.subheader("📅 Week Range")
    start_week = st.sidebar.text_input("Start Week (YYYY-WW)", value="2024-01")
    end_week = st.sidebar.text_input("End Week (YYYY-WW)", value="2024-52")
    
    # Segment selection
    st.sidebar.subheader("🎯 Segments")
    # You'd populate this from the uploaded segment file
    segment_options = ["1 ~ Fast Fashion", "2 ~ Core Collection", "3 ~ Seasonal Basic", "4 ~ Special Price"]
    selected_segments = st.sidebar.multiselect(
        "Select segments to include:",
        segment_options,
        default=segment_options
    )
    
    # Processing button
    if st.sidebar.button("🚀 Process Analysis", type="primary"):
        if len(uploaded_files) < 9:
            st.warning("Please ensure all required files are uploaded.")
            return
        
        with st.spinner("Processing analysis... This may take several minutes."):
            # Process the analysis
            result_df = process_discount_analysis(
                uploaded_files, 
                (start_week, end_week), 
                selected_segments
            )
            
            if result_df is not None:
                # Display results
                st.subheader("📊 Analysis Results")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Items", len(result_df))
                with col2:
                    discount_items = len(result_df[result_df.get('Sconto proposto', '') == 'SI'])
                    st.metric("Items with Discount", discount_items)
                with col3:
                    avg_sva = result_df.get('SVA', pd.Series([0])).mean()
                    st.metric("Average SVA", f"{avg_sva:.2f}")
                with col4:
                    total_stock = result_df.get('Stock residuo', pd.Series([0])).sum()
                    st.metric("Total Residual Stock", f"{total_stock:.0f}")
                
                # Data preview
                st.subheader("📋 Data Preview")
                st.dataframe(result_df.head(100), height=400)
                
                # Download section
                st.subheader("⬇️ Download Results")
                
                # Convert to Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    result_df.to_excel(writer, index=False, sheet_name='Discount Analysis')
                
                # Format the Excel file (simplified version)
                output.seek(0)
                
                # Download button
                current_date = datetime.now().strftime('%d-%m-%Y')
                filename = f"IC_proposte_sconti_{current_date}.xlsx"
                
                st.download_button(
                    label="📥 Download Excel Report",
                    data=output.getvalue(),
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    # Instructions
    with st.expander("ℹ️ Instructions"):
        st.markdown("""
        ### How to use this application:
        
        1. **Upload Files**: Upload all required Excel files using the file uploader
        2. **Configure Parameters**: Set the week range and select segments
        3. **Process Analysis**: Click the "Process Analysis" button
        4. **Review Results**: Check the analysis results and key metrics
        5. **Download Report**: Download the formatted Excel report
        
        ### Required Files:
        - `st_item.xlsx`: Stock turn items data
        - `A.xlsx`: Main analysis data
        - `B.xlsx`: Delta ST data
        - `calendar.xlsx`: Calendar mapping
        - `tracking.xlsx`: Tracking percentages
        - `goals.xlsx`: Function goals
        - `segment.xlsx`: Item segments
        - `images.xlsx`: Image URLs
        - `sequenza.xlsx`: Discount sequence data
        """)

if __name__ == "__main__":
    main()