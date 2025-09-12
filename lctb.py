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
import pickle

warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True
PngImagePlugin.MAX_TEXT_CHUNK = 10000000

# Page configuration
st.set_page_config(
    page_title="Discount Analysis System",
    page_icon="💰",
    layout="wide"
)

def load_uploaded_model(uploaded_file, model_type):
    """Load models from uploaded files"""
    try:
        if model_type == "keras":
            # Save uploaded file temporarily and load
            with open("temp_model.keras", "wb") as f:
                f.write(uploaded_file.getbuffer())
            model = tf.keras.models.load_model("temp_model.keras")
            return model
        elif model_type == "pkl":
            # Load pickle model directly from bytes
            model = joblib.load(io.BytesIO(uploaded_file.getvalue()))
            return model
    except Exception as e:
        st.error(f"Error loading {model_type} model: {e}")
        return None

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

def process_discount_analysis(files_dict, week_range, selected_segments, discount_model, gradient_model):
    """Main processing function"""
    try:
        st.info("Processing discount analysis...")
        
        # Get main dataframes
        st_item = files_dict['st_item']
        A = files_dict['A']
        B = files_dict['B']
        calendar = files_dict['calendar']
        tracking = files_dict['tracking']
        goals = files_dict['goals']
        segment = files_dict['segment']
        images_df = files_dict['images']
        df_sequenza = files_dict['sequenza']
        
        # Process calendar data
        calendar['YearWeek'] = calendar['YearWeek'].astype(str)
        calendar[['anno', 'settimana']] = calendar['YearWeek'].str.split('-', n=1, expand=True)
        calendar['anno'] = calendar['anno'].astype(int)
        calendar['settimana'] = calendar['settimana'].astype(int)
        calendar = calendar.sort_values(by=['anno', 'settimana']).drop(columns=['anno', 'settimana']).reset_index(drop=True)
        
        # Get current week
        current_week = calendar.iloc[-1]["YearWeek"]
        
        # Filter tracking data
        tracking_filtered = tracking[tracking["% Stores with Tracking within 6 weeks"] >= 0.3]
        tracking_below = tracking[tracking["% Stores with Tracking within 6 weeks"] < 0.3]
        
        # Clean data
        A = A.dropna(subset=['Item Code']).fillna(0)
        B = B.dropna(subset=['Item Code']).fillna(0)
        st_item = st_item.dropna(subset=['Item Code'])
        A = A[A['Commercial YearWeek'] != 0].reset_index(drop=True)
        
        # Merge st_item with A
        st_item = st_item.merge(A[['Item Code', 'Commercial YearWeek', 'Commercial YearMonth']], on='Item Code', how='left')
        
        # Filter A dataframe by week range
        start_week, end_week = week_range
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
        
        # Categorize ST items
        function_months = set(zip(st_item['Function'], st_item['Commercial YearMonth']))
        df_clusters = pd.DataFrame(columns=st_item.columns)
        
        for func, year_month in function_months:
            df_clusters = categorize_st(st_item, func, year_month, df_clusters)
        
        # Define item sets
        items_in_exposition_month = set(A_filtered['Item Code'])
        items_not_in_exposition_month = set(A['Item Code']) - set(A_filtered['Item Code'])
        item_codes_comuni_con_B = set(A_filtered['Item Code']).intersection(set(B['Item Code']))
        
        # Items without sales in exposition month
        df_items_in_exposition_month_without_sales = A[(A["Item Code"].isin(items_in_exposition_month)) & (A["First Sale YearWeek"] == 0)]
        items_in_exposition_month_without_sales = set(df_items_in_exposition_month_without_sales["Item Code"])
        
        # Items with tracking data
        df_items_above_tracked_in_exposition_month = tracking_filtered[tracking_filtered["Item Code"].isin(items_in_exposition_month)]
        items_above_tracked_in_exposition_month = set(df_items_above_tracked_in_exposition_month["Item Code"])
        
        df_items_below_tracked_in_exposition_month = tracking_below[tracking_below["Item Code"].isin(items_in_exposition_month)]
        items_below_tracked_in_exposition_month = set(df_items_below_tracked_in_exposition_month["Item Code"])
        
        df_items_not_tracked = A[A["First Tracking YearWeek"] == 0]
        items_not_tracked = set(df_items_not_tracked["Item Code"])
        
        # Process B data for Delta ST calculations
        B_filtered = B[B['Item Code'].isin(items_above_tracked_in_exposition_month)]
        
        def remove_leading_zero(year_week):
            year, week = year_week.split('-')
            week = str(int(week))
            return f"{year}-{week}"
        
        B_filtered['YearWeek'] = B_filtered['YearWeek'].apply(remove_leading_zero)
        current_yw = calendar.iloc[-1]["YearWeek"]
        
        # Calculate proposals for tracked items
        for item in items_above_tracked_in_exposition_month:
            parts = current_yw.split("-")
            week_number = int(parts[1])
            
            if week_number != 1:
                week_number_p2w = str(week_number)
                week_number_p3w = str(week_number - 1)
            else:
                week_number_p2w = "52"
                week_number_p3w = "51"
                
            week_p2w = parts[0] + "-" + week_number_p2w
            week_p3w = parts[0] + "-" + week_number_p3w
            
            p2w_data = B_filtered.loc[(B_filtered["Item Code"] == item) & (B_filtered['YearWeek'] == week_p2w), 'Delta ST PW']
            p3w_data = B_filtered.loc[(B_filtered["Item Code"] == item) & (B_filtered['YearWeek'] == week_p3w), 'Delta ST PW']
            
            def format_percent(x):
                if x is None:
                    return "-"
                else:
                    return f"{x*100:.2f}%".replace('.', ',')
            
            p2w = p2w_data.values[0] if not p2w_data.empty else None
            p3w = p3w_data.values[0] if not p3w_data.empty else None
            
            item_index = df_clusters.index[df_clusters['Item Code'] == item].tolist()
            if item_index:
                item_index = item_index[0]
                cluster = df_clusters.at[item_index, 'ST_Cluster']
                df_clusters.at[item_index, 'Delta ST P2W'] = format_percent(p2w)
                df_clusters.at[item_index, 'Delta ST P3W'] = format_percent(p3w)
                
                item_function = df_clusters.at[item_index, 'Function']
                mask_function = (goals['Function'] == item_function)
                
                if not goals.loc[mask_function].empty:
                    row = goals.loc[mask_function].iloc[0]
                    theoretical_increase = row['Teorethical Increase %']
                    num_life_weeks = row['NumLifeWeeks']
                    if num_life_weeks == -1:
                        threshold = 0.025
                    else:
                        threshold = 0.75 * theoretical_increase
                else:
                    threshold = 0.0196
                    theoretical_increase = 0.0196
                
                # Determine proposal based on cluster and performance
                if p2w is not None and p2w > theoretical_increase * 1.25:
                    df_clusters.at[item_index, 'Proposal'] = "Nessuno Sconto"
                else:
                    if cluster == "Basso":
                        if (p2w and p2w < threshold) or (p3w and p3w < threshold):
                            df_clusters.at[item_index, 'Proposal'] = "Sconto Alto"
                        else:
                            df_clusters.at[item_index, 'Proposal'] = "Sconto Medio"
                    elif cluster == "Medio Basso":
                        if (p2w and p2w < threshold) or (p3w and p3w < threshold):
                            df_clusters.at[item_index, 'Proposal'] = "Sconto Medio"
                        else:
                            df_clusters.at[item_index, 'Proposal'] = "Sconto Basso"
                    elif cluster in ["Alto", "Medio Alto"]:
                        if p2w and p2w < threshold:
                            if p3w and p3w < threshold:
                                df_clusters.at[item_index, 'Proposal'] = "Sconto Basso"
                            else:
                                df_clusters.at[item_index, 'Proposal'] = "Nessuno Sconto"
                        else:
                            df_clusters.at[item_index, 'Proposal'] = "Nessuno Sconto"
        
        # Set proposals for other item categories
        for item_list, proposal in [
            (items_in_exposition_month_without_sales, "Sconto Alto (NO SALES)"),
            (items_not_in_exposition_month, "NESSUNA PROPOSTA (item fuori da exposition months)"),
            (items_not_tracked, "NESSUNA PROPOSTA (item senza tracking)"),
            (items_below_tracked_in_exposition_month, "NESSUNA PROPOSTA (item in exposition months con tracking sotto 30%)")
        ]:
            for item in item_list:
                item_index = df_clusters.index[df_clusters['Item Code'] == item].tolist()
                if item_index:
                    item_index = item_index[0]
                    df_clusters.at[item_index, 'Proposal'] = proposal
        
        # Merge with other data
        A_excluded = A.drop(columns=['Commercial YearWeek', 'Commercial YearMonth'], errors='ignore')
        merged_df = pd.merge(df_clusters, A_excluded, on="Item Code", how="left")
        merged_df2 = pd.merge(merged_df, tracking, on="Item Code", how="left")
        
        # Calculate additional metrics
        merged_df2['AVG ST Function per CommercialMonth'] = merged_df2.groupby(["Function", "Commercial YearMonth"])['ST item'].transform('mean').round(4)
        merged_df2['AVG ST Function'] = merged_df2.groupby(["Function"])['ST item'].transform('mean').round(4)
        
        condition = merged_df2["Metodo Cluster"] == "Cluster funzione/mese commerciale"
        merged_df2["ST Difference"] = (
            merged_df2["ST item"].round(4) - 
            merged_df2.groupby(["Function", "Commercial YearMonth"])["ST item"].transform("mean").round(4)
        ).where(condition,
            merged_df2["ST item"].round(4) - 
            merged_df2.groupby(["Function"])["ST item"].transform("mean").round(4)
        )
        
        # Filter by segments
        segment_filtered = segment[segment["Segment"].isin(selected_segments)]
        items_in_right_segment = set(segment_filtered["Cod item"])
        merged_df_final = merged_df2[merged_df2["Item Code"].isin(items_in_right_segment)]
        
        # Merge with segment data
        merged_df2 = pd.merge(merged_df2, segment, left_on="Item Code", right_on="Cod item", how="left")
        merged_df2 = merged_df2.drop(columns=["Cod item"])
        
        # Add TFI mapping
        def format_tfi(x):
            return f"{x*100:.2f}%".replace('.', ',')
        
        mapping_tfi = goals.set_index("Function")["Teorethical Increase %"].apply(format_tfi).to_dict()
        merged_df2["TFI"] = merged_df2["Function"].map(mapping_tfi)
        merged_df2["TFI"] = merged_df2["TFI"].fillna("1,96%")
        
        # Calculate stock and SVA
        merged_df2["Sales item"] = pd.to_numeric(merged_df2["Sales item"], errors='coerce')
        merged_df2["Delivered item"] = pd.to_numeric(merged_df2["Delivered item"], errors='coerce')
        merged_df2["Stock residuo"] = merged_df2["Delivered item"] - merged_df2["Sales item"]
        
        perc_basso = 0.2
        perc_medio = 0.3
        perc_alto = 0.5
        
        merged_df2["SVA"] = merged_df2.apply(lambda row: 
            row["Stock residuo"] * perc_basso if row["Proposal"] == "Sconto Basso" else
            row["Stock residuo"] * perc_medio if row["Proposal"] == "Sconto Medio" else
            row["Stock residuo"] * perc_alto if row["Proposal"] == "Sconto Alto" else 0,
            axis=1
        )
        
        merged_df2["Sconto proposto"] = merged_df2.apply(lambda row: 
            "SI" if row["Proposal"] in ["Sconto Basso", "Sconto Medio", "Sconto Alto"] else "NO",
            axis=1
        )
        
        # Add processing date
        elaboration_date = datetime.today().strftime('%d-%m-%Y')
        merged_df2['Data elaborazione'] = elaboration_date
        
        # Filter by minimum delivered quantity
        merged_df2 = merged_df2[merged_df2["Delivered item"] >= 5000]
        
        return merged_df2
        
    except Exception as e:
        st.error(f"Error during processing: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

def main():
    st.title("Discount Analysis System")
    st.markdown("Upload your files and configure parameters to generate discount proposals.")
    
    # Sidebar for file uploads
    st.sidebar.header("File Uploads")
    
    # Excel files upload
    st.sidebar.subheader("Excel Files")
    excel_files = {}
    required_excel_files = {
        'st_item': 'ST Item Excel file',
        'A': 'A Excel file', 
        'B': 'B Excel file',
        'calendar': 'Calendar Excel file',
        'tracking': 'Tracking Excel file',
        'goals': 'Goals Excel file',
        'segment': 'Segment Excel file',
        'images': 'Images Excel file',
        'sequenza': 'Sequenza Excel file'
    }
    
    for key, label in required_excel_files.items():
        uploaded_file = st.sidebar.file_uploader(
            label,
            type=['xlsx', 'xls'],
            key=f"excel_{key}"
        )
        if uploaded_file:
            try:
                excel_files[key] = pd.read_excel(uploaded_file)
                st.sidebar.success(f"✅ {label} loaded")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading {label}: {e}")
    
    # Model files upload
    st.sidebar.subheader("Model Files")
    
    keras_model_file = st.sidebar.file_uploader(
        "Keras Model (.keras file)",
        type=['keras'],
        key="keras_model"
    )
    
    pkl_model_file = st.sidebar.file_uploader(
        "Gradient Boosting Model (.pkl file)", 
        type=['pkl'],
        key="pkl_model"
    )
    
    # Load models if uploaded
    discount_model = None
    gradient_model = None
    
    if keras_model_file:
        discount_model = load_uploaded_model(keras_model_file, "keras")
        if discount_model:
            st.sidebar.success("✅ Keras model loaded")
        else:
            st.sidebar.error("❌ Failed to load Keras model")
    
    if pkl_model_file:
        gradient_model = load_uploaded_model(pkl_model_file, "pkl")
        if gradient_model:
            st.sidebar.success("✅ Gradient boosting model loaded")
        else:
            st.sidebar.error("❌ Failed to load gradient boosting model")
    
    # Configuration section
    st.sidebar.header("Configuration")
    
    # Week range selection
    st.sidebar.subheader("Week Range")
    start_week = st.sidebar.text_input("Start Week (YYYY-WW)", value="2024-01")
    end_week = st.sidebar.text_input("End Week (YYYY-WW)", value="2024-52")
    
    # Segment selection
    st.sidebar.subheader("Segments")
    segment_options = [1, 2, 3, 4]  # Numeric values as in original code
    selected_segments = st.sidebar.multiselect(
        "Select segments to include:",
        segment_options,
        default=segment_options
    )
    
    # Check if all files are uploaded
    all_excel_uploaded = len(excel_files) == len(required_excel_files)
    all_models_uploaded = discount_model is not None and gradient_model is not None
    
    # Status display
    st.subheader("Upload Status")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Excel Files:**")
        for key, label in required_excel_files.items():
            if key in excel_files:
                st.write(f"✅ {label}")
            else:
                st.write(f"❌ {label}")
    
    with col2:
        st.write("**Model Files:**")
        if discount_model:
            st.write("✅ Keras model")
        else:
            st.write("❌ Keras model")
        
        if gradient_model:
            st.write("✅ Gradient boosting model")
        else:
            st.write("❌ Gradient boosting model")
    
    # Processing button
    if st.sidebar.button("Process Analysis", type="primary"):
        if not all_excel_uploaded:
            st.error("Please upload all required Excel files.")
            return
        
        if not all_models_uploaded:
            st.error("Please upload both model files.")
            return
        
        if not selected_segments:
            st.error("Please select at least one segment.")
            return
        
        with st.spinner("Processing analysis... This may take several minutes."):
            # Process the analysis
            result_df = process_discount_analysis(
                excel_files,
                (start_week, end_week),
                selected_segments,
                discount_model,
                gradient_model
            )
            
            if result_df is not None:
                # Display results
                st.subheader("Analysis Results")
                
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
                st.subheader("Data Preview")
                st.dataframe(result_df.head(100), height=400)
                
                # Download section
                st.subheader("Download Results")
                
                # Convert to Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    result_df.to_excel(writer, index=False, sheet_name='Discount Analysis')
                
                output.seek(0)
                
                # Download button
                current_date = datetime.now().strftime('%d-%m-%Y')
                filename = f"IC_proposte_sconti_{current_date}.xlsx"
                
                st.download_button(
                    label="Download Excel Report",
                    data=output.getvalue(),
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    # Instructions
    with st.expander("Instructions"):
        st.markdown("""
        ### How to use this application:
        
        1. **Upload Files**: Upload all required Excel files and model files using the sidebar
        2. **Configure Parameters**: Set the week range and select segments
        3. **Process Analysis**: Click the "Process Analysis" button
        4. **Review Results**: Check the analysis results and key metrics
        5. **Download Report**: Download the formatted Excel report
        
        ### Required Files:
        - **Excel Files**: st_item, A, B, calendar, tracking, goals, segment, images, sequenza
        - **Model Files**: discount_predictive_model_v2.keras, optimized_gradient_boosting_model.pkl
        """)

if __name__ == "__main__":
    main()
