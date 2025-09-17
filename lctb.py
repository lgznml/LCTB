import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import requests
from io import BytesIO
from PIL import Image, ImageFile, PngImagePlugin
from tensorflow.keras.models import load_model
from concurrent.futures import ThreadPoolExecutor, as_completed
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill
import tempfile

# Configure PIL
ImageFile.LOAD_TRUNCATED_IMAGES = True
PngImagePlugin.MAX_TEXT_CHUNK = 10000000

st.set_page_config(layout="wide", page_title="IC Proposte Sconti - Streamlit")
st.title("IC Proposte Sconti - Streamlit Port of Local Script")

st.markdown(
    "Questo è un port (non 1:1) dell'applicazione di analisi e proposta sconti creata come script locale. "
    "Carica i file richiesti nella barra laterale (o lascia i percorsi locali se esegui in macchina con accesso ai file)."
)

# ----------------------------- Sidebar: upload files / settings -----------------------------
st.sidebar.header("Input files & models")

def uploaded_or_local(uploader, local_path_placeholder):
    f = uploader
    if f is None:
        path = st.sidebar.text_input(f"Percorso locale (se non carichi file):", value=local_path_placeholder)
        return path
    else:
        return f

# Excel files
st.sidebar.subheader("Excel files (A, B, st_item, calendar, tracking, goals, segment, sequenza, immagini)")
A_file = st.sidebar.file_uploader("A.xlsx", type=["xlsx"])  
B_file = st.sidebar.file_uploader("B.xlsx", type=["xlsx"])  
st_item_file = st.sidebar.file_uploader("st_item.xlsx", type=["xlsx"])  
calendar_file = st.sidebar.file_uploader("calendar.xlsx", type=["xlsx"])  
tracking_file = st.sidebar.file_uploader("% tracking per negozio.xlsx", type=["xlsx"])  
goals_file = st.sidebar.file_uploader("function_goals.xlsx", type=["xlsx"])  
segment_file = st.sidebar.file_uploader("segment.xlsx", type=["xlsx"])  
sequenza_file = st.sidebar.file_uploader("sequenza articoli sconto.xlsx", type=["xlsx"])  
images_file = st.sidebar.file_uploader("immagini FW 25.xlsx", type=["xlsx"])  

# Models
st.sidebar.subheader("Models (put them in repo or upload here)")
forecast_model_file = st.sidebar.file_uploader("optimized_gradient_boosting_model.pkl", type=["pkl"])  
image_model_file = st.sidebar.file_uploader("discount_predictive_model_v2.keras", type=["keras", "h5", "hdf5"])  

# Additional settings
st.sidebar.subheader("Filters / runtime options")
category_choice = st.sidebar.selectbox("Categoria (Cod Category)", options=("auto", 31, 32, 33), index=0)
min_delivered = st.sidebar.number_input("Min Delivered item to keep (final filter)", min_value=0, value=5000)

# Week range selection
st.sidebar.subheader("Intervallo settimane (formato AAAA-WW)")
start_week_input = st.sidebar.text_input("Settimana iniziale (es. 2024-01)")
end_week_input = st.sidebar.text_input("Settimana finale (es. 2024-20)")

# Run button
run_button = st.sidebar.button("Esegui elaborazione")

# Helper functions

def read_excel_input(f, sheet_name=0):
    if hasattr(f, "read"):
        try:
            return pd.read_excel(f, sheet_name=sheet_name)
        except Exception as e:
            st.error(f"Errore lettura excel: {e}")
            return pd.DataFrame()
    else:
        # assume it's a path
        try:
            return pd.read_excel(f)
        except Exception as e:
            st.error(f"Errore lettura excel dal path: {e}")
            return pd.DataFrame()


def is_valid_yearweek(yearweek):
    try:
        year, week = yearweek.split('-')
        year = int(year)
        week = int(week)
        return 1 <= week <= 52
    except:
        return False


def remove_leading_zero(year_week):
    try:
        year, week = year_week.split('-')
        week = str(int(week))
        return f"{year}-{week}"
    except:
        return year_week


# Main processing (runs when user clicks 'Esegui elaborazione')
if run_button:
    # Read all inputs, with fallback to local paths
    st.info("Caricamento file...")
    A = read_excel_input(A_file, sheet_name=0)
    B = read_excel_input(B_file, sheet_name=0)
    st_item = read_excel_input(st_item_file, sheet_name=0)
    calendar = read_excel_input(calendar_file, sheet_name=0)
    tracking = read_excel_input(tracking_file, sheet_name=0)
    goals = read_excel_input(goals_file, sheet_name=0)
    segment = read_excel_input(segment_file, sheet_name=0)
    sequenza = read_excel_input(sequenza_file, sheet_name=0)
    images_df = read_excel_input(images_file, sheet_name=0)

    # Validate required data
    required = {"A": A, "B": B, "st_item": st_item, "calendar": calendar, "tracking": tracking, "goals": goals}
    for name, df in required.items():
        if df is None or df.empty:
            st.warning(f"Attenzione: il file {name} sembra vuoto o non caricato. Controlla l'input.")

    # Preprocessing same as original script (adapted)
    try:
        calendar['YearWeek'] = calendar['YearWeek'].astype(str)
        calendar[['anno', 'settimana']] = calendar['YearWeek'].str.split('-', n=1, expand=True)
        calendar['anno'] = calendar['anno'].astype(int)
        calendar['settimana'] = calendar['settimana'].astype(int)
        calendar = calendar.sort_values(by=['anno', 'settimana']).drop(columns=['anno', 'settimana']).reset_index(drop=True)
    except Exception as e:
        st.error(f"Errore preprocessing calendar: {e}")

    # Tracking filter
    tracking_filtered = tracking[tracking["% Stores with Tracking within 6 weeks"] >= 0.3]
    tracking_below = tracking[tracking["% Stores with Tracking within 6 weeks"] < 0.3]

    # Clean A, B, st_item
    A = A.dropna(subset=['Item Code']).fillna(0)
    B = B.dropna(subset=['Item Code']).fillna(0)
    st_item = st_item.dropna(subset=['Item Code'])
    A = A[A['Commercial YearWeek'] != 0].reset_index(drop=True)

    # Merge st_item with A to get commercial fields
    st_item = st_item.merge(A[['Item Code', 'Commercial YearWeek', 'Commercial YearMonth']], on='Item Code', how='left')

    # determine category
    try:
        category_val = int(max(st_item["Cod Category"]))
    except Exception:
        category_val = category_choice if category_choice != "auto" else 31

    # determine current_week
    try:
        current_week = calendar.iloc[-1]["YearWeek"]
    except Exception:
        current_week = datetime.today().strftime('%Y-%W')
    st.write(f"Current week detected: {current_week}")

    # Week range filtering
    if not (is_valid_yearweek(start_week_input) and is_valid_yearweek(end_week_input)):
        st.warning("Formato week range non valido. Inserisci AAAA-WW nel sidebar.)")

    # Create filter_by_week_range equivalent
    def filter_by_week_range_df(A_df, calendar_df, start_week, end_week):
        calendar_df = calendar_df.copy()
        calendar_df['YearWeek'] = calendar_df['YearWeek'].astype(str)
        calendar_df[['anno', 'settimana']] = calendar_df['YearWeek'].str.split('-', n=1, expand=True)
        calendar_df['anno'] = calendar_df['anno'].astype(int)
        calendar_df['settimana'] = calendar_df['settimana'].astype(int)
        calendar_df = calendar_df.sort_values(by=['anno', 'settimana']).reset_index(drop=True)
        start_year, start_week_num = map(int, start_week.split('-'))
        end_year, end_week_num = map(int, end_week.split('-'))
        mask = (
            ((calendar_df['anno'] > start_year) | ((calendar_df['anno'] == start_year) & (calendar_df['settimana'] >= start_week_num))) &
            ((calendar_df['anno'] < end_year) | ((calendar_df['anno'] == end_year) & (calendar_df['settimana'] <= end_week_num)))
        )
        yearweeks = calendar_df[mask]['YearWeek'].drop_duplicates().tolist()
        if not yearweeks:
            st.warning("Nessuna settimana trovata nell'intervallo selezionato.")
            return pd.DataFrame()
        st.write(f"Filtrando YearWeeks da {start_week} a {end_week}: {yearweeks}")
        return A_df[A_df['First Tracking YearWeek'].astype(str).isin(yearweeks)]

    if start_week_input and end_week_input and is_valid_yearweek(start_week_input) and is_valid_yearweek(end_week_input):
        A_filtered = filter_by_week_range_df(A, calendar, start_week_input, end_week_input)
    else:
        A_filtered = A.copy()

    # Clustering function
    def categorize_st(df, function_name, year_month, df_classified):
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

    function_months = set(zip(st_item['Function'], st_item['Commercial YearMonth']))
    df_clusters = pd.DataFrame(columns=st_item.columns)
    for func, year_month in function_months:
        df_clusters = categorize_st(st_item, func, year_month, df_clusters)

    # Compute sets and dataframes used later
    items_in_exposition_month = set(A_filtered['Item Code'].astype(str))
    items_not_in_exposition_month = set(A['Item Code'].astype(str)) - items_in_exposition_month
    item_codes_comuni_con_B = set(A_filtered['Item Code'].astype(str)).intersection(set(B['Item Code'].astype(str)))
    total_items = set(A['Item Code'].astype(str))
    df_items_nulli = A[(A["First Tracking YearWeek"] == 0) & (A["First Sale YearWeek"] == 0)]
    lista_items_nulli = set(df_items_nulli['Item Code'].astype(str))

    df_items_in_exposition_month = A[A["Item Code"].astype(str).isin(items_in_exposition_month)]
    df_items_not_in_exposition_month = A[A["Item Code"].astype(str).isin(items_not_in_exposition_month)]
    df_item_codes_comuni_con_B = A[A["Item Code"].astype(str).isin(item_codes_comuni_con_B)]
    items_not_in_exposition_month_and_null = set(lista_items_nulli).intersection(set(items_not_in_exposition_month))
    df_items_not_in_exposition_month_and_null = A[A["Item Code"].astype(str).isin(items_not_in_exposition_month_and_null)]
    df_items_in_exposition_month_without_sales = A[(A["Item Code"].astype(str).isin(items_in_exposition_month)) & (A["First Sale YearWeek"] == 0)]
    items_in_exposition_month_without_sales = set(df_items_in_exposition_month_without_sales["Item Code"].astype(str))
    df_items_in_exposition_month_with_sales = A[(A["Item Code"].astype(str).isin(items_in_exposition_month)) & (A["First Sale YearWeek"] != 0)]
    items_in_exposition_month_with_sales = set(df_items_in_exposition_month_with_sales["Item Code"].astype(str))

    df_items_above_tracked_in_exposition_month = tracking_filtered[tracking_filtered["Item Code"].astype(str).isin(items_in_exposition_month)]
    items_above_tracked_in_exposition_month = set(df_items_above_tracked_in_exposition_month["Item Code"].astype(str))
    df_items_below_tracked_in_exposition_month = tracking_below[tracking_below["Item Code"].astype(str).isin(items_in_exposition_month)]
    items_below_tracked_in_exposition_month = set(df_items_below_tracked_in_exposition_month["Item Code"].astype(str))
    df_items_not_tracked = A[A["First Tracking YearWeek"] == 0]
    items_not_tracked = set(df_items_not_tracked["Item Code"].astype(str))

    # Start populating df_clusters with Delta ST P2W and P3W and Proposal
    B_filtered = B[B['Item Code'].astype(str).isin(items_above_tracked_in_exposition_month)].copy()
    B_filtered['YearWeek'] = B_filtered['YearWeek'].astype(str).apply(remove_leading_zero)
    current_yw = calendar.iloc[-1]["YearWeek"]

    def format_percent(x):
        if x is None or pd.isna(x):
            return "-"
        else:
            try:
                return f"{x*100:.2f}%".replace('.', ',')
            except:
                return str(x)

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
        p2w_data = B_filtered.loc[(B_filtered["Item Code"].astype(str) == str(item)) & (B_filtered['YearWeek'] == week_p2w), 'Delta ST PW']
        p3w_data = B_filtered.loc[(B_filtered["Item Code"].astype(str) == str(item)) & (B_filtered['YearWeek'] == week_p3w), 'Delta ST PW']
        p2w = p2w_data.values[0] if not p2w_data.empty else None
        p3w = p3w_data.values[0] if not p3w_data.empty else None
        item_index = df_clusters.index[df_clusters['Item Code'].astype(str) == str(item)].tolist()
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
            # Proposal logic (same as original)
            try:
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
            except Exception:
                df_clusters.at[item_index, 'Proposal'] = "Nessuna Informazione"

    # Special groups proposals
    special_assignments = [
        (items_in_exposition_month_without_sales, "Sconto Alto (NO SALES)"),
        (items_not_in_exposition_month, "NESSUNA PROPOSTA (item fuori da exposition months)"),
        (items_not_tracked, "NESSUNA PROPOSTA (item senza tracking)"),
        (items_below_tracked_in_exposition_month, "NESSUNA PROPOSTA (item in exposition months con tracking sotto 30%)")
    ]
    for item_list, proposal in special_assignments:
        for item in item_list:
            item_index = df_clusters.index[df_clusters['Item Code'].astype(str) == str(item)].tolist()
            if item_index:
                item_index = item_index[0]
                df_clusters.at[item_index, 'Proposal'] = proposal

    # Merge with A (excluded columns) and tracking
    A_excluded = A.drop(columns=['Commercial YearWeek', 'Commercial YearMonth'], errors='ignore')
    merged_df = pd.merge(df_clusters, A_excluded, on="Item Code", how="left")
    merged_df2 = pd.merge(merged_df, tracking, on="Item Code", how="left")

    # Recycled logic (simplified)
    A_raw = A.copy()
    B_raw = B.copy()
    st_item_raw = st_item.copy()
    A_recycled = A_raw[A_raw.get("Recycled", "") == "Sì"].copy()
    st_item_recycled = st_item_raw.merge(A_recycled, on="Item Code", how='inner')
    df_recycled = st_item_recycled.copy()
    for func in df_recycled["Function"].unique():
        st_percentiles = st_item_raw[st_item_raw["Function"] == func]["ST item"].quantile([0.25, 0.5, 0.75])
        def categorize(row):
            if row["ST item"] <= st_percentiles.loc[0.25]:
                return 'Basso'
            elif row["ST item"] <= st_percentiles.loc[0.5]:
                return 'Medio Basso'
            elif row["ST item"] <= st_percentiles.loc[0.75]:
                return 'Medio Alto'
            else:
                return 'Alto'
        df_recycled.loc[df_recycled["Function"] == func, "ST_Cluster"] = df_recycled[df_recycled["Function"] == func].apply(categorize, axis=1)

    B_raw["YearWeek"] = B_raw["YearWeek"].astype(str).apply(remove_leading_zero)
    recycled_items = set(df_recycled["Item Code"].astype(str))
    B_recycled = B_raw[B_raw['Item Code'].astype(str).isin(recycled_items)].copy()

    for index, row in df_recycled.iterrows():
        item = row["Item Code"]
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
        p2w_data = B_recycled.loc[(B_recycled["Item Code"].astype(str) == str(item)) & (B_recycled['YearWeek'] == week_p2w), "Delta ST PW"]
        p3w_data = B_recycled.loc[(B_recycled["Item Code"].astype(str) == str(item)) & (B_recycled['YearWeek'] == week_p3w), "Delta ST PW"]
        p2w = p2w_data.values[0] if not p2w_data.empty else None
        p3w = p3w_data.values[0] if not p3w_data.empty else None
        value_p2w = format_percent(p2w)
        value_p3w = format_percent(p3w)
        mask_function = (goals['Function'] == row["Function"])
        if not goals.loc[mask_function].empty:
            goal_row = goals.loc[mask_function].iloc[0]
            theoretical_increase = goal_row['Teorethical Increase %']
            num_life_weeks = goal_row['NumLifeWeeks']
            if num_life_weeks == -1:
                threshold = 0.025
            else:
                threshold = 0.75 * theoretical_increase
        else:
            threshold = 0.0196 
            theoretical_increase = 0.0196
        cluster = row["ST_Cluster"]
        # simplified proposal logic for recycled
        if p2w is not None and p2w > theoretical_increase * 1.25:
            proposal = "Nessuno Sconto"
        else:
            if cluster == "Basso":
                if (p2w is not None and p2w < threshold) or (p3w is not None and p3w < threshold):
                    proposal = "Sconto Alto"
                else:
                    proposal = "Sconto Medio"
            elif cluster == "Medio Basso":
                if (p2w is not None and p2w < threshold) or (p3w is not None and p3w < threshold):
                    proposal = "Sconto Medio"
                else:
                    proposal = "Sconto Basso"
            elif cluster in ["Alto", "Medio Alto"]:
                if p2w is not None and p2w < threshold:
                    if p3w is not None and p3w < threshold:
                        proposal = "Sconto Basso"
                    else:
                        proposal = "Nessuno Sconto"
                else:
                    proposal = "Nessuno Sconto"
        if row.get("Weeks since First Sale Date", 0) < 10:
            proposal = "NESSUNA PROPOSTA (articolo rico con prima vendita troppo recente)"
        df_recycled.at[index, "Proposal"] = proposal
        df_recycled.at[index, "Delta ST P2W"] = value_p2w
        df_recycled.at[index, "Delta ST P3W"] = value_p3w

    df_recycled['Metodo Cluster'] = "Cluster funzione (articolo ricondizionato)"
    # add recycled placeholders
    for col in ['% Store with Tracking', '% Stores with Tracking within 6 weeks', 'First Tracking YearWeek', 'Intake Quantity', 'Displayed Quantity', 'Total Item Tracked', 'First Planned Tracking YearWeek']:
        df_recycled[col] = df_recycled.get(col, "-")

    merged_df2 = pd.concat([merged_df2, df_recycled], ignore_index=True)

    # Compute averages and differences
    merged_df2['AVG ST Function per CommercialMonth'] = merged_df2.groupby(["Function", "Commercial YearMonth"])['ST item'].transform('mean').round(4)
    merged_df2['AVG ST Function'] = merged_df2.groupby(["Function"])['ST item'].transform('mean').round(4)
    merged_df2['ST Difference'] = (merged_df2['ST item'].round(4) - merged_df2.groupby(["Function", "Commercial YearMonth"])['ST item'].transform('mean')).round(4)
    condition = merged_df2["Metodo Cluster"] == "Cluster funzione/mese commerciale"
    merged_df2["ST Difference"] = (
        merged_df2["ST item"].round(4) - 
        merged_df2.groupby(["Function", "Commercial YearMonth"])["ST item"].transform("mean").round(4)
    ).where(condition,
        merged_df2["ST item"].round(4) - 
        merged_df2.groupby(["Function"])["ST item"].transform("mean").round(4)
    )

    # Replace zeros / format
    cols_da_sostituire = [col for col in merged_df2.columns if col not in ["Delta ST P2W", "Delta ST P3W"]]
    merged_df2[cols_da_sostituire] = merged_df2[cols_da_sostituire].replace({0: "-", -1: "Nessuna Vendita"})

    # Numeric conversions
    cols_to_numeric = ['ST item', 'AVG ST Function', 'Sales item', '% Store with Tracking', 'Retail Price', 'Displayed Quantity']
    for col in cols_to_numeric:
        if col in merged_df2.columns:
            merged_df2[col] = pd.to_numeric(merged_df2[col], errors='coerce').fillna(0)

    merged_df2['Stock residuo'] = merged_df2['Delivered item'] - merged_df2['Sales item']
    merged_df2['Intake Quantity'] = pd.to_numeric(merged_df2.get('Intake Quantity', 0), errors='coerce')
    merged_df2['Displayed Quantity'] = pd.to_numeric(merged_df2.get('Displayed Quantity', 0), errors='coerce')
    merged_df2['Total Item Tracked'] = merged_df2['Displayed Quantity'] / merged_df2['Intake Quantity'].replace(0, np.nan)

    # SVA and Sconto proposto
    perc_basso = 0.2
    perc_medio = 0.3
    perc_alto = 0.5
    merged_df2['SVA'] = merged_df2.apply(lambda row: 
        row['Stock residuo'] * perc_basso if row.get('Proposal', '') == "Sconto Basso" else
        row['Stock residuo'] * perc_medio if row.get('Proposal', '') == "Sconto Medio" else
        row['Stock residuo'] * perc_alto if row.get('Proposal', '') == "Sconto Alto" else 0,
        axis=1
    )
    merged_df2['Sconto proposto'] = merged_df2.apply(lambda row: "SI" if row.get('Proposal','') in ["Sconto Basso","Sconto Medio","Sconto Alto"] else "NO", axis=1)

    # Merge sequenza
    if not sequenza.empty:
        merged_df2 = pd.merge(merged_df2, sequenza, on="Item Code", how="left")
        merged_df2['Tipologia sconto applicato'] = merged_df2['Tipologia sconto applicato'].fillna('-')
        merged_df2['Settimana applicazione sconto'] = merged_df2['Settimana applicazione sconto'].fillna('-')
        merged_df2['ST alla settimana di applicazione dello sconto'] = merged_df2['ST alla settimana di applicazione dello sconto'].fillna('-')
        merged_df2.loc[merged_df2["Settimana applicazione sconto"] != "-", "Delta ST previsto"] = "-"

    # Image classification (optional)
    if not images_df.empty and image_model_file is not None:
        st.info("Eseguo predizioni immagine (potrebbe essere lenta)")
        try:
            # Load image model (from upload or path)
            if hasattr(image_model_file, 'read'):
                # save to temp file first
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.keras')
                tmp.write(image_model_file.read())
                tmp.flush()
                image_model = load_model(tmp.name)
            else:
                image_model = load_model(str(image_model_file))

            # attach image URLs to merged_df2
            if 'Item Code' in merged_df2.columns and 'Item Code' in images_df.columns:
                merged_df2 = pd.merge(merged_df2, images_df[['Item Code','Picture']], on='Item Code', how='left')
                merged_df2.rename(columns={'Picture':'Image URL'}, inplace=True)
                merged_df2['Image URL'] = merged_df2['Image URL'].fillna('URL non presente')

            # download and preprocess images in parallel
            session = requests.Session()
            def preprocess_image_bytes(b):
                img = Image.open(BytesIO(b)).convert('RGB')
                img.thumbnail((224,224), Image.LANCZOS)
                new_img = Image.new('RGB', (224,224))
                left = (224 - img.size[0]) // 2
                top = (224 - img.size[1]) // 2
                new_img.paste(img, (left, top))
                arr = np.array(new_img)/255.0
                return arr

            results = {}
            urls = merged_df2.get('Image URL', pd.Series())
            indices = merged_df2.index.tolist()
            # prepare download tasks
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {}
                for idx in indices:
                    url = str(merged_df2.at[idx, 'Image URL']) if 'Image URL' in merged_df2.columns else 'URL non presente'
                    if url == 'URL non presente' or url is None:
                        continue
                    futures[executor.submit(lambda u: session.get(u, timeout=30), url)] = idx
                progress = st.progress(0)
                completed = 0
                total = len(futures)
                images_for_pred = []
                idx_for_pred = []
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        resp = future.result()
                        if resp.status_code == 200:
                            arr = preprocess_image_bytes(resp.content)
                            images_for_pred.append(arr)
                            idx_for_pred.append(idx)
                    except Exception as e:
                        pass
                    completed += 1
                    progress.progress(int(completed/total * 100) if total>0 else 100)

                if images_for_pred:
                    batch = np.stack(images_for_pred, axis=0)
                    st.info("Generazione predizioni immagini...")
                    preds = image_model.predict(batch)
                    pred_dict = {}
                    for i, p in enumerate(preds):
                        cat = merged_df2.at[idx_for_pred[i], 'Cod Category'] if 'Cod Category' in merged_df2.columns else None
                        pred_value = float(p[0])
                        if cat == 31:
                            soglia = 0.57
                        elif cat == 32:
                            soglia = 0.70
                        else:
                            soglia = 0.75
                        pred_dict[idx_for_pred[i]] = "Potenzialmente impattante" if pred_value >= soglia else "Potenzialmente non impattante"

                    merged_df2['Discount Prediction'] = merged_df2.index.map(lambda idx: pred_dict.get(idx, 'Prediction not available'))
                else:
                    merged_df2['Discount Prediction'] = 'Prediction not available'
        except Exception as e:
            st.error(f"Errore predizione immagini: {e}")
    else:
        merged_df2['Discount Prediction'] = 'No image/model provided'

    # Forecast model (optional)
    if forecast_model_file is not None:
        st.info("Carico modello previsionale e genero Delta ST previsto")
        try:
            if hasattr(forecast_model_file, 'read'):
                tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
                tmpf.write(forecast_model_file.read())
                tmpf.flush()
                loaded_model = joblib.load(tmpf.name)
            else:
                loaded_model = joblib.load(str(forecast_model_file))

            # recreate features
            merged_df2 = merged_df2.replace('-', 0).replace('Nessuna Vendita', 0)
            merged_df2['ST_x_AVG_ST'] = merged_df2.get('ST item',0) * merged_df2.get('AVG ST Function',0)
            merged_df2['ST_x_Tracking'] = merged_df2.get('ST item',0) * merged_df2.get('% Store with Tracking',0)
            merged_df2['Sales_x_ST'] = merged_df2.get('Sales item',0) * merged_df2.get('ST item',0)
            merged_df2['Price_x_ST'] = merged_df2.get('Retail Price',0) * merged_df2.get('ST item',0)
            merged_df2['ST_item_squared'] = merged_df2.get('ST item',0) ** 2
            merged_df2['Sales_item_squared'] = merged_df2.get('Sales item',0) ** 2
            merged_df2['Tracking_squared'] = merged_df2.get('% Store with Tracking',0) ** 2
            merged_df2['ST_to_Sales_Ratio'] = merged_df2['ST item'] / merged_df2['Sales item'].replace(0, 0.001)
            merged_df2['Tracking_to_Display_Ratio'] = merged_df2['% Store with Tracking'] / merged_df2['Displayed Quantity'].replace(0, 0.001)

            X_input = merged_df2.copy().replace("-", 0)
            merged_df2['Delta ST previsto'] = loaded_model.predict(X_input)

            # drop temporary features
            merged_df2.drop([
                'ST_x_AVG_ST','ST_x_Tracking','Sales_x_ST','Price_x_ST','ST_item_squared',
                'Sales_item_squared','Tracking_squared','ST_to_Sales_Ratio','Tracking_to_Display_Ratio'
            ], axis=1, inplace=True, errors='ignore')

        except Exception as e:
            st.error(f"Errore caricamento modello previsionale: {e}")
    else:
        merged_df2['Delta ST previsto'] = 'Model not provided'

    # Final filters
    try:
        merged_df2 = merged_df2[merged_df2.get('Delivered item',0) >= min_delivered]
    except Exception:
        pass

    # Format TFI
    if 'Function' in merged_df2.columns and 'Teorethical Increase %' in goals.columns:
        mapping_tfi = goals.set_index('Function')['Teorethical Increase %'].apply(lambda x: f"{x*100:.2f}%".replace('.',',')) if not goals.empty else {}
        merged_df2['TFI'] = merged_df2['Function'].map(mapping_tfi).fillna('1,96%')

    # Add processing date
    elaboration_date = datetime.today().strftime('%d-%m-%Y')
    merged_df2['Data elaborazione'] = elaboration_date

    st.success("Elaborazione completata")

    # Show sample
    st.subheader("Anteprima risultato (prime 200 righe)")
    st.dataframe(merged_df2.head(200))

    # Save to Excel and provide download
    tmpfile = BytesIO()
    try:
        # choose file name based on category
        if category_val == 31:
            fname = f"IC_proposte_sconti_WOMAN_{elaboration_date}.xlsx"
        elif category_val == 32:
            fname = f"IC_proposte_sconti_MAN_{elaboration_date}.xlsx"
        elif category_val == 33:
            fname = f"IC_proposte_sconti_KIDS_{elaboration_date}.xlsx"
        else:
            fname = f"IC_proposte_sconti_{elaboration_date}.xlsx"

        # write to excel
        merged_df2.to_excel(tmpfile, index=False)
        tmpfile.seek(0)
        st.download_button(label="Download Excel", data=tmpfile, file_name=fname, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Also apply openpyxl formatting and offer a second download (saved to temp file)
        tmpfile.seek(0)
        wb = load_workbook(tmpfile)
        ws = wb.active
        header_config = {
            "Proposal": {"round": 4, "num_format": '0.0000', "font": Font(bold=True), "fill": PatternFill(start_color="EBF1DE", end_color="EBF1DE", fill_type="solid")},
            "ST Difference": {"round": 4, "num_format": '0.0000', "font": Font(bold=True), "fill": PatternFill(start_color="E4DFEC", end_color="E4DFEC", fill_type="solid")},
            "ST item": {"round": 4, "num_format": '0.0000', "font": Font(bold=True), "fill": PatternFill(start_color="DAEEF3", end_color="DAEEF3", fill_type="solid")},
            "Sales item": {"round": 2, "num_format": '0.00'},
            "Delivered item": {"round": 2, "num_format": '0.00'},
            "SVA": {"round": 2, "num_format": '0.00'},
            "Stock residuo": {"round": 2, "num_format": '0.00'},
            "Total Item Tracked": {"round": 4, "num_format": '0.0000'},
            "TFI": {"fill": PatternFill(start_color="EFF7FF", end_color="EFF7FF", fill_type="solid")},
            "Delta ST P2W": {"fill": PatternFill(start_color="EFF7FF", end_color="EFF7FF", fill_type="solid")},
            "Delta ST P3W": {"fill": PatternFill(start_color="EFF7FF", end_color="EFF7FF", fill_type="solid")}
        }
        # Apply simple formatting: round numeric cols when present
        header_columns = {}
        for cell in ws[1]:
            if cell.value in header_config:
                header_columns[cell.value] = cell.column
        for header, config in header_config.items():
            if header not in header_columns:
                continue
            col_idx = header_columns[header]
            for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=col_idx, max_col=col_idx):
                for cell in row:
                    if cell.value == "-":
                        cell.value = 0
                    if isinstance(cell.value, (int, float)) and "round" in config:
                        cell.value = round(cell.value, config["round"]) 
                        try:
                            cell.number_format = config["num_format"]
                        except:
                            pass
                    if "font" in config:
                        cell.font = config["font"]
                    if "fill" in config:
                        cell.fill = config["fill"]

        tmp2 = BytesIO()
        wb.save(tmp2)
        tmp2.seek(0)
        st.download_button(label="Download Excel formattato (openpyxl)", data=tmp2, file_name=f"formatted_{fname}", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    except Exception as e:
        st.error(f"Errore salvataggio excel: {e}")

    st.info("Fine esecuzione. Controlla i download nella barra laterale.")
