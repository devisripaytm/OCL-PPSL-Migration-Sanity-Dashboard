"""
Migration Sanity Validation Dashboard
A comprehensive Streamlit dashboard for visualizing dataset migration validation results.
Supports both Google Sheets auto-loading (with service account) and manual CSV upload.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import re
from datetime import datetime

# Google Sheets authentication
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Migration Sanity Validation Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# GOOGLE SHEETS CONFIGURATION
# =============================================================================
# Replace with your Google Sheet ID (from the URL)
# URL format: https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit
GOOGLE_SHEET_ID = "1qDLLxbEpiJUv5Q6QAsUSOfi4X0FUxRQptsexh22Y5cc"

# Tab names in the Google Sheet (must match exactly)
SHEET_TABS = {
    "ok": "ok_datasets",
    "partial_ok": "partial_ok_datasets",
    "not_ok": "not_ok_datasets",
    "summary": "not_ok_summary"
}

# Cache duration in seconds (300 = 5 minutes)
CACHE_TTL = 300

# =============================================================================
# COLOR SCHEME
# =============================================================================
COLORS = {
    "ok": "#10B981",           # Emerald green
    "partial_ok": "#F59E0B",   # Amber (OK but need attention)
    "not_ok": "#EF4444",       # Red
    "primary": "#3B82F6",      # Blue
    "secondary": "#6B7280",    # Gray
    "background": "#1F2937",   # Dark gray
    "surface": "#374151",      # Medium gray
    "text": "#F9FAFB"          # Light gray
}

CASE_COLORS = px.colors.qualitative.Set3

# Case Descriptions for each category
OK_CASE_DESCRIPTIONS = {
    "Case 1": "FULL, max date match, status match",
    "Case 2": "Not FULL, max date match, status match"
}

# Renamed from PARTIAL_OK to OK_BUT_NEED_ATTENTION
OK_BUT_NEED_ATTENTION_CASE_DESCRIPTIONS = {
    "Case 1": "Not FULL, only last date not matching",
    "Case 2": "FULL, status mismatch, count match",
    "Case 3": "Not FULL, status mismatch, count match",
    "Case 4": "Not FULL, status mismatch, last date",
    "Case 5": "Max date mismatch, last date not match",
    "Case 6": "Zero counts (0=0), no data in S3"
}

# Next Steps Suggested for "OK but need attention" cases
OK_BUT_NEED_ATTENTION_NEXT_STEPS = {
    "Case 1": "NA. Acceptable.",
    "Case 2": "Manual checking required. Make status same at both places (Inactive at new OCL).",
    "Case 3": "Manual checking required. Make status same at both places (Inactive at new OCL).",
    "Case 4": "Not possible. Manual check. P0. May need to inactive dataset at one place.",
    "Case 5": "Manual check case.",
    "Case 6": "No counts in both. Dependent on backend engineering to check S3 manifest."
}

# Keep old name for backward compatibility
PARTIAL_OK_CASE_DESCRIPTIONS = OK_BUT_NEED_ATTENTION_CASE_DESCRIPTIONS

NOT_OK_CASE_DESCRIPTIONS = {
    "Case 1": "FULL, count mismatch",
    "Case 2": "Not FULL, count mismatch",
    "Case 3": "FULL, status & count mismatch",
    "Case 4": "Not FULL, status & count mismatch",
    "Case 5": "Max date mismatch, FULL, count match",
    "Case 6": "Max date mismatch, FULL, count mismatch",
    "Case 7": "Max date mismatch, Not FULL, count match",
    "Case 8": "Max date mismatch, Not FULL, mismatch",
    "Case 9": "Max date & status mismatch, FULL, match",
    "Case 10": "Max date & status mismatch, FULL",
    "Case 11": "Max date & status mismatch, Not FULL",
    "Case 12": "Max date & status mismatch, last date",
    "Case 13": "Max date & status mismatch, full mismatch"
}

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d1b2a 100%);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1e293b, #334155);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 8px 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 8px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(30, 41, 59, 0.8);
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #F9FAFB !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: rgba(51, 65, 85, 0.5);
        border-radius: 12px;
        padding: 16px;
        border: 2px dashed rgba(59, 130, 246, 0.5);
    }
    
    /* Success/warning badges */
    .success-badge {
        background-color: #10B981;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .warning-badge {
        background-color: #F59E0B;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .error-badge {
        background-color: #EF4444;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* DataFrame styling */
    .dataframe {
        font-size: 0.85rem;
    }
    
    /* Filter section */
    .filter-section {
        background-color: rgba(30, 41, 59, 0.6);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #3B82F6, #8B5CF6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(90deg, #2563EB, #7C3AED);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# GOOGLE SHEETS DATA LOADING FUNCTIONS (Service Account Authentication)
# =============================================================================

def get_gspread_client():
    """
    Get authenticated gspread client using service account credentials from Streamlit secrets.
    
    Returns:
        gspread.Client or None if authentication fails
    """
    if not GSPREAD_AVAILABLE:
        st.warning("⚠️ gspread library not available. Please install: pip install gspread google-auth")
        return None
    
    try:
        # Check if credentials are in Streamlit secrets
        if "gcp_service_account" not in st.secrets:
            st.warning("⚠️ 'gcp_service_account' not found in secrets. Please configure secrets in Streamlit Cloud.")
            return None
        
        # Define scopes
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly"
        ]
        
        # IMPORTANT: Convert Streamlit secrets object to a plain dictionary
        # st.secrets returns a special object that needs to be converted for google-auth
        creds_dict = dict(st.secrets["gcp_service_account"])
        
        # Create credentials from secrets
        credentials = Credentials.from_service_account_info(
            creds_dict,
            scopes=scopes
        )
        
        # Authorize gspread client
        client = gspread.authorize(credentials)
        return client
    
    except KeyError as e:
        st.error(f"❌ Missing key in service account credentials: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Authentication error: {type(e).__name__}: {str(e)}")
        return None


@st.cache_data(ttl=CACHE_TTL)
def load_sheet_data_authenticated(_client, sheet_id: str, tab_name: str) -> pd.DataFrame:
    """
    Load data from a specific Google Sheets tab using authenticated gspread client.
    
    Args:
        _client: Authenticated gspread client (underscore prefix to exclude from cache key)
        sheet_id: The Google Sheet ID from the URL
        tab_name: The name of the tab/worksheet to load
    
    Returns:
        DataFrame with the sheet data, or empty DataFrame on error
    """
    try:
        # Open the spreadsheet by ID
        spreadsheet = _client.open_by_key(sheet_id)
        
        # Get the specific worksheet
        worksheet = spreadsheet.worksheet(tab_name)
        
        # Get all records as a list of dicts
        records = worksheet.get_all_records()
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        return df
    
    except gspread.exceptions.WorksheetNotFound:
        # Tab doesn't exist - return empty DataFrame
        st.warning(f"⚠️ Worksheet '{tab_name}' not found in the spreadsheet")
        return pd.DataFrame()
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"❌ Spreadsheet not found. Check Sheet ID: {sheet_id[:20]}...")
        return pd.DataFrame()
    except gspread.exceptions.APIError as e:
        st.error(f"❌ Google Sheets API error for '{tab_name}': {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading '{tab_name}': {type(e).__name__}: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL)
def load_sheet_data_public(sheet_id: str, tab_name: str) -> pd.DataFrame:
    """
    Load data from a public Google Sheets tab using CSV export URL.
    Fallback method when service account is not configured.
    
    Args:
        sheet_id: The Google Sheet ID from the URL
        tab_name: The name of the tab/worksheet to load
    
    Returns:
        DataFrame with the sheet data, or empty DataFrame on error
    """
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={tab_name}"
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        return pd.DataFrame()


def load_all_sheets_data(sheet_id: str) -> tuple:
    """
    Load all 4 sheets from Google Sheets.
    Uses service account authentication if available, falls back to public URL.
    
    Returns:
        Tuple of (df_ok, df_partial, df_not_ok, df_summary)
    """
    # Try authenticated access first
    client = get_gspread_client()
    
    if client:
        st.sidebar.success("✅ Connected to Google Sheets API")
        # Use authenticated access
        df_ok = load_sheet_data_authenticated(client, sheet_id, SHEET_TABS["ok"])
        df_partial = load_sheet_data_authenticated(client, sheet_id, SHEET_TABS["partial_ok"])
        df_not_ok = load_sheet_data_authenticated(client, sheet_id, SHEET_TABS["not_ok"])
        df_summary = load_sheet_data_authenticated(client, sheet_id, SHEET_TABS["summary"])
    else:
        st.sidebar.warning("⚠️ Using public URL fallback (no auth)")
        # Fallback to public URL method
        df_ok = load_sheet_data_public(sheet_id, SHEET_TABS["ok"])
        df_partial = load_sheet_data_public(sheet_id, SHEET_TABS["partial_ok"])
        df_not_ok = load_sheet_data_public(sheet_id, SHEET_TABS["not_ok"])
        df_summary = load_sheet_data_public(sheet_id, SHEET_TABS["summary"])
    
    return df_ok, df_partial, df_not_ok, df_summary


def check_sheets_configured() -> bool:
    """Check if Google Sheets is properly configured (ID set and credentials available)."""
    id_configured = GOOGLE_SHEET_ID != "your-google-sheet-id-here" and len(GOOGLE_SHEET_ID) > 10
    
    # Check if service account credentials are available
    has_credentials = "gcp_service_account" in st.secrets if hasattr(st, 'secrets') else False
    
    return id_configured and (has_credentials or GSPREAD_AVAILABLE)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_case_number(sanity_reason):
    """Extract case number from sanity reason string."""
    if pd.isna(sanity_reason) or str(sanity_reason).strip() == '':
        return None  # Return None to filter out later
    reason_str = str(sanity_reason)
    # Try to find "Case X" pattern
    match = re.search(r'Case\s*(\d+)', reason_str, re.IGNORECASE)
    if match:
        return f"Case {match.group(1)}"
    # If no case pattern found, skip this row
    return None


def get_unique_dataset_count(df, id_column="Dataset ID"):
    """Get unique dataset count from dataframe."""
    if df is None or df.empty:
        return 0
    return df[id_column].nunique()


def create_metric_card(icon, label, value, color, percentage=None):
    """Create a styled metric card with optional percentage."""
    if percentage is not None:
        percentage_html = f'<div style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">({percentage:.1f}%)</div>'
    else:
        percentage_html = ''
    return f'<div class="metric-card"><div class="metric-icon">{icon}</div><div class="metric-value" style="color: {color};">{value:,}</div>{percentage_html}<div class="metric-label">{label}</div></div>'


def create_percentage_card(icon, label, value, color):
    """Create a styled percentage metric card."""
    return f'<div class="metric-card"><div class="metric-icon">{icon}</div><div class="metric-value" style="color: {color};">{value:.1f}%</div><div class="metric-label">{label}</div></div>'


@st.cache_data
def convert_df_to_csv(df):
    """Convert DataFrame to CSV for download."""
    return df.to_csv(index=False).encode('utf-8')


def convert_dfs_to_excel(dfs_dict):
    """Convert multiple DataFrames to Excel with multiple sheets."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in dfs_dict.items():
            if df is not None and not df.empty:
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    output.seek(0)
    return output


def apply_filters(df, filters_dict):
    """Apply multiple filters to a DataFrame."""
    filtered_df = df.copy()
    for column, values in filters_dict.items():
        if values and column in filtered_df.columns:
            if isinstance(values, list):
                filtered_df = filtered_df[filtered_df[column].isin(values)]
            else:
                filtered_df = filtered_df[filtered_df[column] == values]
    return filtered_df


def search_dataset(df, search_term, id_column="Dataset ID"):
    """Search for dataset by ID."""
    if not search_term:
        return df
    search_term = str(search_term).strip()
    return df[df[id_column].astype(str).str.contains(search_term, case=False, na=False)]


def get_sanity_run_date(df):
    """Extract sanity run date from DataFrame."""
    date_cols = ['sanity run date', 'sanity_run_date', 'Sanity run date']
    for col in date_cols:
        if col in df.columns:
            dates = df[col].dropna()
            if not dates.empty:
                return dates.iloc[0]
    return "N/A"


def get_case_distribution(df, reason_column=None, case_descriptions=None):
    """Get case-wise count and percentage distribution."""
    if df is None or df.empty:
        return None
    
    # Find the reason column
    if reason_column is None:
        possible_cols = ['Sanity reason', 'sanity reason', 'sanity_reason', 'Sanity Reason']
        for col in possible_cols:
            if col in df.columns:
                reason_column = col
                break
    
    if reason_column is None or reason_column not in df.columns:
        return None
    
    df_copy = df.copy()
    df_copy['Case'] = df_copy[reason_column].apply(extract_case_number)
    df_copy = df_copy[df_copy['Case'].notna()]
    
    if df_copy.empty:
        return None
    
    # Get unique dataset counts per case
    case_counts = df_copy.groupby('Case')['Dataset ID'].nunique().reset_index()
    case_counts.columns = ['Case', 'Count']
    case_counts = case_counts.sort_values('Case', key=lambda x: x.str.extract(r'(\d+)')[0].astype(int))
    
    total = case_counts['Count'].sum()
    case_counts['Percentage'] = (case_counts['Count'] / total * 100).round(1)
    
    # Add descriptions if provided
    if case_descriptions:
        case_counts['Description'] = case_counts['Case'].map(case_descriptions).fillna('')
    
    return case_counts, total


def get_success_rate_by_run(df_ok, df_partial, df_not_ok):
    """Calculate success rate for each sanity run date."""
    all_data = []
    date_col_name = None
    
    for df, status in [(df_ok, 'OK'), (df_partial, 'OK but need attention'), (df_not_ok, 'Not OK')]:
        if df is None or df.empty:
            continue
        
        # Find date column
        date_col = None
        for col in ['sanity run date', 'sanity_run_date', 'Sanity run date']:
            if col in df.columns:
                date_col = col
                date_col_name = col
                break
        
        if date_col:
            df_temp = df[['Dataset ID', date_col]].copy()
            df_temp['validation_status'] = status
            df_temp = df_temp.drop_duplicates(subset=['Dataset ID'])
            all_data.append(df_temp)
    
    if not all_data:
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    
    if date_col_name is None:
        return None
    
    # Group by sanity run date
    run_summary = combined.groupby(date_col_name).apply(
        lambda x: pd.Series({
            'Total': x['Dataset ID'].nunique(),
            'OK': (x['validation_status'] == 'OK').sum(),
            'OK but need attention': (x['validation_status'] == 'OK but need attention').sum(),
            'Not OK': (x['validation_status'] == 'Not OK').sum()
        })
    ).reset_index()
    
    run_summary.columns = ['Sanity Run Date', 'Total', 'OK', 'OK but need attention', 'Not OK']
    run_summary['Success Rate (%)'] = (run_summary['OK'] / run_summary['Total'] * 100).round(2)
    run_summary = run_summary.sort_values('Sanity Run Date', ascending=False)
    
    return run_summary


def create_case_distribution_table(case_data, total, title):
    """Create a styled case distribution section with table and chart."""
    if case_data is None:
        return
    
    st.markdown(f"#### 📊 {title}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Display table
        display_df = case_data.copy()
        display_df['Percentage'] = display_df['Percentage'].apply(lambda x: f"{x}%")
        
        # Add total row
        total_row = pd.DataFrame({
            'Case': ['**Total**'],
            'Description': [''] if 'Description' in display_df.columns else None,
            'Count': [total],
            'Percentage': ['100%']
        })
        if 'Description' not in display_df.columns:
            total_row = total_row.drop('Description', axis=1)
        
        display_df = pd.concat([display_df, total_row], ignore_index=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    with col2:
        # Bar chart
        fig = px.bar(
            case_data,
            x='Case',
            y='Count',
            color='Case',
            color_discrete_sequence=CASE_COLORS,
            text='Count'
        )
        fig.update_traces(textposition='outside', textfont_size=11)
        fig.update_layout(
            title=dict(text="Case Distribution", font=dict(size=14, color='white')),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(title="", gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title="Count", gridcolor='rgba(255,255,255,0.1)'),
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)


def create_case_distribution_table_with_next_steps(case_data, total, title):
    """Create a styled case distribution section with table, chart, and Next Steps column."""
    if case_data is None:
        return
    
    st.markdown(f"#### 📊 {title}")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Display table with Next Steps
        display_df = case_data.copy()
        display_df['Percentage'] = display_df['Percentage'].apply(lambda x: f"{x}%")
        
        # Reorder columns to show Next Steps
        cols_order = ['Case', 'Description', 'Count', 'Percentage', 'Next Steps']
        cols_order = [c for c in cols_order if c in display_df.columns]
        display_df = display_df[cols_order]
        
        # Add total row
        total_row_data = {'Case': ['**Total**'], 'Count': [total], 'Percentage': ['100%']}
        if 'Description' in display_df.columns:
            total_row_data['Description'] = ['']
        if 'Next Steps' in display_df.columns:
            total_row_data['Next Steps'] = ['']
        total_row = pd.DataFrame(total_row_data)
        
        display_df = pd.concat([display_df, total_row], ignore_index=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    with col2:
        # Bar chart
        fig = px.bar(
            case_data,
            x='Case',
            y='Count',
            color='Case',
            color_discrete_sequence=CASE_COLORS,
            text='Count'
        )
        fig.update_traces(textposition='outside', textfont_size=11)
        fig.update_layout(
            title=dict(text="Case Distribution", font=dict(size=14, color='white')),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(title="", gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title="Count", gridcolor='rgba(255,255,255,0.1)'),
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)


def create_success_rate_chart(run_history):
    """Create line chart for success rate trend."""
    if run_history is None or run_history.empty:
        return None
    
    fig = px.line(
        run_history,
        x='Sanity Run Date',
        y='Success Rate (%)',
        markers=True,
        color_discrete_sequence=[COLORS['primary']]
    )
    
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=10)
    )
    
    fig.update_layout(
        title=dict(text="Success Rate Over Sanity Runs", font=dict(size=18, color='white')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(title="Sanity Run Date", gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title="Success Rate (%)", gridcolor='rgba(255,255,255,0.1)', range=[0, 100])
    )
    
    return fig


# =============================================================================
# SOURCE TYPE BREAKDOWN FUNCTIONS
# =============================================================================

def get_source_type_column(df):
    """Find the source type column in dataframe."""
    possible_cols = ['old_source_type', 'Old Source Type', 'old source type', 'source_type', 'Source Type']
    for col in possible_cols:
        if col in df.columns:
            return col
    return None


def create_source_type_overview_table(df_ok, df_partial, df_not_ok):
    """Create a table showing counts by source type across OK but need attention and Not OK categories."""
    source_data = {}
    
    # Only include OK but need attention and Not OK (exclude OK)
    for df, category in [(df_partial, 'OK but need attention'), (df_not_ok, 'Not OK')]:
        if df is None or df.empty:
            continue
        
        source_col = get_source_type_column(df)
        if source_col is None:
            continue
        
        # Get unique dataset counts per source type
        df_copy = df.copy()
        df_copy[source_col] = df_copy[source_col].fillna('Unknown').astype(str)
        source_counts = df_copy.groupby(source_col)['Dataset ID'].nunique()
        
        for source_type, count in source_counts.items():
            if source_type not in source_data:
                source_data[source_type] = {'OK but need attention': 0, 'Not OK': 0}
            source_data[source_type][category] = count
    
    if not source_data:
        return None
    
    # Create DataFrame
    rows = []
    for source_type, counts in source_data.items():
        rows.append({
            'Source Type': source_type,
            'OK but need attention': counts['OK but need attention'],
            'Not OK': counts['Not OK'],
            'Total': counts['OK but need attention'] + counts['Not OK']
        })
    
    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values('Total', ascending=False)
    
    # Add total row
    total_row = pd.DataFrame([{
        'Source Type': '**Total**',
        'OK but need attention': result_df['OK but need attention'].sum(),
        'Not OK': result_df['Not OK'].sum(),
        'Total': result_df['Total'].sum()
    }])
    
    result_df = pd.concat([result_df, total_row], ignore_index=True)
    
    return result_df


def create_source_type_pie_chart(df, title, color_scheme=None):
    """Create a colorful pie chart for source type distribution."""
    if df is None or df.empty:
        return None
    
    source_col = get_source_type_column(df)
    if source_col is None:
        return None
    
    df_copy = df.copy()
    df_copy[source_col] = df_copy[source_col].fillna('Unknown').astype(str)
    
    # Get unique dataset counts per source type
    source_counts = df_copy.groupby(source_col)['Dataset ID'].nunique().reset_index()
    source_counts.columns = ['Source Type', 'Count']
    source_counts = source_counts.sort_values('Count', ascending=False)
    
    if source_counts.empty:
        return None
    
    # Use a colorful palette
    if color_scheme is None:
        color_scheme = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel1 + px.colors.qualitative.Bold
    
    fig = go.Figure(data=[go.Pie(
        labels=source_counts['Source Type'],
        values=source_counts['Count'],
        hole=0.4,
        marker_colors=color_scheme[:len(source_counts)],
        textinfo='label+percent',
        textfont_size=11,
        textfont_color='white',
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>"
    )])
    
    total_count = source_counts['Count'].sum()
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='white')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            font=dict(size=11)
        ),
        annotations=[dict(
            text=f'{total_count:,}<br>Total',
            x=0.5, y=0.5,
            font_size=16,
            font_color='white',
            showarrow=False
        )],
        height=450,
        margin=dict(l=20, r=120, t=60, b=20)
    )
    
    return fig


def create_source_type_case_breakdown(df, case_count, title):
    """Create source type breakdown with case-wise counts."""
    if df is None or df.empty:
        return None
    
    source_col = get_source_type_column(df)
    reason_col = next((c for c in ['Sanity reason', 'sanity reason', 'sanity_reason', 'Sanity Reason'] if c in df.columns), None)
    
    if source_col is None or reason_col is None:
        return None
    
    df_copy = df.copy()
    df_copy[source_col] = df_copy[source_col].fillna('Unknown').astype(str)
    df_copy['Case'] = df_copy[reason_col].apply(extract_case_number)
    df_copy = df_copy[df_copy['Case'].notna()]
    
    if df_copy.empty:
        return None
    
    # Get unique source types
    source_types = df_copy[source_col].unique()
    
    # Build the breakdown
    rows = []
    for source_type in source_types:
        source_df = df_copy[df_copy[source_col] == source_type]
        row = {'Source Type': source_type}
        
        # Count unique datasets per case
        total_datasets = source_df['Dataset ID'].nunique()
        row['Total'] = total_datasets
        
        for i in range(1, case_count + 1):
            case_name = f'Case {i}'
            case_df = source_df[source_df['Case'] == case_name]
            row[case_name] = case_df['Dataset ID'].nunique()
        
        rows.append(row)
    
    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values('Total', ascending=False)
    
    # Calculate % of Total
    grand_total = result_df['Total'].sum()
    result_df['% of Total'] = (result_df['Total'] / grand_total * 100).round(1).astype(str) + '%'
    
    # Add total row
    total_row = {'Source Type': '**Total**', 'Total': grand_total, '% of Total': '100%'}
    for i in range(1, case_count + 1):
        case_name = f'Case {i}'
        if case_name in result_df.columns:
            total_row[case_name] = result_df[case_name].sum()
    
    result_df = pd.concat([result_df, pd.DataFrame([total_row])], ignore_index=True)
    
    # Reorder columns
    cols = ['Source Type', 'Total'] + [f'Case {i}' for i in range(1, case_count + 1)] + ['% of Total']
    cols = [c for c in cols if c in result_df.columns]
    result_df = result_df[cols]
    
    return result_df


# =============================================================================
# CHART FUNCTIONS
# =============================================================================

def create_distribution_pie(ok_count, partial_count, not_ok_count):
    """Create pie chart for OK/OK but need attention/Not OK distribution."""
    fig = go.Figure(data=[go.Pie(
        labels=['OK', 'OK but need attention', 'Not OK'],
        values=[ok_count, partial_count, not_ok_count],
        hole=0.5,
        marker_colors=[COLORS['ok'], COLORS['partial_ok'], COLORS['not_ok']],
        textinfo='label+percent',
        textfont_size=12,
        textfont_color='white',
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>"
    )])
    
    fig.update_layout(
        title=dict(text="Validation Results Distribution", font=dict(size=18, color='white')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        annotations=[dict(
            text=f'{ok_count + partial_count + not_ok_count:,}<br>Total',
            x=0.5, y=0.5,
            font_size=16,
            font_color='white',
            showarrow=False
        )]
    )
    return fig


def create_case_distribution_bar(df, case_column='Sanity reason'):
    """Create bar chart for case distribution with category color coding."""
    # Find the correct column name
    possible_cols = ['Sanity reason', 'sanity reason', 'sanity_reason', 'Sanity Reason']
    col_name = None
    for col in possible_cols:
        if col in df.columns:
            col_name = col
            break
    
    if col_name is None:
        return None
    
    df_copy = df.copy()
    df_copy['Case'] = df_copy[col_name].apply(extract_case_number)
    # Filter out None/null cases (rows without valid case numbers)
    df_copy = df_copy[df_copy['Case'].notna()]
    
    if df_copy.empty:
        return None
    
    # Check if Category column exists (from combined dataframe)
    has_category = 'Category' in df_copy.columns
    
    if has_category:
        # Create case label with category prefix
        def get_case_label(row):
            case = row['Case']
            category = row['Category']
            case_num = case.replace('Case ', '') if case else ''
            if category == 'OK':
                return f"OK-{case_num}"
            elif category == 'OK but need attention':
                return f"Attn-{case_num}"
            else:  # Not OK
                return f"NotOK-{case_num}"
        
        df_copy['CaseLabel'] = df_copy.apply(get_case_label, axis=1)
        
        # Count by CaseLabel and Category
        case_counts = df_copy.groupby(['CaseLabel', 'Category']).size().reset_index(name='Count')
        
        # Define category colors
        category_colors = {
            'OK': COLORS['ok'],
            'OK but need attention': COLORS['partial_ok'],
            'Not OK': COLORS['not_ok']
        }
        
        # Sort by category order and then case number
        category_order = {'OK': 0, 'OK but need attention': 1, 'Not OK': 2}
        case_counts['SortOrder'] = case_counts['Category'].map(category_order)
        case_counts['CaseNum'] = case_counts['CaseLabel'].str.extract(r'(\d+)').astype(int)
        case_counts = case_counts.sort_values(['SortOrder', 'CaseNum'])
        
        fig = px.bar(
            case_counts,
            x='CaseLabel',
            y='Count',
            color='Category',
            color_discrete_map=category_colors,
            text='Count',
            category_orders={'CaseLabel': case_counts['CaseLabel'].tolist()}
        )
        
        fig.update_traces(textposition='outside', textfont_size=11)
        fig.update_layout(
            title=dict(text="Case-wise Distribution by Category", font=dict(size=18, color='white')),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(title="Case", gridcolor='rgba(255,255,255,0.1)', tickangle=-45),
            yaxis=dict(title="Count", gridcolor='rgba(255,255,255,0.1)'),
            legend=dict(
                title="Category",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(0,0,0,0.5)'
            ),
            showlegend=True
        )
    else:
        # Fallback to original behavior if no category column
        case_counts = df_copy['Case'].value_counts().reset_index()
        case_counts.columns = ['Case', 'Count']
        case_counts = case_counts.sort_values('Case')
        
        fig = px.bar(
            case_counts,
            x='Case',
            y='Count',
            color='Case',
            color_discrete_sequence=CASE_COLORS,
            text='Count'
        )
        
        fig.update_traces(textposition='outside', textfont_size=12)
        fig.update_layout(
            title=dict(text="Case-wise Distribution", font=dict(size=18, color='white')),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(title="Case", gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title="Count", gridcolor='rgba(255,255,255,0.1)'),
            showlegend=False
        )
    
    return fig


def create_ingest_type_bar(df_ok, df_partial, df_not_ok):
    """Create bar chart for ingest type distribution."""
    data = []
    
    for df, category in [(df_ok, 'OK'), (df_partial, 'OK but need attention'), (df_not_ok, 'Not OK')]:
        if df is not None and not df.empty:
            # Find ingest type column
            ingest_col = None
            for col in ['Ingestion type', 'Ingest type', 'ingest_type']:
                if col in df.columns:
                    ingest_col = col
                    break
            
            if ingest_col:
                for ingest_type in df[ingest_col].unique():
                    count = len(df[df[ingest_col] == ingest_type]['Dataset ID'].unique())
                    data.append({'Category': category, 'Ingest Type': str(ingest_type), 'Count': count})
    
    if not data:
        return None
    
    chart_df = pd.DataFrame(data)
    
    fig = px.bar(
        chart_df,
        x='Category',
        y='Count',
        color='Ingest Type',
        barmode='group',
        color_discrete_sequence=[COLORS['primary'], COLORS['secondary']],
        text='Count'
    )
    
    fig.update_traces(textposition='outside', textfont_size=11)
    fig.update_layout(
        title=dict(text="Distribution by Ingest Type", font=dict(size=18, color='white')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(title="", gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title="Dataset Count", gridcolor='rgba(255,255,255,0.1)'),
        legend=dict(title="Ingest Type")
    )
    return fig


def create_status_flag_pie(df_partial, df_not_ok):
    """Create pie chart for status flag distribution."""
    status_counts = {'Matching': 0, 'Unmatching': 0}
    
    for df in [df_partial, df_not_ok]:
        if df is not None and not df.empty:
            status_col = None
            for col in ['Status flag', 'status_flag', 'Status Flag']:
                if col in df.columns:
                    status_col = col
                    break
            
            if status_col:
                for status, count in df.groupby(status_col)['Dataset ID'].nunique().items():
                    status_str = str(status).strip()
                    if 'match' in status_str.lower():
                        if 'un' in status_str.lower() or 'not' in status_str.lower():
                            status_counts['Unmatching'] += count
                        else:
                            status_counts['Matching'] += count
    
    if sum(status_counts.values()) == 0:
        return None
    
    fig = go.Figure(data=[go.Pie(
        labels=list(status_counts.keys()),
        values=list(status_counts.values()),
        hole=0.4,
        marker_colors=[COLORS['ok'], COLORS['not_ok']],
        textinfo='label+percent',
        textfont_size=14,
        textfont_color='white'
    )])
    
    fig.update_layout(
        title=dict(text="Status Flag Distribution", font=dict(size=18, color='white')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig


def create_max_date_bar(df_partial, df_not_ok):
    """Create bar chart for max date matching distribution."""
    data = []
    
    for df, category in [(df_partial, 'OK but need attention'), (df_not_ok, 'Not OK')]:
        if df is not None and not df.empty:
            max_date_col = None
            for col in ['Max date not matching', 'max_date_matching', 'Max Date']:
                if col in df.columns:
                    max_date_col = col
                    break
            
            if max_date_col:
                for match_status in df[max_date_col].unique():
                    count = len(df[df[max_date_col] == match_status]['Dataset ID'].unique())
                    data.append({'Category': category, 'Max Date Status': str(match_status), 'Count': count})
    
    if not data:
        return None
    
    chart_df = pd.DataFrame(data)
    
    fig = px.bar(
        chart_df,
        x='Category',
        y='Count',
        color='Max Date Status',
        barmode='group',
        color_discrete_sequence=[COLORS['ok'], COLORS['not_ok']],
        text='Count'
    )
    
    fig.update_traces(textposition='outside', textfont_size=11)
    fig.update_layout(
        title=dict(text="Max Date Matching Status", font=dict(size=18, color='white')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(title="", gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title="Dataset Count", gridcolor='rgba(255,255,255,0.1)'),
        legend=dict(title="Status")
    )
    return fig


def create_data_match_histogram(df_summary):
    """Create histogram for % data match distribution."""
    if df_summary is None or df_summary.empty:
        return None
    
    data_match_col = None
    for col in ['% data match', 'data_match_pct', '% Data Match']:
        if col in df_summary.columns:
            data_match_col = col
            break
    
    if data_match_col is None:
        return None
    
    # Filter out non-numeric and summary rows
    df_plot = df_summary[pd.to_numeric(df_summary[data_match_col], errors='coerce').notna()].copy()
    df_plot[data_match_col] = pd.to_numeric(df_plot[data_match_col])
    
    fig = px.histogram(
        df_plot,
        x=data_match_col,
        nbins=20,
        color_discrete_sequence=[COLORS['primary']]
    )
    
    fig.update_layout(
        title=dict(text="Distribution of % Data Match", font=dict(size=18, color='white')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(title="% Data Match", gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title="Count", gridcolor='rgba(255,255,255,0.1)')
    )
    return fig


def create_match_scatter(df_summary):
    """Create scatter plot for % date match vs % data match."""
    if df_summary is None or df_summary.empty:
        return None
    
    date_match_col = None
    data_match_col = None
    
    for col in ['% date match', 'date_match_pct', '% Date Match']:
        if col in df_summary.columns:
            date_match_col = col
            break
    
    for col in ['% data match', 'data_match_pct', '% Data Match']:
        if col in df_summary.columns:
            data_match_col = col
            break
    
    if date_match_col is None or data_match_col is None:
        return None
    
    # Filter out non-numeric rows
    df_plot = df_summary.copy()
    df_plot[date_match_col] = pd.to_numeric(df_plot[date_match_col], errors='coerce')
    df_plot[data_match_col] = pd.to_numeric(df_plot[data_match_col], errors='coerce')
    df_plot = df_plot.dropna(subset=[date_match_col, data_match_col])
    
    if df_plot.empty:
        return None
    
    fig = px.scatter(
        df_plot,
        x=date_match_col,
        y=data_match_col,
        hover_data=['Dataset ID'] if 'Dataset ID' in df_plot.columns else None,
        color_discrete_sequence=[COLORS['primary']]
    )
    
    fig.update_traces(marker=dict(size=10, opacity=0.7))
    fig.update_layout(
        title=dict(text="% Date Match vs % Data Match", font=dict(size=18, color='white')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(title="% Date Match", gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title="% Data Match", gridcolor='rgba(255,255,255,0.1)')
    )
    return fig


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Title with gradient
    st.markdown("""
    <h1 style="text-align: center; background: linear-gradient(90deg, #3B82F6, #8B5CF6, #EC4899); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
               font-size: 2.5rem; font-weight: 800; margin-bottom: 0;">
            🔍 Migration Sanity Validation Dashboard
    </h1>
    """, unsafe_allow_html=True)
    
    # Sidebar - Data Source Selection
    with st.sidebar:
        st.markdown("### 📊 Data Source")
        st.markdown("---")
        
        # Check if Google Sheets is configured
        sheets_configured = check_sheets_configured()
        
        # Data source selector
        if sheets_configured:
            data_source = st.radio(
                "Select Data Source",
                options=["📡 Google Sheets (Auto)", "📁 Upload CSV Files"],
                index=0,
                help="Choose to load data automatically from Google Sheets or upload files manually"
            )
        else:
            data_source = "📁 Upload CSV Files"
            st.info("ℹ️ Google Sheets not configured. Using file upload mode.")
        
        st.markdown("---")
        
        # Initialize data variables
        df_ok = None
        df_partial = None
        df_not_ok = None
        df_summary = None
        data_loaded = False
        
        if data_source == "📡 Google Sheets (Auto)":
            # Google Sheets mode
            st.markdown("### 🔗 Google Sheets")
            
            # Show refresh button
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh Data", use_container_width=True):
                    st.cache_data.clear()
                    st.rerun()
            with col2:
                st.caption(f"Cache: {CACHE_TTL//60}min")
            
            st.markdown("---")
            
            # Load data with spinner
            with st.spinner("Loading from Google Sheets..."):
                df_ok, df_partial, df_not_ok, df_summary = load_all_sheets_data(GOOGLE_SHEET_ID)
            
            # Check loading status
            ok_loaded = df_ok is not None and not df_ok.empty
            partial_loaded = df_partial is not None and not df_partial.empty
            not_ok_loaded = df_not_ok is not None and not df_not_ok.empty
            summary_loaded = df_summary is not None and not df_summary.empty
            
            st.markdown("### Data Status")
            if ok_loaded:
                st.success(f"✅ OK: {len(df_ok):,} rows")
            else:
                st.error("❌ OK: Failed to load")
            
            if partial_loaded:
                st.success(f"✅ OK but need attention: {len(df_partial):,} rows")
            else:
                st.error("❌ OK but need attention: Failed to load")
            
            if not_ok_loaded:
                st.success(f"✅ Not OK: {len(df_not_ok):,} rows")
            else:
                st.error("❌ Not OK: Failed to load")
            
            if summary_loaded:
                st.success(f"✅ Summary: {len(df_summary):,} rows")
            else:
                st.info("ℹ️ Summary: Not available")
            
            data_loaded = ok_loaded and partial_loaded and not_ok_loaded
            
        else:
            # File upload mode
            st.markdown("### 📁 Upload Files")
            
            ok_file = st.file_uploader(
                "✅ OK Datasets",
                type="csv",
                key="ok",
                help="Upload ok_datasets.csv"
            )
            
            partial_file = st.file_uploader(
                "⚠️ OK but need attention Datasets",
                type="csv",
                key="partial",
                help="Upload partial_ok_datasets.csv"
            )
            
            not_ok_file = st.file_uploader(
                "❌ Not OK Datasets",
                type="csv",
                key="notok",
                help="Upload not_ok_datasets.csv"
            )
            
            summary_file = st.file_uploader(
                "📊 Not OK Summary",
                type="csv",
                key="summary",
                help="Upload not_ok_summary.csv (optional)"
            )
            
            st.markdown("---")
            st.markdown("### Upload Status")
            
            if ok_file:
                st.success("✅ OK Datasets uploaded")
            else:
                st.warning("⏳ Please upload OK Datasets")
            
            if partial_file:
                st.success("✅ OK but need attention Datasets uploaded")
            else:
                st.warning("⏳ Please upload OK but need attention Datasets")
            
            if not_ok_file:
                st.success("✅ Not OK Datasets uploaded")
            else:
                st.warning("⏳ Please upload Not OK Datasets")
            
            if summary_file:
                st.success("✅ Summary file uploaded")
            else:
                st.info("ℹ️ Summary file (optional)")
            
            # Load uploaded files
            if ok_file and partial_file and not_ok_file:
                try:
                    df_ok = pd.read_csv(ok_file)
                    df_partial = pd.read_csv(partial_file)
                    df_not_ok = pd.read_csv(not_ok_file)
                    df_summary = pd.read_csv(summary_file) if summary_file else None
                    data_loaded = True
                except Exception as e:
                    st.error(f"Error loading files: {str(e)}")
                    data_loaded = False
    
    # Check if data is loaded
    if not data_loaded:
        if data_source == "📡 Google Sheets (Auto)":
            st.markdown("""
            <div style="text-align: center; padding: 60px; background: linear-gradient(145deg, #1e293b, #334155); 
                        border-radius: 20px; margin: 40px 0;">
                <h2 style="color: #EF4444;">❌ Failed to Load Data from Google Sheets</h2>
                <p style="color: #64748b; font-size: 1.1rem;">
                    Please check:<br>
                    • Google Sheet ID is correct<br>
                    • Sheet is publicly accessible (Anyone with link can view)<br>
                    • Tab names match: ok_datasets, partial_ok_datasets, not_ok_datasets, not_ok_summary
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 60px; background: linear-gradient(145deg, #1e293b, #334155); 
                        border-radius: 20px; margin: 40px 0;">
                <h2 style="color: #94a3b8;">👆 Upload Files to Get Started</h2>
                <p style="color: #64748b; font-size: 1.1rem;">
                    Please upload at least the OK, OK but need attention, and Not OK CSV files<br>
                    to view the validation dashboard.
                </p>
            </div>
            """, unsafe_allow_html=True)
        return
    
    # Show data source indicator
    if data_source == "📡 Google Sheets (Auto)":
        st.caption(f"📡 Data loaded from Google Sheets • Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ==========================================================================
    # TOP-LEVEL FILTERS
    # ==========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    
    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])
    
    with filter_col1:
        dataset_type = st.selectbox(
            "📁 Dataset Type",
            options=["Ingest", "OLAP"],
            index=0,
            help="Filter by dataset type"
        )
    
    with filter_col2:
        decision_type = st.selectbox(
            "🔄 Move to PPSL?",
            options=["Remain at OCL", "Move to PPSL", "Maintain at Both"],
            index=0,
            help="Filter by migration decision"
        )
    
    # Display current filter selections
    with filter_col3:
        st.markdown(f"""
        <div style="background: rgba(59, 130, 246, 0.1); padding: 12px 20px; border-radius: 10px; 
                    border-left: 4px solid #3B82F6; margin-top: 25px;">
            <span style="color: #94a3b8;">Current View:</span>
            <strong style="color: #3B82F6;"> {dataset_type}</strong> datasets |
            <strong style="color: #8B5CF6;"> {decision_type}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Calculate metrics
    ok_count = len(df_ok['Dataset ID'].unique()) if 'Dataset ID' in df_ok.columns else len(df_ok)
    partial_count_raw = len(df_partial['Dataset ID'].unique()) if 'Dataset ID' in df_partial.columns else len(df_partial)
    not_ok_count_raw = len(df_not_ok['Dataset ID'].unique()) if 'Dataset ID' in df_not_ok.columns else len(df_not_ok)
    
    # Count Case 6 from "OK but need attention" - these should be counted as Not OK for metrics
    case6_count = 0
    if not df_partial.empty:
        reason_col = next((c for c in ['Sanity reason', 'sanity reason', 'sanity_reason', 'Sanity Reason'] if c in df_partial.columns), None)
        if reason_col:
            df_case6 = df_partial[df_partial[reason_col].astype(str).str.contains('Case 6', case=False, na=False)]
            case6_count = df_case6['Dataset ID'].nunique() if 'Dataset ID' in df_case6.columns else len(df_case6)
    
    # Adjust counts: Move Case 6 from "OK but need attention" to "Not OK"
    partial_count = partial_count_raw - case6_count
    not_ok_count = not_ok_count_raw + case6_count
    
    total_count = ok_count + partial_count + not_ok_count
    # Success Rate = (OK + OK but need attention (excluding Case 6)) / Total
    success_rate = ((ok_count + partial_count) / total_count * 100) if total_count > 0 else 0
    
    # Calculate percentages for each category
    ok_percentage = (ok_count / total_count * 100) if total_count > 0 else 0
    partial_percentage = (partial_count / total_count * 100) if total_count > 0 else 0
    not_ok_percentage = (not_ok_count / total_count * 100) if total_count > 0 else 0
    
    # Get sanity run date
    sanity_date = get_sanity_run_date(df_ok) or get_sanity_run_date(df_not_ok)
    
    # Display sanity run date
    st.markdown(f"""
    <p style="text-align: center; color: #94a3b8; font-size: 1rem; margin-top: -10px;">
        📅 Sanity Run Date: <strong style="color: #3B82F6;">{sanity_date}</strong>
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(create_metric_card("✅", "OK Datasets", ok_count, COLORS['ok'], ok_percentage), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric_card("⚠️", "OK but need attention", partial_count, COLORS['partial_ok'], partial_percentage), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_metric_card("❌", "Not OK", not_ok_count, COLORS['not_ok'], not_ok_percentage), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_metric_card("📊", "Total Datasets", total_count, COLORS['primary']), unsafe_allow_html=True)
    
    with col5:
        st.markdown(create_percentage_card("📈", "Success Rate", success_rate, COLORS['ok'] if success_rate >= 70 else COLORS['partial_ok']), unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "✅ OK Datasets", "⚠️ OK but need attention", "❌ Not OK", "📈 Summary"])
    
    # ==========================================================================
    # OVERVIEW TAB
    # ==========================================================================
    with tab1:
        st.markdown("### 📊 Validation Overview")
        st.markdown("---")
        
        # Row 1: Distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = create_distribution_pie(ok_count, partial_count, not_ok_count)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Combine all dataframes for case distribution
            all_cases_df = pd.concat([
                df_ok.assign(Category='OK') if not df_ok.empty else pd.DataFrame(),
                df_partial.assign(Category='OK but need attention') if not df_partial.empty else pd.DataFrame(),
                df_not_ok.assign(Category='Not OK') if not df_not_ok.empty else pd.DataFrame()
            ], ignore_index=True)
            
            if not all_cases_df.empty:
                fig_cases = create_case_distribution_bar(all_cases_df)
                if fig_cases:
                    st.plotly_chart(fig_cases, use_container_width=True)
        
        # Row 2: More charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_ingest = create_ingest_type_bar(df_ok, df_partial, df_not_ok)
            if fig_ingest:
                st.plotly_chart(fig_ingest, use_container_width=True)
        
        with col2:
            fig_status = create_status_flag_pie(df_partial, df_not_ok)
            if fig_status:
                st.plotly_chart(fig_status, use_container_width=True)
        
        # Row 3: Max date chart
        col1, col2 = st.columns(2)
        with col1:
            fig_max_date = create_max_date_bar(df_partial, df_not_ok)
            if fig_max_date:
                st.plotly_chart(fig_max_date, use_container_width=True)
        
        # Row 4: Source Type Overview Table
        st.markdown("---")
        st.markdown("### 📊 Group by Source Type")
        
        source_type_overview = create_source_type_overview_table(df_ok, df_partial, df_not_ok)
        if source_type_overview is not None:
            st.dataframe(source_type_overview, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ Source type information not available in the data.")
        
        # Row 5: Source Type Pie Charts
        st.markdown("---")
        st.markdown("### 🎨 Source Type Distribution")
        
        pie_col1, pie_col2 = st.columns(2)
        
        with pie_col1:
            # OK but need attention source type pie chart
            fig_partial_pie = create_source_type_pie_chart(
                df_partial, 
                "OK but need attention - Source Types",
                px.colors.qualitative.Set2
            )
            if fig_partial_pie:
                st.plotly_chart(fig_partial_pie, use_container_width=True)
        
        with pie_col2:
            # Not OK source type pie chart
            fig_notok_pie = create_source_type_pie_chart(
                df_not_ok, 
                "Not OK - Source Types",
                px.colors.qualitative.Set1
            )
            if fig_notok_pie:
                st.plotly_chart(fig_notok_pie, use_container_width=True)
        
        # Export all data
        st.markdown("---")
        st.markdown("### 📥 Export Full Report")
        
        excel_data = convert_dfs_to_excel({
            'OK Datasets': df_ok,
            'OK but need attention': df_partial,
            'Not OK': df_not_ok,
            'Summary': df_summary
        })
        
        st.download_button(
            label="📥 Download Full Report (Excel)",
            data=excel_data,
            file_name=f"ocl_validation_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # ==========================================================================
    # OK DATASETS TAB
    # ==========================================================================
    with tab2:
        st.markdown("### ✅ OK Datasets")
        st.markdown("---")
        
        # Case-wise Distribution Section
        case_result = get_case_distribution(df_ok, case_descriptions=OK_CASE_DESCRIPTIONS)
        if case_result:
            case_data, total = case_result
            create_case_distribution_table(case_data, total, "Case-wise Distribution (OK Datasets)")
            st.markdown("---")
        
        # Filters
        with st.expander("🔍 Filters", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                search_ok = st.text_input("Search Dataset ID", key="search_ok", placeholder="Enter Dataset ID...")
            
            with col2:
                status_col = 'Status' if 'Status' in df_ok.columns else None
                if status_col:
                    status_filter = st.multiselect("Status", options=df_ok[status_col].unique(), key="ok_status")
                else:
                    status_filter = []
            
            with col3:
                ingest_col = next((c for c in ['Ingestion type', 'Ingest type'] if c in df_ok.columns), None)
                if ingest_col:
                    ingest_filter = st.multiselect("Ingest Type", options=df_ok[ingest_col].unique(), key="ok_ingest")
                else:
                    ingest_filter = []
        
        # Apply filters
        filtered_ok = df_ok.copy()
        if search_ok:
            filtered_ok = search_dataset(filtered_ok, search_ok)
        if status_filter and status_col:
            filtered_ok = filtered_ok[filtered_ok[status_col].isin(status_filter)]
        if ingest_filter and ingest_col:
            filtered_ok = filtered_ok[filtered_ok[ingest_col].isin(ingest_filter)]
        
        # Display count
        st.markdown(f"**Showing {len(filtered_ok):,} records**")
        
        # Display table
        st.dataframe(filtered_ok, use_container_width=True, height=500)
        
        # Download
        st.download_button(
            label="📥 Download Filtered Results (CSV)",
            data=convert_df_to_csv(filtered_ok),
            file_name="ok_datasets_filtered.csv",
            mime="text/csv"
        )
    
    # ==========================================================================
    # PARTIAL OK DATASETS TAB
    # ==========================================================================
    with tab3:
        st.markdown("### ⚠️ OK but need attention Datasets")
        st.markdown("---")
        
        # Case-wise Distribution Section with Next Steps
        case_result = get_case_distribution(df_partial, case_descriptions=OK_BUT_NEED_ATTENTION_CASE_DESCRIPTIONS)
        if case_result:
            case_data, total = case_result
            # Add Next Steps column
            case_data['Next Steps'] = case_data['Case'].map(OK_BUT_NEED_ATTENTION_NEXT_STEPS).fillna('')
            create_case_distribution_table_with_next_steps(case_data, total, "Case-wise Distribution (OK but need attention Datasets)")
            st.markdown("---")
        
        # Source Type Section - Pie Chart and Table
        st.markdown("#### 🎨 Source Type Distribution")
        
        # Pie chart (full width)
        fig_partial_source = create_source_type_pie_chart(
            df_partial, 
            "Source Type Distribution - OK but need attention",
            px.colors.qualitative.Pastel
        )
        if fig_partial_source:
            st.plotly_chart(fig_partial_source, use_container_width=True)
        
        st.markdown("#### 📊 Source Type Breakdown by Case")
        # Case-wise breakdown table (full width)
        source_breakdown_partial = create_source_type_case_breakdown(df_partial, case_count=6, title="OK but need attention Datasets")
        if source_breakdown_partial is not None:
            st.dataframe(source_breakdown_partial, use_container_width=True, hide_index=True, height=400)
        else:
            st.info("ℹ️ Source type breakdown not available.")
        
        st.markdown("---")
        
        # Filters - Row 1
        with st.expander("🔍 Filters", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                search_partial = st.text_input("Search Dataset ID", key="search_partial", placeholder="Enter Dataset ID...")
            
            with col2:
                ingest_col = next((c for c in ['Ingest type', 'Ingestion type'] if c in df_partial.columns), None)
                if ingest_col:
                    ingest_partial = st.multiselect("Ingest Type", options=df_partial[ingest_col].unique(), key="partial_ingest")
                else:
                    ingest_partial = []
            
            with col3:
                max_date_col = next((c for c in ['Max date not matching', 'max_date_matching'] if c in df_partial.columns), None)
                if max_date_col:
                    max_date_partial = st.multiselect("Max Date Status", options=df_partial[max_date_col].unique(), key="partial_maxdate")
                else:
                    max_date_partial = []
            
            with col4:
                reason_col = next((c for c in ['Sanity reason', 'sanity reason', 'Sanity Reason'] if c in df_partial.columns), None)
                if reason_col:
                    reasons = df_partial[reason_col].dropna().unique()
                    reason_partial = st.multiselect("Sanity Reason", options=reasons, key="partial_reason")
                else:
                    reason_partial = []
            
            # Filters - Row 2 (New filters)
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                old_status_col = next((c for c in ['Old status', 'old_status', 'Old Status'] if c in df_partial.columns), None)
                if old_status_col:
                    old_status_partial = st.multiselect("Old Status", options=df_partial[old_status_col].dropna().unique(), key="partial_old_status")
                else:
                    old_status_partial = []
            
            with col6:
                new_status_col = next((c for c in ['New status', 'new_status', 'New Status'] if c in df_partial.columns), None)
                if new_status_col:
                    new_status_partial = st.multiselect("New Status", options=df_partial[new_status_col].dropna().unique(), key="partial_new_status")
                else:
                    new_status_partial = []
            
            with col7:
                old_source_col = next((c for c in ['old_source_type', 'Old Source Type', 'old source type'] if c in df_partial.columns), None)
                if old_source_col:
                    old_source_partial = st.multiselect("Old Source Type", options=df_partial[old_source_col].dropna().unique(), key="partial_old_source")
                else:
                    old_source_partial = []
            
            with col8:
                new_source_col = next((c for c in ['new_source_type', 'New Source Type', 'new source type'] if c in df_partial.columns), None)
                if new_source_col:
                    new_source_partial = st.multiselect("New Source Type", options=df_partial[new_source_col].dropna().unique(), key="partial_new_source")
                else:
                    new_source_partial = []
        
        # Apply filters
        filtered_partial = df_partial.copy()
        if search_partial:
            filtered_partial = search_dataset(filtered_partial, search_partial)
        if ingest_partial and ingest_col:
            filtered_partial = filtered_partial[filtered_partial[ingest_col].isin(ingest_partial)]
        if max_date_partial and max_date_col:
            filtered_partial = filtered_partial[filtered_partial[max_date_col].isin(max_date_partial)]
        if reason_partial and reason_col:
            filtered_partial = filtered_partial[filtered_partial[reason_col].isin(reason_partial)]
        # New filters
        if old_status_partial and old_status_col:
            filtered_partial = filtered_partial[filtered_partial[old_status_col].isin(old_status_partial)]
        if new_status_partial and new_status_col:
            filtered_partial = filtered_partial[filtered_partial[new_status_col].isin(new_status_partial)]
        if old_source_partial and old_source_col:
            filtered_partial = filtered_partial[filtered_partial[old_source_col].isin(old_source_partial)]
        if new_source_partial and new_source_col:
            filtered_partial = filtered_partial[filtered_partial[new_source_col].isin(new_source_partial)]
        
        # Add Next Steps Suggested column based on case
        if reason_col and reason_col in filtered_partial.columns:
            filtered_partial['Next Steps Suggested'] = filtered_partial[reason_col].apply(
                lambda x: OK_BUT_NEED_ATTENTION_NEXT_STEPS.get(extract_case_number(x), '') if pd.notna(x) else ''
            )
        
        # Display count
        unique_datasets = filtered_partial['Dataset ID'].nunique() if 'Dataset ID' in filtered_partial.columns else len(filtered_partial)
        st.markdown(f"**Showing {len(filtered_partial):,} records ({unique_datasets:,} unique datasets)**")
        
        # Display table
        st.dataframe(filtered_partial, use_container_width=True, height=500)
        
        # Download
        st.download_button(
            label="📥 Download Filtered Results (CSV)",
            data=convert_df_to_csv(filtered_partial),
            file_name="ok_but_need_attention_datasets_filtered.csv",
            mime="text/csv"
        )
    
    # ==========================================================================
    # NOT OK DATASETS TAB
    # ==========================================================================
    with tab4:
        st.markdown("### ❌ Not OK Datasets")
        st.markdown("---")
        
        # Case-wise Distribution Section
        case_result = get_case_distribution(df_not_ok, case_descriptions=NOT_OK_CASE_DESCRIPTIONS)
        if case_result:
            case_data, total = case_result
            create_case_distribution_table(case_data, total, "Case-wise Distribution (Not OK Datasets)")
            st.markdown("---")
        
        # Source Type Section - Pie Chart and Table
        st.markdown("#### 🎨 Source Type Distribution")
        
        # Pie chart (full width)
        fig_notok_source = create_source_type_pie_chart(
            df_not_ok, 
            "Source Type Distribution - Not OK",
            px.colors.qualitative.Bold
        )
        if fig_notok_source:
            st.plotly_chart(fig_notok_source, use_container_width=True)
        
        st.markdown("#### 📊 Source Type Breakdown by Case")
        # Case-wise breakdown table (full width)
        source_breakdown_notok = create_source_type_case_breakdown(df_not_ok, case_count=13, title="Not OK Datasets")
        if source_breakdown_notok is not None:
            st.dataframe(source_breakdown_notok, use_container_width=True, hide_index=True, height=400)
        else:
            st.info("ℹ️ Source type breakdown not available.")
        
        st.markdown("---")
        
        # Filters - Row 1
        with st.expander("🔍 Filters", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                search_notok = st.text_input("Search Dataset ID", key="search_notok", placeholder="Enter Dataset ID...")
            
            with col2:
                ingest_col = next((c for c in ['Ingest type', 'Ingestion type'] if c in df_not_ok.columns), None)
                if ingest_col:
                    ingest_notok = st.multiselect("Ingest Type", options=df_not_ok[ingest_col].unique(), key="notok_ingest")
                else:
                    ingest_notok = []
            
            with col3:
                status_col = next((c for c in ['Status flag', 'status_flag'] if c in df_not_ok.columns), None)
                if status_col:
                    status_notok = st.multiselect("Status Flag", options=df_not_ok[status_col].unique(), key="notok_status")
                else:
                    status_notok = []
            
            with col4:
                reason_col = next((c for c in ['Sanity reason', 'sanity reason', 'Sanity Reason'] if c in df_not_ok.columns), None)
                if reason_col:
                    reasons = df_not_ok[reason_col].dropna().unique()
                    reason_notok = st.multiselect("Sanity Reason", options=reasons, key="notok_reason")
                else:
                    reason_notok = []
            
            # Filters - Row 2 (New filters)
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                old_status_col_notok = next((c for c in ['Old status', 'old_status', 'Old Status'] if c in df_not_ok.columns), None)
                if old_status_col_notok:
                    old_status_notok = st.multiselect("Old Status", options=df_not_ok[old_status_col_notok].dropna().unique(), key="notok_old_status")
                else:
                    old_status_notok = []
            
            with col6:
                new_status_col_notok = next((c for c in ['New status', 'new_status', 'New Status'] if c in df_not_ok.columns), None)
                if new_status_col_notok:
                    new_status_notok = st.multiselect("New Status", options=df_not_ok[new_status_col_notok].dropna().unique(), key="notok_new_status")
                else:
                    new_status_notok = []
            
            with col7:
                old_source_col_notok = next((c for c in ['old_source_type', 'Old Source Type', 'old source type'] if c in df_not_ok.columns), None)
                if old_source_col_notok:
                    old_source_notok = st.multiselect("Old Source Type", options=df_not_ok[old_source_col_notok].dropna().unique(), key="notok_old_source")
                else:
                    old_source_notok = []
            
            with col8:
                new_source_col_notok = next((c for c in ['new_source_type', 'New Source Type', 'new source type'] if c in df_not_ok.columns), None)
                if new_source_col_notok:
                    new_source_notok = st.multiselect("New Source Type", options=df_not_ok[new_source_col_notok].dropna().unique(), key="notok_new_source")
                else:
                    new_source_notok = []
        
        # Apply filters
        filtered_notok = df_not_ok.copy()
        if search_notok:
            filtered_notok = search_dataset(filtered_notok, search_notok)
        if ingest_notok and ingest_col:
            filtered_notok = filtered_notok[filtered_notok[ingest_col].isin(ingest_notok)]
        if status_notok and status_col:
            filtered_notok = filtered_notok[filtered_notok[status_col].isin(status_notok)]
        if reason_notok and reason_col:
            filtered_notok = filtered_notok[filtered_notok[reason_col].isin(reason_notok)]
        # New filters
        if old_status_notok and old_status_col_notok:
            filtered_notok = filtered_notok[filtered_notok[old_status_col_notok].isin(old_status_notok)]
        if new_status_notok and new_status_col_notok:
            filtered_notok = filtered_notok[filtered_notok[new_status_col_notok].isin(new_status_notok)]
        if old_source_notok and old_source_col_notok:
            filtered_notok = filtered_notok[filtered_notok[old_source_col_notok].isin(old_source_notok)]
        if new_source_notok and new_source_col_notok:
            filtered_notok = filtered_notok[filtered_notok[new_source_col_notok].isin(new_source_notok)]
        
        # Display count
        unique_datasets = filtered_notok['Dataset ID'].nunique() if 'Dataset ID' in filtered_notok.columns else len(filtered_notok)
        st.markdown(f"**Showing {len(filtered_notok):,} records ({unique_datasets:,} unique datasets)**")
        
        # Add count difference column
        if 'old count' in filtered_notok.columns and 'new count' in filtered_notok.columns:
            filtered_notok['Count Diff'] = pd.to_numeric(filtered_notok['old count'], errors='coerce') - pd.to_numeric(filtered_notok['new count'], errors='coerce')
        
        # Display table
        st.dataframe(
            filtered_notok,
            use_container_width=True,
            height=500
        )
        
        # Download
        st.download_button(
            label="📥 Download Filtered Results (CSV)",
            data=convert_df_to_csv(filtered_notok),
            file_name="not_ok_datasets_filtered.csv",
            mime="text/csv"
        )
    
    # ==========================================================================
    # SUMMARY TAB
    # ==========================================================================
    with tab5:
        st.markdown("### 📈 Not OK Summary Analysis")
        st.markdown("---")
        
        # Calculate summary statistics from actual data (fixes N/A issue)
        st.markdown("#### 📊 Summary Statistics")
        
        # Calculate % Data Match and % Data Unmatch from actual counts
        total_datasets_calc = ok_count + partial_count + not_ok_count
        pct_data_match = (ok_count / total_datasets_calc * 100) if total_datasets_calc > 0 else 0
        pct_data_unmatch = (not_ok_count / total_datasets_calc * 100) if total_datasets_calc > 0 else 0
        
        # Calculate average % date match/unmatch from Not OK summary if available
        avg_date_match = 0
        avg_date_unmatch = 0
        
        if df_summary is not None and not df_summary.empty:
            # Find date match column
            date_match_col = next((c for c in ['% date match', '% Date Match', 'date_match_pct'] if c in df_summary.columns), None)
            date_unmatch_col = next((c for c in ['% date unmatch', '% Date Unmatch', 'date_unmatch_pct'] if c in df_summary.columns), None)
            
            if date_match_col:
                # Clean and convert to numeric (remove % signs if present)
                df_temp = df_summary.copy()
                df_temp[date_match_col] = df_temp[date_match_col].astype(str).str.replace('%', '').str.strip()
                df_temp[date_match_col] = pd.to_numeric(df_temp[date_match_col], errors='coerce')
                avg_date_match = df_temp[date_match_col].mean()
                if pd.isna(avg_date_match):
                    avg_date_match = 0
            
            if date_unmatch_col:
                df_temp = df_summary.copy()
                df_temp[date_unmatch_col] = df_temp[date_unmatch_col].astype(str).str.replace('%', '').str.strip()
                df_temp[date_unmatch_col] = pd.to_numeric(df_temp[date_unmatch_col], errors='coerce')
                avg_date_unmatch = df_temp[date_unmatch_col].mean()
                if pd.isna(avg_date_unmatch):
                    avg_date_unmatch = 0
        
        # Display 4 summary metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("% Data Match", f"{pct_data_match:.1f}%", help="OK datasets / Total datasets")
        with metric_col2:
            st.metric("% Data Unmatch", f"{pct_data_unmatch:.1f}%", help="Not OK datasets / Total datasets")
        with metric_col3:
            st.metric("% Avg Date Match", f"{avg_date_match:.1f}%", help="Average % date match across Not OK datasets")
        with metric_col4:
            st.metric("% Avg Date Unmatch", f"{avg_date_unmatch:.1f}%", help="Average % date unmatch across Not OK datasets")
        
        st.markdown("---")
        
        if df_summary is None or df_summary.empty:
            st.warning("⚠️ No summary file uploaded. Please upload not_ok_summary.csv to view detailed analysis.")
        else:
            # 6 New Filters for Summary Tab
            st.markdown("#### 🔍 Filters")
            
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            filter_col4, filter_col5, filter_col6 = st.columns(3)
            
            with filter_col1:
                # Ingest Type filter (from df_not_ok)
                ingest_options = ['All']
                ingest_col_notok = next((c for c in ['Ingest type', 'Ingestion type', 'ingest_type'] if c in df_not_ok.columns), None)
                if ingest_col_notok:
                    ingest_options += list(df_not_ok[ingest_col_notok].dropna().unique())
                summary_ingest_filter = st.selectbox("Ingest Type", options=ingest_options, key="summary_ingest")
            
            with filter_col2:
                # Status Flag filter
                status_options = ['All']
                status_col_notok = next((c for c in ['Status flag', 'status_flag', 'Status Flag'] if c in df_not_ok.columns), None)
                if status_col_notok:
                    status_options += list(df_not_ok[status_col_notok].dropna().unique())
                summary_status_filter = st.selectbox("Status Flag", options=status_options, key="summary_status")
            
            with filter_col3:
                # Old Status filter
                old_status_options = ['All']
                old_status_col_notok = next((c for c in ['Old status', 'old_status', 'Old Status'] if c in df_not_ok.columns), None)
                if old_status_col_notok:
                    old_status_options += list(df_not_ok[old_status_col_notok].dropna().unique())
                summary_old_status_filter = st.selectbox("Old Status", options=old_status_options, key="summary_old_status")
            
            with filter_col4:
                # New Status filter
                new_status_options = ['All']
                new_status_col_notok = next((c for c in ['New status', 'new_status', 'New Status'] if c in df_not_ok.columns), None)
                if new_status_col_notok:
                    new_status_options += list(df_not_ok[new_status_col_notok].dropna().unique())
                summary_new_status_filter = st.selectbox("New Status", options=new_status_options, key="summary_new_status")
            
            with filter_col5:
                # Old Source Type filter
                old_source_options = ['All']
                old_source_col_notok = next((c for c in ['old_source_type', 'Old Source Type', 'old source type'] if c in df_not_ok.columns), None)
                if old_source_col_notok:
                    old_source_options += list(df_not_ok[old_source_col_notok].dropna().unique())
                summary_old_source_filter = st.selectbox("Old Source Type", options=old_source_options, key="summary_old_source")
            
            with filter_col6:
                # New Source Type filter
                new_source_options = ['All']
                new_source_col_notok = next((c for c in ['new_source_type', 'New Source Type', 'new source type'] if c in df_not_ok.columns), None)
                if new_source_col_notok:
                    new_source_options += list(df_not_ok[new_source_col_notok].dropna().unique())
                summary_new_source_filter = st.selectbox("New Source Type", options=new_source_options, key="summary_new_source")
            
            st.markdown("---")
            
            # Apply filters to get filtered dataset IDs from df_not_ok
            filtered_not_ok_for_summary = df_not_ok.copy()
            
            if summary_ingest_filter != 'All' and ingest_col_notok:
                filtered_not_ok_for_summary = filtered_not_ok_for_summary[filtered_not_ok_for_summary[ingest_col_notok] == summary_ingest_filter]
            if summary_status_filter != 'All' and status_col_notok:
                filtered_not_ok_for_summary = filtered_not_ok_for_summary[filtered_not_ok_for_summary[status_col_notok] == summary_status_filter]
            if summary_old_status_filter != 'All' and old_status_col_notok:
                filtered_not_ok_for_summary = filtered_not_ok_for_summary[filtered_not_ok_for_summary[old_status_col_notok] == summary_old_status_filter]
            if summary_new_status_filter != 'All' and new_status_col_notok:
                filtered_not_ok_for_summary = filtered_not_ok_for_summary[filtered_not_ok_for_summary[new_status_col_notok] == summary_new_status_filter]
            if summary_old_source_filter != 'All' and old_source_col_notok:
                filtered_not_ok_for_summary = filtered_not_ok_for_summary[filtered_not_ok_for_summary[old_source_col_notok] == summary_old_source_filter]
            if summary_new_source_filter != 'All' and new_source_col_notok:
                filtered_not_ok_for_summary = filtered_not_ok_for_summary[filtered_not_ok_for_summary[new_source_col_notok] == summary_new_source_filter]
            
            # Get filtered dataset IDs
            filtered_dataset_ids = filtered_not_ok_for_summary['Dataset ID'].unique() if 'Dataset ID' in filtered_not_ok_for_summary.columns else []
            
            # Filter summary by dataset IDs
            filtered_summary = df_summary.copy()
            if len(filtered_dataset_ids) > 0 and 'Dataset ID' in filtered_summary.columns:
                filtered_summary = filtered_summary[filtered_summary['Dataset ID'].isin(filtered_dataset_ids)]
            
            st.caption(f"Showing {len(filtered_summary):,} records after filtering")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = create_data_match_histogram(filtered_summary)
                if fig_hist:
                    st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                fig_scatter = create_match_scatter(filtered_summary)
                if fig_scatter:
                    st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.markdown("---")
            
            # Full summary table
            st.markdown("#### 📋 Full Summary Table")
            
            # Search filter
            col1, col2 = st.columns([1, 3])
            with col1:
                search_summary = st.text_input("Search Dataset ID", key="search_summary", placeholder="Enter Dataset ID...")
            
            # Apply search to already filtered summary
            if search_summary:
                filtered_summary = search_dataset(filtered_summary, search_summary)
            
            st.dataframe(filtered_summary, use_container_width=True, height=400)
            
            # Identify worst performing datasets
            st.markdown("---")
            st.markdown("#### ⚠️ Lowest Match Datasets")
            
            date_match_col_summary = next((c for c in ['% date match', '% Date Match', 'date_match_pct'] if c in filtered_summary.columns), None)
            if date_match_col_summary:
                df_sorted = filtered_summary.copy()
                df_sorted[date_match_col_summary] = df_sorted[date_match_col_summary].astype(str).str.replace('%', '').str.strip()
                df_sorted[date_match_col_summary] = pd.to_numeric(df_sorted[date_match_col_summary], errors='coerce')
                df_sorted = df_sorted.dropna(subset=[date_match_col_summary])
                df_sorted = df_sorted.nsmallest(10, date_match_col_summary)
                st.dataframe(df_sorted, use_container_width=True)
            else:
                st.info("ℹ️ No date match column found in summary data.")
            
            # Download
            st.download_button(
                label="📥 Download Summary (CSV)",
                data=convert_df_to_csv(filtered_summary),
                file_name="not_ok_summary_filtered.csv",
                mime="text/csv"
            )
    
    # ==========================================================================
    # SANITY RUN HISTORY SECTION (Bottom)
    # ==========================================================================
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <h2 style="text-align: center; color: #F9FAFB;">
        📅 Sanity Run History
    </h2>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Calculate success rate by run
    run_history = get_success_rate_by_run(df_ok, df_partial, df_not_ok)
    
    if run_history is not None and not run_history.empty:
        # Show line chart if multiple runs
        if len(run_history) > 1:
            fig_success = create_success_rate_chart(run_history)
            if fig_success:
                st.plotly_chart(fig_success, use_container_width=True)
        
        st.markdown("#### 📊 Run-wise Breakdown")
        
        # Add styling to the dataframe
        def style_success_rate(val):
            """Color success rate based on value."""
            try:
                val_float = float(val)
                if val_float >= 70:
                    return f'color: {COLORS["ok"]}'
                elif val_float >= 50:
                    return f'color: {COLORS["partial_ok"]}'
                else:
                    return f'color: {COLORS["not_ok"]}'
            except:
                return ''
        
        # Display the table
        st.dataframe(
            run_history,
            use_container_width=True,
            height=min(400, 50 + len(run_history) * 35)
        )
        
        # Summary metrics for run history
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Runs", len(run_history))
        with col2:
            avg_success = run_history['Success Rate (%)'].mean()
            st.metric("Avg Success Rate", f"{avg_success:.1f}%")
        with col3:
            max_success = run_history['Success Rate (%)'].max()
            st.metric("Best Run", f"{max_success:.1f}%")
        with col4:
            min_success = run_history['Success Rate (%)'].min()
            st.metric("Worst Run", f"{min_success:.1f}%")
        
        # Download run history
        st.download_button(
            label="📥 Download Run History (CSV)",
            data=convert_df_to_csv(run_history),
            file_name="sanity_run_history.csv",
            mime="text/csv"
        )
    else:
        st.info("ℹ️ Run history requires 'sanity run date' column in the uploaded files.")


if __name__ == "__main__":
    main()

