# B3_Configurable_Minute_SOL_Optuna_V7_CombinedImb_ONLINE_5MIN_NTFY.py # <-- Renamed
# Runs prediction online every 5 minutes using PostgreSQL data.
# Uses SMOTEENN + scale_pos_weight tuning within Optuna CV loop.
# Sends ntfy.sh notifications on high probability.
# *** ADDED Reversal-Bot Features ***

import pandas as pd
import numpy as np
import time
import os
import warnings
import traceback
from datetime import datetime, timedelta
import xgboost as xgb
import optuna
import pandas_ta as ta
import psycopg2 # <--- DB driver
import requests # <--- For sending HTTP requests (notifications)
import logging # <--- For better logging

# --- Import SMOTEENN ---
from imblearn.combine import SMOTEENN

# Modeling Imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.exceptions import UndefinedMetricWarning

# --- Suppress Warnings & Configure Logging ---
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Configure logging

# ==============================================================================
# --- Configuration ---
# ==============================================================================
# Database Configuration
DB_PARAMS = {
    'dbname': 'ohlcv_db',
    'user': 'postgres',
    'password': 'bubZ$tep433', # Consider using environment variables for passwords
    'host': 'localhost',
    'port': '5432'
}
DB_TABLE_NAME = 'ohlcv' # <--- Your minute aggregate table name
SYMBOL_NAME = 'BTC' # Symbol to process

# Model & Feature Configuration
PREDICTION_WINDOW_MINUTES = int(10); TARGET_THRESHOLD_PCT = 0.224
TRAIN_WINDOW_MINUTES = 8 * 60; # 480 minutes
XGB_FIXED_PARAMS = {"objective":"binary:logistic", "eval_metric":"aucpr", "use_label_encoder": False, "random_state":42, "tree_method":"hist", "n_jobs":-1, "n_estimators":141}
N_OPTUNA_TRIALS = 64; OPTUNA_CV_SPLITS = 3
OPTUNA_EVAL_THRESHOLD = 0.40

# Online Loop & Notification Configuration
TARGET_CYCLE_TIME_SECONDS = 5 * 60 # Target cycle time (5 minutes)
DATA_FETCH_BUFFER_MINUTES = 120 # Extra minutes to fetch for feature lags/target calc

# --- ntfy.sh Configuration ---
NTFY_TOPIC = "bubZ_solana_224" # <--- YOUR SPECIFIC TOPIC NAME HERE
NOTIFICATION_THRESHOLD = 0.90 # Send notification if probability >= this value

# --- Reversal-Bot Feature Constants (Added) ---
RSI_LEN_RB = 60
BB_LEN_RB = 120
BB_STD_RB = 2 # Standard BB deviation, needed for _bbands even if only mid is used
ADX_LEN_RB = 420
DIV_LOOKBACK_RB = 420
ADX_THRESH_RB = 10.0
EPS = 1e-9 # Ensure epsilon is defined (was lowercase e before, changed to EPS for clarity)

# ==============================================================================
# --- Derived Variables ---
# ==============================================================================
# Update MAX_FEATURE_LAG to the maximum lookback used by *any* feature
# Consider standard TA lags (e.g., 60) and new Reversal-Bot lags
MAX_FEATURE_LAG = max(60, RSI_LEN_RB, BB_LEN_RB, ADX_LEN_RB, DIV_LOOKBACK_RB)
MINUTES_TO_FETCH = TRAIN_WINDOW_MINUTES + PREDICTION_WINDOW_MINUTES + MAX_FEATURE_LAG + DATA_FETCH_BUFFER_MINUTES
# epsilon = 1e-9 # Defined above as EPS

# ==============================================================================
# --- Utility indicator implementations (Added from new script) ---
# ==============================================================================

def _rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    # Use adjust=False for consistency if needed, ewm defaults might differ slightly
    rs = gain.ewm(alpha=1/length, adjust=False).mean() / (loss.ewm(alpha=1/length, adjust=False).mean() + EPS)
    return 100 - (100 / (1 + rs))


def _bbands(series: pd.Series, length: int, stdev: float):
    # Ensure enough data points for rolling calculations
    if len(series) < length:
        nan_series = pd.Series(np.nan, index=series.index)
        return nan_series, nan_series, nan_series # Return NaNs if not enough data

    mid = series.rolling(length, min_periods=max(1, length // 2)).mean() # Use min_periods
    sd  = series.rolling(length, min_periods=max(1, length // 2)).std()
    upper = mid + stdev * sd
    lower = mid - stdev * sd
    return upper, mid, lower


def _dmi(high, low, close, length):
    # Ensure enough data points
    if len(high) < length + 1: # Need diff() and ewm
         return pd.Series(np.nan, index=high.index)

    up  = high.diff()
    dn  = -low.diff()

    # Ensure alignment and handle potential NaNs from diff()
    up = up.fillna(0)
    dn = dn.fillna(0)

    plus_dm  = np.where((up  > dn) & (up  > 0), up,  0.)
    minus_dm = np.where((dn  > up) & (dn  > 0), dn,  0.)

    # Calculate TR, handling potential NaNs in close.shift()
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(0) # Fill NaN TRs with 0

    atr = tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean() # Add min_periods

    # Calculate DIs, using .fillna(0) on the EWM results before division if ATR can be zero
    plus_di_num = pd.Series(plus_dm).ewm(alpha=1/length, adjust=False, min_periods=length).mean().fillna(0)
    minus_di_num = pd.Series(minus_dm).ewm(alpha=1/length, adjust=False, min_periods=length).mean().fillna(0)

    plus_di  = 100 * plus_di_num  / (atr + EPS)
    minus_di = 100 * minus_di_num / (atr + EPS)

    # Calculate DX and ADX
    di_sum = plus_di + minus_di
    di_diff = abs(plus_di - minus_di)
    dx = np.where(di_sum > 0, (di_diff / di_sum) * 100, 0) # Avoid division by zero

    adx = pd.Series(dx).ewm(alpha=1/length, adjust=False, min_periods=length).mean() # Add min_periods
    return adx

# ==============================================================================
# --- Notification Function (Unchanged) ---
# ==============================================================================
def send_ntfy_notification(title, message, priority=4):
    # ... (function remains the same) ...
    if not NTFY_TOPIC:
        logging.warning("ntfy topic not configured. Skipping notification.")
        return
    safe_topic = requests.utils.quote(NTFY_TOPIC)
    url = f"https://ntfy.sh/{safe_topic}"
    headers = { "Title": title.encode('utf-8'), "Priority": str(priority), "Tags": "chart_with_upwards_trend,sol" }
    try:
        response = requests.post(url, data=message.encode('utf-8'), headers=headers, timeout=15)
        response.raise_for_status()
        logging.info(f"ntfy notification sent successfully to topic '{NTFY_TOPIC}': {title}")
    except requests.exceptions.Timeout:
        logging.error(f"Error sending ntfy notification: Request timed out")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending ntfy notification: {e}")
        if e.response is not None:
             logging.error(f"ntfy Response Status: {e.response.status_code}")
             logging.error(f"ntfy Response Body: {e.response.text}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during ntfy send: {e}")


# ==============================================================================
# --- Database Functions (Unchanged, assuming conversion to float is correct) ---
# ==============================================================================
def connect_db(params):
    # ... (function remains the same) ...
    conn = None
    try:
        conn = psycopg2.connect(**params)
        # logging.info("Database connection established successfully.") # Moved to main loop
        return conn
    except psycopg2.Error as e:
        logging.error(f"Error connecting to database: {e}")
        return None

def fetch_latest_data(conn, symbol, table_name, limit):
    # ... (function remains the same - ensure numeric conversion happens here) ...
    if not conn:
        logging.error("No database connection available for fetching data.")
        return None
    cursor = None
    try:
        cursor = conn.cursor()
        query = f"""
            SELECT timestamp, open, high, low, close, volumefrom, volumeto
            FROM {table_name}
            WHERE symbol = %s
            ORDER BY timestamp DESC
            LIMIT %s
        """
        cursor.execute(query, (symbol, limit))
        colnames = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        if not data:
             logging.warning(f"No data returned from query for symbol {symbol}.")
             return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volumefrom', 'volumeto'])

        df = pd.DataFrame(data, columns=colnames)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # *** Ensure this conversion is happening ***
        for col in ['open', 'high', 'low', 'close', 'volumefrom', 'volumeto']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = np.nan # Handle missing columns

        df = df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
        if not df.empty:
            logging.info(f"Fetched {len(df)} rows for {symbol}. Latest timestamp: {df['timestamp'].iloc[-1]}")
        else:
            logging.info(f"Fetched 0 rows for {symbol} after processing.")
        return df
    except (psycopg2.Error, pd.errors.EmptyDataError) as e:
        logging.error(f"Error fetching or processing data: {e}")
        return None
    finally:
        if cursor:
            cursor.close()


# ==============================================================================
# --- Feature Engineering Function (MODIFIED) ---
# ==============================================================================
def garman_klass_volatility_min(o, h, l, c, window_min):
     # This function is defined but not used in the main feature calc below? Keep it for now.
    with np.errstate(divide='ignore', invalid='ignore'): log_hl=np.log(h/l.replace(0, np.nan)); log_co=np.log(c/o.replace(0, np.nan))
    gk = 0.5*(log_hl**2) - (2*np.log(2)-1)*(log_co**2); gk = gk.fillna(0)
    min_p = max(1, window_min // 4); rm = gk.rolling(window_min, min_periods=min_p).mean(); rm = rm.clip(lower=0); return np.sqrt(rm)

def parkinson_volatility_min(h, l, window_min):
    # This function is defined but not used in the main feature calc below? Keep it for now.
    with np.errstate(divide='ignore', invalid='ignore'): log_hl_sq = np.log(h/l.replace(0, np.nan))**2
    log_hl_sq = log_hl_sq.fillna(0); min_p = max(1, window_min // 4); rs = log_hl_sq.rolling(window_min, min_periods=min_p).sum()
    f = 1/(4*np.log(2)*window_min) if window_min>0 else 0; return np.sqrt(f*rs)

def calculate_features_min_rare_event(df_input):
    """
    Calculates features. Uses uppercase column names after standardization.
    *** Includes Reversal-Bot features ***
    """
    df = df_input.copy()
    logging.info(f"  Feature Eng Start: Initial rows = {len(df)}")
    if df.empty:
        logging.warning("  Feature Eng Input: DataFrame is empty.")
        return df

    # --- Use original lowercase names for initial checks and base calculations ---
    essential_cols = ['open', 'high', 'low', 'close', 'volumefrom']
    initial_nan_check = df[essential_cols].isnull().sum()
    if initial_nan_check.sum() > 0:
        logging.warning(f"  Dropping {initial_nan_check.sum()} rows with NaNs in essential columns.")
        df = df.dropna(subset=essential_cols)
    if df.empty: logging.error("  Error: Empty DF after essential NaN drop."); return df
    base_cols_numeric = ['open', 'high', 'low', 'close', 'volumefrom', 'volumeto']
    for col in base_cols_numeric:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        else: df[col] = 0.0 # Initialize missing columns as float
    df = df.dropna(subset=essential_cols)
    if df.empty: logging.error("  Error: Empty DF after numeric conversion NaN drop."); return df

    # --- Base Feature Calculations (using lowercase initially) ---
    # ... (keep existing base calculations) ...
    df['price_change_1m_temp'] = df['close'].pct_change(periods=1)
    df['body_abs'] = abs(df['close'] - df['open'])
    df['range'] = df['high'] - df['low']
    df['body_ratio'] = (df['body_abs'] / (df['range'] + EPS)).clip(0, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        df['price_range_pct'] = (df['range'] / df['close'].replace(0, np.nan)) * 100
        df['oc_change_pct'] = (df['close'] - df['open']) / df['open'].replace(0, np.nan) * 100
    min_periods_rolling = 2
    for p in [5, 10, 15, 30, 60]: df[f'ma_{p}m'] = df['close'].rolling(p, min_periods=min_periods_rolling).mean()
    for p in [5, 10, 15, 30, 60]: df[f'rolling_std_{p}m'] = df['price_change_1m_temp'].rolling(p, min_periods=p//2).std() * 100
    lag_periods_price_min = [1, 3, 5, 10, 15, 30, 60]; lag_periods_volume_min = [1, 3, 5, 10, 15, 30, 60]
    for lag in lag_periods_price_min: df[f'lag_{lag}m_price_return'] = df['price_change_1m_temp'].shift(lag) * 100
    df['volume_return_1m'] = df['volumefrom'].pct_change(periods=1).replace([np.inf, -np.inf], 0) * 100
    for lag in lag_periods_volume_min: df[f'lag_{lag}m_volume_return'] = df['volume_return_1m'].shift(lag)
    vol_ma_period = 20; df[f'vol_ma_{vol_ma_period}m'] = df['volumefrom'].rolling(vol_ma_period, min_periods=vol_ma_period//2).mean()
    df['vol_spike_ratio'] = df['volumefrom'] / (df[f'vol_ma_{vol_ma_period}m'] + EPS)
    body_ma_period = 20; df[f'body_ma_{body_ma_period}m'] = df['body_abs'].rolling(body_ma_period, min_periods=body_ma_period//2).mean()
    df['body_spike_ratio'] = df['body_abs'] / (df[f'body_ma_{body_ma_period}m'] + EPS)
    df['prev_close']=df['close'].shift(1)
    df['hml']=df['high']-df['low']
    df['hmpc']=np.abs(df['high']-df['prev_close'])
    df['lmpc']=np.abs(df['low']-df['prev_close'])
    df['tr']=df[['hml','hmpc','lmpc']].max(axis=1); atr_periods_min = [14]; min_p_atr = 10;
    for p in atr_periods_min: df[f'atr_{p}m'] = df['tr'].rolling(p, min_periods=min_p_atr).mean()

    # --- TA FEATURES (RSI, Stoch, MACD, BBands) using pandas_ta ---
    logging.info("  Calculating standard TA features using pandas_ta...")
    try:
        min_ta_warmup = 30
        if len(df) >= min_ta_warmup:
            # Standard TA features
            df.ta.rsi(length=14, append=True) # Standard RSI(14)
            df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            df.ta.bbands(length=20, std=2, append=True) # Standard BBands(20,2)

            # Rename columns before standardizing to uppercase if necessary
            df.rename(columns={'MACDH_12_26_9': 'MACDH_12_26_9'}, inplace=True, errors='ignore')
            # Rename other potential clashes if needed

        else:
             logging.warning(f"  Warning: Insufficient data ({len(df)} rows) for TA warmup ({min_ta_warmup}). Skipping pandas_ta features.")

    except Exception as e_ta:
        logging.error(f"!! Error calculating pandas_ta features: {e_ta}")
        # Ensure expected columns exist as NaN if TA fails
        for col in ['RSI_14', 'STOCHK_14_3_3', 'MACDH_12_26_9', 'BBP_20_2.0', 'BBB_20_2.0', 'BBM_20_2.0']: # Added BBM
             if col not in df.columns: df[col] = np.nan

    # --- Standardize all column names to UPPERCASE ---
    logging.info("  Standardizing all column names to uppercase.")
    df.columns = df.columns.str.upper()

    # --- Now use UPPERCASE names for all subsequent calculations ---

    # --- Reversal-Bot Features (Added) ---
    logging.info("  Calculating Reversal-Bot specific features...")
    try: # Add try-except block for safety
        # Ensure base columns exist in uppercase
        base_cols_needed = ['CLOSE', 'HIGH', 'LOW']
        if not all(col in df.columns for col in base_cols_needed):
             raise ValueError(f"Missing one or more base columns for RB features: {base_cols_needed}")

        # Calculate RB features using helper functions
        df['RSI_RB'] = _rsi(df['CLOSE'], RSI_LEN_RB)
        _, df['BB_MID_RB'], _ = _bbands(df['CLOSE'], BB_LEN_RB, BB_STD_RB)
        df['ADX_RB'] = _dmi(df['HIGH'], df['LOW'], df['CLOSE'], ADX_LEN_RB)

        # Divergence Logic (Requires RSI_RB)
        df['BULL_REG_RB'] = 0 # Initialize columns
        df['BULL_HIDDEN_RB'] = 0
        if 'RSI_RB' in df.columns and not df['RSI_RB'].isnull().all() and len(df) > DIV_LOOKBACK_RB:
            # Use .copy() to avoid SettingWithCopyWarning if slicing later
            close_series = df['CLOSE'].copy()
            rsi_series = df['RSI_RB'].copy()

            low_now = close_series.rolling(DIV_LOOKBACK_RB, min_periods=1).min()
            low_prev = low_now.shift(1)
            rsi_low_prev = rsi_series.rolling(DIV_LOOKBACK_RB, min_periods=1).min().shift(1)

            high_now_close = close_series.rolling(DIV_LOOKBACK_RB, min_periods=1).max()
            high_prev_close = high_now_close.shift(1)
            rsi_high_prev = rsi_series.rolling(DIV_LOOKBACK_RB, min_periods=1).max().shift(1)

            # Calculate boolean conditions (handles NaNs gracefully by resulting in False)
            cond_reg = (low_now < low_prev) & (rsi_series > rsi_low_prev)
            cond_hidden = (high_now_close > high_prev_close) & (rsi_series < rsi_high_prev)

            # Assign results - converts boolean True/False to 1/0
            df['BULL_REG_RB'] = cond_reg.astype(int)
            df['BULL_HIDDEN_RB'] = cond_hidden.astype(int)
        # No else needed, initialized to 0

        df['BULL_DIV_RB'] = (df[['BULL_REG_RB', 'BULL_HIDDEN_RB']].any(axis=1)).astype(int)

        # Other Binary Filters (Requires BB_MID_RB and ADX_RB)
        if 'BB_MID_RB' in df.columns and not df['BB_MID_RB'].isnull().all():
            df['BULL_BB_RB'] = (df['CLOSE'] < df['BB_MID_RB']).astype(int)
        else:
            df['BULL_BB_RB'] = 0 # Default if BB_MID_RB calculation failed or column missing

        if 'ADX_RB' in df.columns and not df['ADX_RB'].isnull().all():
            df['ADX_OVER_THRESH_RB'] = (df['ADX_RB'] > ADX_THRESH_RB).astype(int)
        else:
            df['ADX_OVER_THRESH_RB'] = 0 # Default if ADX_RB calculation failed or column missing

        # Final Entry Flag
        df['BULLISH_ENTRY_RB'] = (
            (df['BULL_DIV_RB'] == 1) & (df['BULL_BB_RB'] == 1) & (df['ADX_OVER_THRESH_RB'] == 1)
        ).astype(int)

        logging.info("  Finished calculating Reversal-Bot specific features.")

    except Exception as e_rb:
        logging.error(f"!! Error calculating Reversal-Bot features: {e_rb}")
        traceback.print_exc()
        # Ensure columns exist even if calculation fails, filled with default/NaN/0
        rb_cols = ['RSI_RB', 'BB_MID_RB', 'ADX_RB', 'BULL_REG_RB', 'BULL_HIDDEN_RB',
                   'BULL_DIV_RB', 'BULL_BB_RB', 'ADX_OVER_THRESH_RB', 'BULLISH_ENTRY_RB']
        for col in rb_cols:
            if col not in df.columns:
                if col in ['BULL_REG_RB', 'BULL_HIDDEN_RB', 'BULL_DIV_RB', 'BULL_BB_RB', 'ADX_OVER_THRESH_RB', 'BULLISH_ENTRY_RB']:
                    df[col] = 0 # Default binary flags to 0
                else:
                    df[col] = np.nan # Default numeric indicators to NaN

    # --- Original Derived TA / Binary Features (using UPPERCASE names) ---
    # These use the standard TA features (RSI_14, BBP_20_2.0 etc.)
    atr_base_col = f'ATR_{atr_periods_min[0]}M'.upper() # Ensure ATR col name is upper
    if atr_base_col in df.columns and 'CLOSE' in df.columns:
        df['ATR_NORM'] = df[atr_base_col] / (df['CLOSE'] + EPS)
    else:
        df['ATR_NORM'] = np.nan

    # --- Derived TA Features from standard indicators ---
    rsi_col_upper = 'RSI_14'
    stochk_col_upper = 'STOCHK_14_3_3'
    macd_hist_col_upper = 'MACDH_12_26_9'
    bbp_col_upper = 'BBP_20_2.0'
    bbb_col_upper = 'BBB_20_2.0'

    # Initialize derived columns (using standard TA names)
    df['RSI_14_VALUE'] = 0.0; df['RSI_14_OVERSOLD'] = 0; df['RSI_14_OVERBOUGHT'] = 0; df['RSI_OB_CONFIRM'] = 0; df['RSI_OS_CONFIRM'] = 0
    df['STOCH_K_VALUE'] = 0.0; df['STOCH_K_OVERSOLD'] = 0; df['STOCH_K_OVERBOUGHT'] = 0
    df['MACD_HIST_VALUE'] = 0.0; df['MACD_HIST_POSITIVE'] = 0; df['MACD_HIST_INCREASING'] = 0
    df['BBP_VALUE'] = 0.0; df['BBP_NEAR_UPPER'] = 0; df['BBP_NEAR_LOWER'] = 0; df['BBB_WIDTH'] = 0.0

    if rsi_col_upper in df.columns and not df[rsi_col_upper].isnull().all():
        df['RSI_14_VALUE'] = df[rsi_col_upper]
        df['RSI_14_OVERSOLD'] = (df[rsi_col_upper] < 30).astype(int)
        df['RSI_14_OVERBOUGHT'] = (df[rsi_col_upper] > 70).astype(int)
        if 'CLOSE' in df and 'OPEN' in df:
            df['RSI_OB_CONFIRM'] = ((df[rsi_col_upper] > 70) & (df['CLOSE'] > df['OPEN'])).astype(int)
            df['RSI_OS_CONFIRM'] = ((df[rsi_col_upper] < 30) & (df['CLOSE'] < df['OPEN'])).astype(int)

    if stochk_col_upper in df.columns and not df[stochk_col_upper].isnull().all():
        df['STOCH_K_VALUE'] = df[stochk_col_upper]
        df['STOCH_K_OVERSOLD'] = (df[stochk_col_upper] < 20).astype(int)
        df['STOCH_K_OVERBOUGHT'] = (df[stochk_col_upper] > 80).astype(int)

    if macd_hist_col_upper in df.columns and not df[macd_hist_col_upper].isnull().all():
         df['MACD_HIST_VALUE'] = df[macd_hist_col_upper]
         df['MACD_HIST_POSITIVE'] = (df[macd_hist_col_upper] > 0).astype(int)
         macd_hist_numeric = pd.to_numeric(df[macd_hist_col_upper], errors='coerce')
         # Ensure shift aligns properly, handle NaNs
         df['MACD_HIST_INCREASING'] = (macd_hist_numeric > macd_hist_numeric.shift(1).fillna(method='bfill')).astype(int)

    if bbp_col_upper in df.columns and not df[bbp_col_upper].isnull().all():
        df['BBP_VALUE'] = df[bbp_col_upper]
        df['BBP_NEAR_UPPER'] = (df[bbp_col_upper] > 0.9).astype(int)
        df['BBP_NEAR_LOWER'] = (df[bbp_col_upper] < 0.1).astype(int)

    if bbb_col_upper in df.columns and not df[bbb_col_upper].isnull().all():
        df['BBB_WIDTH'] = df[bbb_col_upper]

    # --- RSI Divergence Proxy (using standard RSI_14) ---
    df['RSI_BULL_DIV_30M'] = 0
    if rsi_col_upper in df.columns and not df[rsi_col_upper].isnull().all():
        for n_div in [30]:
            if len(df) > n_div and 'LOW' in df.columns:
                # Use .copy() to avoid SettingWithCopyWarning
                low_series = df['LOW'].copy()
                rsi_14_series = df[rsi_col_upper].copy()

                min_price_n = low_series.rolling(window=n_div, min_periods=n_div//2).min()
                min_rsi_n = rsi_14_series.rolling(window=n_div, min_periods=n_div//2).min()
                price_lower_low = low_series < min_price_n.shift(1)
                rsi_higher_low = rsi_14_series > min_rsi_n.shift(1)

                # Combine conditions and assign
                df[f'RSI_BULL_DIV_{n_div}M'] = (price_lower_low & rsi_higher_low).astype(int)
            # No else needed, initialized to 0

    # --- Original Binary Interaction Features (using standard TA features) ---
    logging.info("  Calculating Original Binary Interaction features...")
    # ... (keep existing interaction features, ensuring they use correct UPPERCASE names like RSI_14_OVERSOLD etc.) ...
    df['INT_OS_BULL_CANDLE'] = 0
    if 'RSI_14_OVERSOLD' in df and 'STOCH_K_OVERSOLD' in df and 'CLOSE' in df and 'OPEN' in df and 'BODY_RATIO' in df:
        df['INT_OS_BULL_CANDLE'] = (((df['RSI_14_OVERSOLD'] == 1) | (df['STOCH_K_OVERSOLD'] == 1)) & (df['CLOSE'] > df['OPEN']) & (df['BODY_RATIO'] > 0.3)).astype(int)
    df['INT_VOL_SPIKE_BULL_CANDLE'] = 0
    if 'VOL_SPIKE_RATIO' in df and 'BODY_SPIKE_RATIO' in df and 'CLOSE' in df and 'OPEN' in df:
        df['INT_VOL_SPIKE_BULL_CANDLE'] = ((df['VOL_SPIKE_RATIO'] > 2.0) & (df['CLOSE'] > df['OPEN']) & (df['BODY_SPIKE_RATIO'] > 1.5)).astype(int)
    df['INT_BBP_UPPER_MACD_INC'] = 0
    if 'BBP_NEAR_UPPER' in df and 'MACD_HIST_INCREASING' in df:
        df['INT_BBP_UPPER_MACD_INC'] = ((df['BBP_NEAR_UPPER'] == 1) & (df['MACD_HIST_INCREASING'] == 1)).astype(int)
    df['INT_DIV_CONFIRM_MACD'] = 0
    div_confirm_col = f'RSI_BULL_DIV_30M' # Uses the standard RSI div proxy
    if div_confirm_col in df and 'MACD_HIST_POSITIVE' in df:
         df['INT_DIV_CONFIRM_MACD'] = ((df[div_confirm_col].shift(1).fillna(0) == 1) & (df['MACD_HIST_POSITIVE'] == 1)).astype(int) # Added fillna(0) for shift
    df['INT_OS_VOL_SPIKE'] = 0
    if 'RSI_14_OVERSOLD' in df and 'STOCH_K_OVERSOLD' in df and 'VOL_SPIKE_RATIO' in df:
         df['INT_OS_VOL_SPIKE'] = (((df['RSI_14_OVERSOLD'] == 1) | (df['STOCH_K_OVERSOLD'] == 1)) & (df['VOL_SPIKE_RATIO'] > 2.0)).astype(int)
    df['INT_BBP_UPPER_VOL_SPIKE'] = 0
    if 'BBP_NEAR_UPPER' in df and 'VOL_SPIKE_RATIO' in df:
         df['INT_BBP_UPPER_VOL_SPIKE'] = ((df['BBP_NEAR_UPPER'] == 1) & (df['VOL_SPIKE_RATIO'] > 2.0)).astype(int)


    # --- GPT Suggested Binary Interaction Features (using standard TA features) ---
    logging.info("  Calculating GPT Binary Interaction features...")
    # ... (keep existing GPT features, ensuring they use correct UPPERCASE names like RSI_14_OVERBOUGHT etc.) ...
    df['GPT_HIGH_VOL_AND_OB'] = 0
    if 'VOL_SPIKE_RATIO' in df and 'RSI_14_OVERBOUGHT' in df:
        df['GPT_HIGH_VOL_AND_OB'] = ((df['VOL_SPIKE_RATIO'] > 1.5) & (df['RSI_14_OVERBOUGHT'] == 1)).astype(int)
    df['GPT_MOMENTUM_MACD_CONFIRM'] = 0
    lag_col_gpt = 'LAG_5M_PRICE_RETURN'.upper()
    if lag_col_gpt in df and 'MACD_HIST_POSITIVE' in df:
        df['GPT_MOMENTUM_MACD_CONFIRM'] = ((df[lag_col_gpt] > 0) & (df['MACD_HIST_POSITIVE'] == 1)).astype(int)
    df['GPT_MULTI_TF_UPTREND'] = 0
    ma10_col = 'MA_10M'.upper(); ma30_col = 'MA_30M'.upper(); ma60_col = 'MA_60M'.upper()
    if ma10_col in df and ma30_col in df and ma60_col in df and 'CLOSE' in df:
         df['GPT_MULTI_TF_UPTREND'] = ((df['CLOSE'] > df[ma10_col]) & (df['CLOSE'] > df[ma30_col]) & (df['CLOSE'] > df[ma60_col])).astype(int)
    df['GPT_MEAN_REV_BODY_RSI'] = 0
    if 'RSI_14_OVERSOLD' in df and 'BODY_SPIKE_RATIO' in df:
        df['GPT_MEAN_REV_BODY_RSI'] = ((df['RSI_14_OVERSOLD'] == 1) & (df['BODY_SPIKE_RATIO'] > 1.2)).astype(int)
    df['GPT_VOL_PRICE_DIVERGENCE'] = 0
    lag_col_vp_gpt = 'LAG_10M_PRICE_RETURN'.upper()
    if 'VOL_SPIKE_RATIO' in df and lag_col_vp_gpt in df:
         df['GPT_VOL_PRICE_DIVERGENCE'] = ((df['VOL_SPIKE_RATIO'] > 1.5) & (df[lag_col_vp_gpt] < -0.2)).astype(int)
    df['GPT_BREAKOUT_VOL_ATR'] = 0
    range_col = 'PRICE_RANGE_PCT'.upper(); atr_norm_col = 'ATR_NORM'.upper()
    if range_col in df and atr_norm_col in df:
        # Ensure rolling means don't produce all NaNs
        rolling_range_mean = df[range_col].rolling(30, min_periods=15).mean()
        rolling_atr_norm_mean = df[atr_norm_col].rolling(30, min_periods=15).mean()
        # Check for NaN before comparison
        cond_breakout = (
            (df[range_col] > rolling_range_mean * 1.25) &
            (df[atr_norm_col] > rolling_atr_norm_mean) &
            rolling_range_mean.notna() &
            rolling_atr_norm_mean.notna()
        )
        df['GPT_BREAKOUT_VOL_ATR'] = cond_breakout.astype(int)

    df['GPT_STOCH_RSI_OVERBOUGHT'] = 0
    if 'STOCH_K_OVERBOUGHT' in df and 'RSI_14_OVERBOUGHT' in df:
        df['GPT_STOCH_RSI_OVERBOUGHT'] = ((df['STOCH_K_OVERBOUGHT'] == 1) & (df['RSI_14_OVERBOUGHT'] == 1)).astype(int)
    df['GPT_BULL_DIVERGENCE_STRONG'] = 0
    div_strong_col = 'RSI_BULL_DIV_30M' # Uses standard RSI div proxy
    if div_strong_col in df and 'VOL_SPIKE_RATIO' in df:
         df['GPT_BULL_DIVERGENCE_STRONG'] = ((df[div_strong_col] == 1) & (df['VOL_SPIKE_RATIO'] > 1.3)).astype(int)


    # --- Cleanup ---
    # Keep the original cleanup logic. The new RB features will remain unless explicitly added here.
    cols_to_drop_pre_upper = ['price_change_1m_temp', 'volume_return_1m', 'body_abs', 'range', 'vol_ma_20m', 'body_ma_20m', 'prev_close', 'hml', 'hmpc', 'lmpc', 'tr']
    # Consider if you want to drop the base TA features (RSI_14, BBP_20_2.0 etc.) or keep them alongside the RB features.
    # Keeping them for now to have both sets available.
    # base_ta_cols_upper = [c for c in [rsi_col_upper, stochk_col_upper, macd_hist_col_upper, bbp_col_upper, bbb_col_upper, atr_base_col] if c is not None and c in df.columns]
    final_cols_to_drop = [c.upper() for c in cols_to_drop_pre_upper] # Ensure drop list is uppercase
    # final_cols_to_drop.extend(base_ta_cols_upper) # Uncomment to drop base TA
    df = df.drop(columns=[col for col in final_cols_to_drop if col in df.columns], errors='ignore')

    # --- Final check for infinities ---
    numeric_cols_final = df.select_dtypes(include=np.number).columns
    if not numeric_cols_final.empty and df[numeric_cols_final].isin([np.inf, -np.inf]).any().any():
        inf_count = df[numeric_cols_final].isin([np.inf, -np.inf]).sum().sum()
        logging.warning(f"  Warning: {inf_count} Infinities detected after feature calculation. Replacing with NaN.")
        df = df.replace([np.inf, -np.inf], np.nan)

    logging.info(f"  Feature Eng End: Total columns = {df.shape[1]}, Rows = {len(df)}")
    # Log the final columns being used
    final_feature_list = [col for col in df.columns if col not in ['TIMESTAMP', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUMEFROM', 'VOLUMETO'] and not col.startswith('TARGET_')]
    logging.info(f"  Available Feature Columns ({len(final_feature_list)}): {final_feature_list[:10]}...") # Log first few features

    return df


# ==============================================================================
# --- Optuna Objective Function (Unchanged) ---
# ==============================================================================
def objective(trial, X, y, fixed_params, cv_strategy):
    # ... (function remains the same) ...
    param = {
        "max_depth":        trial.suggest_int("max_depth", 4, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 2.24, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1.0, 20.0, log=True),
        "gamma":            trial.suggest_float("gamma", 0, 2.24),
        "subsample":        trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.95),
        "learning_rate":    trial.suggest_float("learning_rate", 0.002, 0.2, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 250, log=True),
    }
    xgb_params = {**fixed_params, **param}
    cv_scores = []; fold = -1
    try:
        y_np = y.to_numpy() if isinstance(y, pd.Series) else np.array(y)
        X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else np.array(X)
        for fold, (train_idx, val_idx) in enumerate(cv_strategy.split(X_np, y_np)):
            X_train_fold, X_val_fold = X_np[train_idx], X_np[val_idx]
            y_train_fold, y_val_fold = y_np[train_idx], y_np[val_idx]
            X_train_resampled, y_train_resampled = X_train_fold, y_train_fold
            if len(np.unique(y_train_fold)) >= 2:
                minority_count = np.sum(y_train_fold == 1)
                majority_count = np.sum(y_train_fold == 0)
                if minority_count >= 6 and majority_count >=6: # SMOTEENN safe condition
                    try:
                        smoteenn_random_state = fixed_params.get("random_state", 42) + fold
                        sampler = SMOTEENN(random_state=smoteenn_random_state)
                        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_fold, y_train_fold)
                    except ValueError as e_smoteenn: # Catch specific ValueError
                        logging.warning(f"  SMOTEENN Error fold {fold+1}: {e_smoteenn}. Training on original data.")
                        X_train_resampled, y_train_resampled = X_train_fold, y_train_fold
                    except Exception as e_smoteenn_other: # Catch other potential errors
                         logging.error(f"  Unexpected SMOTEENN Error fold {fold+1}: {e_smoteenn_other}. Training on original data.")
                         X_train_resampled, y_train_resampled = X_train_fold, y_train_fold

            if len(np.unique(y_train_resampled)) < 2:
                 logging.warning(f"  Skipping fold {fold+1} training: Single class after potential resampling.")
                 cv_scores.append(0.0); continue # Append 0 score for this fold

            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_train_resampled, y_train_resampled, verbose=False)
            preds_proba = model.predict_proba(X_val_fold)[:, 1]
            preds_binary = (preds_proba >= OPTUNA_EVAL_THRESHOLD).astype(int)
            precision = precision_score(y_val_fold, preds_binary, zero_division=0)
            cv_scores.append(precision)
            trial.report(precision, fold)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()

        # Handle case where all folds were skipped or resulted in errors
        average_score = np.mean([s for s in cv_scores if not np.isnan(s)]) if cv_scores and not all(np.isnan(s) for s in cv_scores) else 0.0

    except optuna.exceptions.TrialPruned:
        raise # Re-raise TrialPruned
    except Exception as e:
        logging.error(f"Error in Optuna trial {trial.number}, fold {fold+1}: {e}")
        traceback.print_exc() # Print stack trace for Optuna errors
        return 0.0 # Return 0 score on other errors

    return average_score


# ==============================================================================
# --- Main Online Prediction Loop (Largely Unchanged, will use all features) ---
# ==============================================================================
def main_prediction_loop():
    """Main loop to fetch data, train model, predict, and notify approx every 5 minutes."""
    last_processed_timestamp = None
    conn = None

    while True:
        loop_start_time = time.time()
        prediction_made = False
        logging.info(f"--- Starting 5-Minute Cycle ---")

        try:
            # --- 1. Connect and Fetch Data ---
            if conn is None or conn.closed != 0:
                 logging.info("Attempting to connect to database...")
                 conn = connect_db(DB_PARAMS)
                 if conn is None: # Check if connect_db returned None
                      logging.error("Database connection failed. Skipping this cycle.")
                      time.sleep(60) # Wait longer after connection failure
                      continue
                 else:
                      logging.info("Database connection established successfully.")

            df_raw_data = fetch_latest_data(conn, SYMBOL_NAME, DB_TABLE_NAME, MINUTES_TO_FETCH)

            if df_raw_data is None: # Check return from fetch_latest_data
                logging.error("Database fetch failed or returned None. Skipping cycle.")
                if conn:
                    try: conn.close(); logging.warning("Closed DB connection after fetch failure.")
                    except Exception: pass
                conn = None
                time.sleep(60) # Wait longer after fetch failure
                continue

            if df_raw_data.empty:
                logging.warning("No data fetched or empty DataFrame returned. Skipping prediction.")
                # Allow loop timing to handle wait
            else:
                latest_available_timestamp = df_raw_data['timestamp'].iloc[-1]

                if last_processed_timestamp is None or latest_available_timestamp > last_processed_timestamp:
                    logging.info(f"New data detected. Latest available timestamp: {latest_available_timestamp}")

                    # --- 2. Feature Engineering (Now includes RB features) ---
                    start_fe = time.time()
                    df_features = calculate_features_min_rare_event(df_raw_data) # Returns UPPERCASE cols
                    if df_features.empty:
                         logging.error("Feature calculation resulted in empty DataFrame. Skipping.")
                         last_processed_timestamp = latest_available_timestamp # Update timestamp even on skip
                         continue
                    logging.info(f"Feature engineering complete. Took {time.time() - start_fe:.2f}s.")

                    # --- 3. Define Target Variable ---
                    target_col = f'TARGET_RETURN_{PREDICTION_WINDOW_MINUTES}M' # UPPERCASE
                    if 'CLOSE' in df_features.columns:
                        close_series = df_features['CLOSE']
                        # Calculate target, avoiding division by zero or near-zero close prices
                        df_features[target_col] = close_series.shift(-PREDICTION_WINDOW_MINUTES).sub(close_series).div(close_series.replace(0, np.nan) + EPS).mul(100)
                    else:
                        logging.error("Critical Error: 'CLOSE' column not found after FE.")
                        last_processed_timestamp = latest_available_timestamp
                        continue

                    # --- 4. Prepare Data for Modeling ---
                    # Define columns to exclude (base OHLCV, timestamp, target, intermediate TA)
                    base_cols_ohlcv_upper = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUMEFROM', 'VOLUMETO']
                    intermediate_ta_cols = [ # Add cols you definitely don't want as direct features
                        'TIMESTAMP', target_col] + base_cols_ohlcv_upper + \
                        [c for c in df_features.columns if 'STOCHD_' in c.upper()] + \
                        [c for c in df_features.columns if 'MACDS_' in c.upper()] + \
                        [c for c in df_features.columns if c.upper() == 'BBL_20_2.0'] + \
                        [c for c in df_features.columns if c.upper() == 'BBU_20_2.0'] + \
                        [c for c in df_features.columns if c.upper() == 'BBM_20_2.0'] # Example: Exclude BB components if only using BBP/BBB

                    cols_to_exclude_features = list(set(intermediate_ta_cols)) # Ensure unique

                    # Select features: numeric, not excluded, not datetime
                    final_feature_cols = sorted([ # Sort for consistency
                        col for col in df_features.columns if col not in cols_to_exclude_features
                        and not pd.api.types.is_datetime64_any_dtype(df_features[col])
                        and pd.api.types.is_numeric_dtype(df_features[col])
                        and df_features[col].nunique() > 1 # Exclude constant columns
                    ])


                    if not final_feature_cols:
                        logging.error("No valid numeric features found after selection/filtering. Skipping.")
                        last_processed_timestamp = latest_available_timestamp
                        continue

                    logging.info(f"Using {len(final_feature_cols)} features. First few: {final_feature_cols[:5]}")

                    # Prepare data for the LATEST prediction
                    if not df_features.empty:
                        # Select only the final available row *after* target calc (which introduces NaNs at end)
                        # Ensure features are calculated *before* selecting the row for prediction
                        latest_feature_row = df_features.iloc[-1 - PREDICTION_WINDOW_MINUTES] if len(df_features) > PREDICTION_WINDOW_MINUTES else df_features.iloc[-1]
                        X_predict_latest = latest_feature_row[final_feature_cols].to_frame().T.copy() # Get as DF row
                        X_predict_latest.replace([np.inf, -np.inf], np.nan, inplace=True)
                        # Fill NaNs - Use 0 for simplicity online, or consider training mean/median
                        X_predict_latest.fillna(0, inplace=True) # Simple fillna for prediction
                    else:
                        logging.warning("Feature DataFrame empty before prediction prep.")
                        last_processed_timestamp = latest_available_timestamp
                        continue

                    # Prepare data for TRAINING
                    df_train_ready = df_features.copy()
                    rows_before_drop = len(df_train_ready)
                    # Columns required for training (target + all selected features)
                    cols_for_nan_drop = [target_col] + final_feature_cols
                    df_train_ready.replace([np.inf, -np.inf], np.nan, inplace=True)
                    df_train_ready.dropna(subset=cols_for_nan_drop, inplace=True)
                    rows_after_drop = len(df_train_ready)
                    logging.info(f"Dropped {rows_before_drop - rows_after_drop} rows with NaNs/Infs in target or features for training.")

                    if target_col not in df_train_ready.columns: # Should not happen if check above passed
                        logging.error(f"Target column '{target_col}' missing after NaN drop. Skipping.")
                        last_processed_timestamp = latest_available_timestamp
                        continue

                    if len(df_train_ready) < TRAIN_WINDOW_MINUTES // 2: # Check if enough data remains
                        logging.warning(f"Insufficient training data ({len(df_train_ready)} rows) after NaN drop (min required: {TRAIN_WINDOW_MINUTES // 2}). Skipping.")
                        last_processed_timestamp = latest_available_timestamp
                        continue

                    # Select the training window
                    df_train_final = df_train_ready.tail(TRAIN_WINDOW_MINUTES)
                    if df_train_final.empty:
                         logging.warning("Training data empty after tail selection. Skipping.")
                         last_processed_timestamp = latest_available_timestamp
                         continue

                    X_train = df_train_final[final_feature_cols]
                    y_train_binary = (df_train_final[target_col] >= TARGET_THRESHOLD_PCT).astype(int)
                    train_positive_rate = y_train_binary.mean() * 100
                    logging.info(f"Final training data shape: {X_train.shape}, Target shape: {y_train_binary.shape}")
                    logging.info(f"Positive Target Rate in Training Data: {train_positive_rate:.2f}%")

                    # Check for single class issue again after all processing
                    if y_train_binary.nunique() < 2:
                        logging.warning("Only one class present in final training data. Skipping.")
                        last_processed_timestamp = latest_available_timestamp
                        continue

                    # --- 5. Hyperparameter Tuning with Optuna ---
                    logging.info(f"Running Optuna ({N_OPTUNA_TRIALS} trials, cv={OPTUNA_CV_SPLITS} TSS+SMOTEENN+ScalePosWeight, scoring Precision@{OPTUNA_EVAL_THRESHOLD})...")
                    optuna_start_time = time.time()
                    best_params_step = None
                    study = None # Initialize study variable
                    try:
                        cv_strategy = TimeSeriesSplit(n_splits=OPTUNA_CV_SPLITS)
                        # Configure pruner
                        pruner_warmup = max(1, N_OPTUNA_TRIALS // 10); # Warmup for 10% of trials
                        pruner_min_trials = max(5, N_OPTUNA_TRIALS // 5) # Prune after 20% or 5 trials
                        pruner = optuna.pruners.MedianPruner(n_warmup_steps=pruner_warmup, n_min_trials=pruner_min_trials)
                        sampler = optuna.samplers.TPESampler(seed=int(time.time())) # New seed each time
                        study = optuna.create_study(direction='maximize', pruner=pruner, sampler=sampler)
                        obj_func = lambda trial: objective(trial, X_train, y_train_binary, XGB_FIXED_PARAMS, cv_strategy)
                        study.optimize(obj_func, n_trials=N_OPTUNA_TRIALS, n_jobs=1, show_progress_bar=False) # n_jobs=1 recommended with SMOTEENN inside objective

                        # Check if study completed successfully and has trials
                        if study.best_trial:
                             best_params_step = study.best_params
                             best_score_step = study.best_value
                             logging.info(f"Optuna finished in {time.time() - optuna_start_time:.2f}s.")
                             logging.info(f"Best Params: {best_params_step}, Best CV Precision(@{OPTUNA_EVAL_THRESHOLD}): {best_score_step:.4f}")
                        else:
                             logging.warning("Optuna study completed without any successful trials.")
                             best_params_step = None # Ensure it's None

                    except Exception as e_optuna:
                         logging.error(f"!! Error during Optuna optimization: {e_optuna}")
                         traceback.print_exc()
                         best_params_step = None # Ensure it's None on error
                         # Continue to finally block

                    # --- 6. Fit Final Model & Predict ---
                    if best_params_step is None:
                        logging.error("Optuna did not produce best parameters (or failed). Skipping prediction.")
                        last_processed_timestamp = latest_available_timestamp
                        continue # Skip fitting/prediction

                    final_model_params = {**XGB_FIXED_PARAMS, **best_params_step}
                    logging.info("Fitting final model on training data...")
                    model_final = xgb.XGBClassifier(**final_model_params)
                    try:
                        # Fit on the original (non-resampled) training data before prediction
                        model_final.fit(X_train, y_train_binary, verbose=False)

                        # Log feature importances
                        if hasattr(model_final, 'feature_importances_'):
                             feature_importances = pd.Series(model_final.feature_importances_, index=X_train.columns).sort_values(ascending=False)
                             logging.info(f"Top 10 Feature Importances:\n{feature_importances.head(10)}")

                        # Ensure prediction data is ready and columns match
                        if not X_predict_latest.empty and list(X_train.columns) == list(X_predict_latest.columns):
                             prediction_proba = model_final.predict_proba(X_predict_latest)[:, 1][0]

                             # --- 7. Output Prediction & Send Notification ---
                             logging.info(f"--- PREDICTION for {latest_available_timestamp} ({SYMBOL_NAME}) ---")
                             logging.info(f"Predicted Probability (P({target_col} >= {TARGET_THRESHOLD_PCT}%)): {prediction_proba:.6f}")

                             # *** Check threshold and send ntfy notification ***
                             if prediction_proba >= NOTIFICATION_THRESHOLD:
                                 logging.info(f"Probability {prediction_proba:.4f} >= threshold {NOTIFICATION_THRESHOLD}. Sending notification to topic '{NTFY_TOPIC}'.")
                                 notif_title = f"{SYMBOL_NAME} Alert {prediction_proba:.2%} (>{TARGET_THRESHOLD_PCT}%)" # More specific title
                                 notif_message = (
                                     f"{SYMBOL_NAME} pred prob: {prediction_proba:.4f}\n"
                                     f"Time: {latest_available_timestamp}\n"
                                     f"Target: >={TARGET_THRESHOLD_PCT}% in {PREDICTION_WINDOW_MINUTES}m\n"
                                     f"Threshold: {NOTIFICATION_THRESHOLD:.2f}" # Add threshold info
                                 )
                                 # Use priority 5 (urgent) for high probability alerts
                                 send_ntfy_notification(title=notif_title, message=notif_message, priority=5)
                             # *** End notification logic ***

                             prediction_made = True
                        else:
                             logging.error("Prediction input empty or columns mismatch after final check. Cannot predict.")
                             if list(X_train.columns) != list(X_predict_latest.columns):
                                 logging.error(f"Train cols ({len(X_train.columns)}) vs Predict cols ({len(X_predict_latest.columns)})")
                                 # Log differences if needed for debug

                    except Exception as e_fit_predict:
                        logging.error(f"!! Error during final model fitting or prediction: {e_fit_predict}")
                        traceback.print_exc()

                    # Update timestamp only after a successful processing attempt (even if prediction failed)
                    last_processed_timestamp = latest_available_timestamp
                else:
                    logging.info(f"No new data since {last_processed_timestamp}. Waiting for next cycle.")

        # --- Exception Handling & Finally block ---
        except psycopg2.OperationalError as db_op_error:
            logging.error(f"Database Operational Error: {db_op_error}. Attempting reconnect on next cycle.")
            if conn:
                try: conn.close(); logging.warning("Closed potentially problematic DB connection.")
                except Exception: pass
            conn = None; time.sleep(60) # Wait longer after DB error
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received. Exiting.")
            break
        except Exception as e_main:
            logging.error(f"!! Unexpected error in main loop: {e_main}")
            traceback.print_exc()
            if conn: # Ensure connection is closed on unexpected error
                try: conn.close(); logging.warning("Closed DB connection after unexpected error.")
                except Exception: pass
            conn = None; time.sleep(60) # Wait longer after general error
        finally:
            # Close connection *only if* a prediction was made in this cycle (or keep open)
            # Decide based on preference: close frequently vs potentially faster but riskier keep-alive
            if conn and conn.closed == 0: # Check if connection exists and is open
                 if prediction_made: # Example: Close only after work is done
                      try:
                          conn.close();
                          logging.info("DB connection closed after successful prediction cycle.")
                          conn = None # Set to None after closing
                      except Exception as e_close:
                          logging.warning(f"Error closing DB connection: {e_close}"); conn = None
                 # else: # Keep connection open if no prediction was made
                 #    pass

            # --- Loop timing ---
            loop_end_time = time.time()
            elapsed_time = loop_end_time - loop_start_time
            wait_time = max(0, TARGET_CYCLE_TIME_SECONDS - elapsed_time)
            logging.info(f"Cycle took {elapsed_time:.2f}s. Waiting {wait_time:.2f}s for next target interval...")
            time.sleep(wait_time)

    # Cleanup on exit
    if conn and conn.closed == 0:
        try: conn.close(); logging.info("Closed database connection on exit.")
        except Exception: pass

# ==============================================================================
# --- Script Execution ---
# ==============================================================================
if __name__ == "__main__":
    # --- Prerequisite Check ---
    if not NTFY_TOPIC:
        logging.warning("NTFY_TOPIC is not set in the configuration. Notifications will be disabled.")
        # Optionally exit if notifications are critical:
        # import sys
        # sys.exit("Error: NTFY_TOPIC is required.")

    logging.info(f"--- Starting Online Prediction Script for {SYMBOL_NAME} (Combined Features) ---")
    logging.info(f"Prediction Window: {PREDICTION_WINDOW_MINUTES} mins, Target Threshold: {TARGET_THRESHOLD_PCT}%")
    logging.info(f"Fetching {MINUTES_TO_FETCH} minutes, Training on {TRAIN_WINDOW_MINUTES} minutes, Max Feature Lag: {MAX_FEATURE_LAG}")
    logging.info(f"Targeting a full run cycle every {TARGET_CYCLE_TIME_SECONDS} seconds.")
    logging.warning(f"Optuna configured for {N_OPTUNA_TRIALS} trials per cycle (Eval Threshold: {OPTUNA_EVAL_THRESHOLD}). Monitor resource usage.")
    logging.info(f"Notifications will be sent to ntfy.sh topic '{NTFY_TOPIC}' when probability >= {NOTIFICATION_THRESHOLD}")

    main_prediction_loop() # Start the main loop

    logging.info(f"\n{'='*30} Script Stopped for {SYMBOL_NAME} {'='*30}")