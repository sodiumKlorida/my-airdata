import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
import json
import os
from datetime import datetime, date
from zoneinfo import ZoneInfo
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from typing import Dict, List, Tuple, Optional, Any
from DB.config import create_connection


def pm25_to_aqi(pm25_value: float) -> float:
    """Konversi PM2.5 (µg/m³) ke Air Quality Index (AQI)"""
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for (c_low, c_high, aqi_low, aqi_high) in breakpoints:
        if c_low <= pm25_value <= c_high:
            aqi = ((aqi_high - aqi_low) / (c_high - c_low)) * (pm25_value - c_low) + aqi_low
            return round(aqi, 2)
    return 500.0

class AirQualityDataFetcher:
    """Handles fetching air quality data from the API."""
    
    def __init__(self, base_url: str = "https://airnet.waqi.info/airnet/sse/historic/daily/420154"):
        self.base_url = base_url
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.pollutants = ['pm25', 'pm10', 'co', 'no2', 'o3', 'so2']
    
    def fetch_pollutant_data(self, pollutant: str) -> pd.DataFrame:
        """Fetch data for a specific pollutant."""
        url = f"{self.base_url}?specie={pollutant}"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code != 200:
            raise Exception(f"Failed to fetch data for {pollutant}: {response.status_code}")
        
        text_data = response.text
        raw_matches = re.findall(r'data:\s*(\{.*?\})', text_data)
        
        day_list = []
        median_list = []
        
        for obj in raw_matches:
            cleaned = obj.strip()
            cleaned = cleaned.replace("NaN", "null").replace("Infinity", "null")
            cleaned = re.sub(r",\s*}", "}", cleaned)
            
            try:
                data = json.loads(cleaned)
                if 'day' in data and 'median' in data:
                    day_list.append(data['day'])
                    median_list.append(data['median'])
            except json.JSONDecodeError:
                continue
        
        df = pd.DataFrame({
            'day': day_list,
            'median': median_list
        })
        
        print(f"{pollutant}: {len(df)} data points successfully fetched.")
        return df
    
    def fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch data for all pollutants."""
        data_dict = {}
        
        for pollutant in self.pollutants:
            try:
                data_dict[pollutant] = self.fetch_pollutant_data(pollutant)
            except Exception as e:
                print(f"Error fetching {pollutant}: {e}")
                continue
        
        return data_dict


class DataPreprocessor:
    """Handles data preprocessing and cleaning."""
    
    def __init__(self):
        self.wib = ZoneInfo("Asia/Jakarta")
    
    def merge_pollutant_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge all pollutant data into a single DataFrame."""
        df_merged = None
        
        for pollutant, df in data_dict.items():
            df = df.rename(columns={'median': pollutant})
            if df_merged is None:
                df_merged = df
            else:
                df_merged = pd.merge(df_merged, df, on='day', how='outer')
        
        df_merged = df_merged.sort_values('day').reset_index(drop=True)
        return df_merged
    
    def clean_and_interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data and perform interpolation."""
        # Linear interpolation
        df_cleaned = df.interpolate(method='linear', limit_direction='both')
        # Forward and backward fill for remaining NaNs
        df_cleaned = df_cleaned.fillna(method='ffill').fillna(method='bfill')
        return df_cleaned
    
    def remove_today_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove today's data from the DataFrame."""
        today_wib = datetime.now(self.wib).date()
        print(f"\nToday date is:{today_wib}")
        df['day'] = pd.to_datetime(df['day'], format='%Y-%m-%d')
        today = pd.to_datetime(today_wib)
        
        df_filtered = df[df['day'] != today].copy()
        return df_filtered
    
    def reorder_columns(self, df: pd.DataFrame, new_order: List[str]) -> pd.DataFrame:
        """Reorder DataFrame columns."""
        return df[new_order].copy()
    
    def preprocess_data(self, data_dict: Dict[str, pd.DataFrame], 
                       column_order: List[str] = None) -> pd.DataFrame:
        """Complete preprocessing pipeline."""
        if column_order is None:
            column_order = ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']
        
        # Merge data
        df_merged = self.merge_pollutant_data(data_dict)
        
        # Clean and interpolate
        df_cleaned = self.clean_and_interpolate(df_merged)
        
        # Remove today's data
        df_filtered = self.remove_today_data(df_cleaned)
        
        # Reorder columns (excluding 'day' column)
        df_final = self.reorder_columns(df_filtered, column_order)
        
        return df_final


class LSTMModel:
    """LSTM model builder and trainer."""
    
    @staticmethod
    def build_lstm(units: int = 50, dropout: float = 0.2, 
                   lr: float = 0.001, input_shape: Tuple[int, int] = (3, 1)) -> Sequential:
        """Build LSTM model architecture."""
        model = Sequential()
        model.add(LSTM(units, input_shape=input_shape, kernel_regularizer=l2(1e-4)))
        model.add(Dropout(dropout))
        model.add(Dense(1))
        
        optimizer = Adam(learning_rate=lr)
        model.compile(loss="mse", optimizer=optimizer)
        return model


class ModelManager:
    """Manages model loading, saving, and retraining."""
    
    def __init__(self, model_dir: str, retrained_dir: str = None):
        self.model_dir = model_dir
        self.retrained_dir = retrained_dir or os.path.join(model_dir, "Retrained")
        os.makedirs(self.retrained_dir, exist_ok=True)
    
    def load_model_components(self, pollutant: str) -> Tuple[Any, Dict, Any]:
        """Load model, parameters, and scaler for a pollutant."""
        # Load parameters
        param_path = os.path.join(self.model_dir, f"best_params_{pollutant.upper()}_1.json")
        with open(param_path, "r") as f:
            best_params_dict = json.load(f)
        
        # Load model
        model_path = os.path.join(self.model_dir, f"final_model_{pollutant.upper()}_1.keras")
        model = load_model(model_path)
        
        # Load scaler
        scaler_path = os.path.join(self.model_dir, f"scaler_{pollutant.upper()}.pkl")
        scaler = joblib.load(scaler_path)
        
        return model, best_params_dict, scaler
    
    def retrain_model(self, pollutant: str, df_supervised: pd.DataFrame, 
                     best_params_dict: Dict, original_model: Any, 
                     original_scaler: Any, lag_offsets: List[int]) -> Tuple[Any, Any, pd.DataFrame]:
        """Retrain model with updated data."""
        # Create lag features
        df_supervised_final = df_supervised[['y']].copy()
        for offset in lag_offsets:
            df_supervised_final[f'lag_{offset}'] = df_supervised_final['y'].shift(offset)
        df_supervised_final = df_supervised_final.dropna().reset_index(drop=True)
        
        # Prepare features
        feature_columns = ['y'] + [f'lag_{offset}' for offset in lag_offsets]
        
        # Scaling
        scaled_data = original_scaler.transform(df_supervised_final[feature_columns])
        X = scaled_data[:, 1:]  # lag columns
        y = scaled_data[:, 0]   # target column
        X = X.reshape(X.shape[0], len(lag_offsets), 1)
        
        # Split train/validation
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Build and retrain model
        retrained_model = LSTMModel.build_lstm(
            units=int(best_params_dict["units"]),
            dropout=best_params_dict["dropout"],
            lr=best_params_dict["learning_rate"],
            input_shape=(len(lag_offsets), 1)
        )
        
        retrained_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=int(best_params_dict["epochs"]),
            batch_size=32,
            verbose=1
        )
        
        print(f"Retraining completed for {pollutant}! Total samples: {len(df_supervised_final)}")
        
        return retrained_model, original_scaler, df_supervised_final
    
    def save_retrained_model(self, pollutant: str, model: Any) -> str:
        """Save retrained model."""
        model_name = f"{pollutant}_retrained.keras"
        save_path = os.path.join(self.retrained_dir, model_name)
        model.save(save_path)
        print(f"Retrained model saved as: {save_path}")
        return save_path


class AirQualityForecaster:
    """Handles sequential forecasting with retraining."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    def sequential_forecast_with_retrain(self, pollutant: str, df_supervised: pd.DataFrame,
                                       best_params: List, original_scaler: Any, 
                                       original_model: Any, num_steps: int = 7,
                                       lag_offsets: List[int] = [1, 2, 3]) -> Tuple[List, Any, Any, pd.DataFrame]:
        """Perform sequential forecasting with retraining at each step."""
        num_lags = len(lag_offsets)
        num_features = num_lags + 1
        extended_df = df_supervised.copy()
        
        # Create lag columns if missing
        feature_columns = ['y'] + [f'lag_{offset}' for offset in lag_offsets]
        missing_lags = [f'lag_{offset}' for offset in lag_offsets if f'lag_{offset}' not in extended_df.columns]
        
        if missing_lags:
            print(f"Creating missing lag columns: {missing_lags}")
            if 'y' not in extended_df.columns:
                raise ValueError("df_supervised must have column 'y'.")
            
            max_offset = max(lag_offsets)
            if len(extended_df) < max_offset:
                raise ValueError(f"Data length ({len(extended_df)}) is too short for max lag offset ({max_offset})")
            
            for offset in lag_offsets:
                col_name = f'lag_{offset}'
                extended_df[col_name] = extended_df['y'].shift(offset)
            
            initial_len = len(extended_df)
            extended_df = extended_df.dropna().reset_index(drop=True)
            dropped_rows = initial_len - len(extended_df)
            if dropped_rows > 0:
                print(f"Dropped {dropped_rows} rows with NaN after creating lags.")
        
        current_scaler = original_scaler
        current_model = original_model
        forecasts = []
        
        # Initialize last_lags_unscaled
        last_lags_unscaled = []
        for offset in lag_offsets:
            col_name = f'lag_{offset}'
            last_lags_unscaled.append(extended_df[col_name].iloc[-1])
        last_lags_unscaled = np.array(last_lags_unscaled)
        
        # Dummy for initial scaled lags
        dummy = np.zeros((1, num_features))
        dummy[0, 1:] = last_lags_unscaled
        dummy[0, 0] = extended_df['y'].iloc[-1]
        scaled_dummy = current_scaler.transform(dummy)
        last_row_scaled = scaled_dummy[0, 1:]
        
        for day in range(num_steps):
            # Predict
            current_seq = last_row_scaled.reshape(1, num_lags, 1)
            pred_scaled = current_model.predict(current_seq, verbose=0)[0][0]
            
            # Inverse transform
            dummy_inv = np.zeros((1, num_features))
            dummy_inv[0, 0] = pred_scaled
            dummy_inv[0, 1:] = last_lags_unscaled
            pred_original = current_scaler.inverse_transform(dummy_inv)[0, 0]
            forecasts.append(pred_original)
            
            # Create new row
            new_row_dict = {'y': pred_original}
            for offset in lag_offsets:
                prev_index = len(extended_df) - offset
                if prev_index >= 0:
                    lag_value = extended_df['y'].iloc[prev_index]
                else:
                    lag_value = 0.0
                    print(f"Warning: Insufficient data for lag_{offset} at day {day+1}, using 0.")
                new_row_dict[f'lag_{offset}'] = lag_value
            
            new_row = pd.DataFrame([new_row_dict])
            extended_df = pd.concat([extended_df, new_row], ignore_index=True)
            
            # Retrain model
            total_len = len(extended_df)
            train_size = int(total_len * 0.8)
            val_size = total_len - train_size
            
            if val_size < 1:
                val_size = 1
                train_size = total_len - 1
            
            train_ext = extended_df.iloc[:train_size]
            val_ext = extended_df.iloc[train_size:]
            
            # Create new scaler
            new_scaler = MinMaxScaler()
            train_ext_scaled = new_scaler.fit_transform(train_ext[feature_columns])
            val_ext_scaled = new_scaler.transform(val_ext[feature_columns])
            
            X_train_ext = train_ext_scaled[:, 1:]
            y_train_ext = train_ext_scaled[:, 0]
            X_val_ext = val_ext_scaled[:, 1:]
            y_val_ext = val_ext_scaled[:, 0]
            
            X_train_ext = X_train_ext.reshape(X_train_ext.shape[0], num_lags, 1)
            X_val_ext = X_val_ext.reshape(X_val_ext.shape[0], num_lags, 1)
            
            # Build and retrain model
            retrained = LSTMModel.build_lstm(
                units=int(best_params[0]), 
                dropout=best_params[1], 
                lr=best_params[2], 
                input_shape=(num_lags, 1)
            )
            retrained.fit(
                X_train_ext, y_train_ext,
                validation_data=(X_val_ext, y_val_ext),
                epochs=int(best_params[3]),
                batch_size=32, 
                verbose=0
            )
            current_model = retrained
            current_scaler = new_scaler
            
            # Update for next iteration
            new_lags_unscaled = []
            for offset in lag_offsets:
                new_lags_unscaled.append(extended_df[f'lag_{offset}'].iloc[-1])
            new_lags_unscaled = np.array(new_lags_unscaled)
            
            dummy_next = np.zeros((1, num_features))
            dummy_next[0, 1:] = new_lags_unscaled
            dummy_next[0, 0] = pred_original
            scaled_dummy_next = current_scaler.transform(dummy_next)
            last_row_scaled = scaled_dummy_next[0, 1:]
        
        return forecasts, current_model, current_scaler, extended_df


class AirQualityPipeline:
    """Main pipeline class that orchestrates the entire air quality forecasting process."""
    
    def __init__(self, model_dir: str, drive_mount_path: str = None):
        self.model_dir = model_dir
        self.drive_mount_path = drive_mount_path
        
        # Initialize components
        self.data_fetcher = AirQualityDataFetcher()
        self.preprocessor = DataPreprocessor()
        self.model_manager = ModelManager(model_dir)
        self.forecaster = AirQualityForecaster(self.model_manager)
        
        # Mount Google Drive if path provided
        if drive_mount_path:
            self._mount_drive()
    
    def _mount_drive(self):
        """Mount Google Drive if using Colab."""
        try:
            from google.colab import drive
            drive.mount(self.drive_mount_path)
            print("Google Drive mounted successfully")
        except ImportError:
            print("Google Colab not detected, skipping drive mount")
    
    def fetch_and_preprocess_data(self) -> pd.DataFrame:
        """Fetch and preprocess air quality data."""
        print("Fetching air quality data...")
        data_dict = self.data_fetcher.fetch_all_data()
        
        print("Preprocessing data...")
        df_final = self.preprocessor.preprocess_data(data_dict)
        
        return df_final
    
    def retrain_all_models(self, df_final: pd.DataFrame) -> Dict[str, str]:
        """Retrain all pollutant models."""
        print("Retraining all models...")
        retrained_models = {}
        
        pollutants = df_final.columns.tolist()
        
        for pollutant in pollutants:
            print(f"Retraining {pollutant} model...")
            
            # Prepare data
            data_tis = df_final[pollutant].copy()
            df_supervised = pd.DataFrame({'y': data_tis.values})
            
            # Load model components
            model, best_params_dict, scaler = self.model_manager.load_model_components(pollutant)
            
            # Retrain model
            retrained_model, _, _ = self.model_manager.retrain_model(
                pollutant, df_supervised, best_params_dict, model, scaler, 
                best_params_dict["lags"]
            )
            
            # Save retrained model
            save_path = self.model_manager.save_retrained_model(pollutant, retrained_model)
            retrained_models[pollutant] = save_path
        
        return retrained_models
    
    def generate_forecasts(self, df_final: pd.DataFrame, num_steps: int = 14) -> Dict[str, List]:
        """Generate forecasts for all pollutants."""
        print(f"Generating {num_steps}-day forecasts...")
        forecast_dict = {}
        scaler_dict = {}
        
        pollutants = df_final.columns.tolist()
        
        for pollutant in pollutants:
            print(f"Forecasting {pollutant}...")
            
            # Prepare data
            data_tis = df_final[pollutant].copy()
            df_supervised = pd.DataFrame({'y': data_tis.values})
            
            # Load model components
            model, best_params_dict, scaler = self.model_manager.load_model_components(pollutant)
            
            # Load retrained model
            retrained_model_path = os.path.join(self.model_manager.retrained_dir, f"{pollutant}_retrained.keras")
            retrained_model = load_model(retrained_model_path)
            
            # Prepare parameters
            best_params_array = [
                best_params_dict["units"],
                best_params_dict["dropout"],
                best_params_dict["learning_rate"],
                best_params_dict["epochs"]
            ]
            
            # Generate forecasts
            forecasts, _, scalers, _ = self.forecaster.sequential_forecast_with_retrain(
                pollutant, df_supervised, best_params_array, scaler, retrained_model,
                num_steps=num_steps, lag_offsets=best_params_dict["lags"]
            )
            
            forecast_dict[pollutant] = forecasts
            scaler_dict[pollutant] = scalers
        
        return forecast_dict, scaler_dict
    
    def run_complete_pipeline(self, num_forecast_days: int = 14) -> Tuple[pd.DataFrame, Dict[str, List]]:
        """Run the complete air quality forecasting pipeline."""
        print("Starting Air Quality Forecasting Pipeline...")
        
        # Step 1: Fetch and preprocess data
        df_final = self.fetch_and_preprocess_data()
        
        # Step 2: Retrain all models
        retrained_models = self.retrain_all_models(df_final)
        
        # Step 3: Generate forecasts
        forecasts, scalers = self.generate_forecasts(df_final, num_forecast_days)
        
        print("Pipeline completed successfully!")
        return df_final, forecasts, scalers
    
# --- Kelas AQI Converter ---
class AQICalculator:
    def __init__(self):
        self.breakpoints = {
            "pm25": [
                (0.0, 12.0, 0, 50),
                (12.0, 35.4, 51, 100),
                (35.4, 55.4, 101, 150),
                (55.4, 150.4, 151, 200),
                (150.4, 250.4, 201, 300),
                (250.4, 350.4, 301, 400),
                (350.4, 500.4, 401, 500)
            ],
            "pm10": [
                (0, 54, 0, 50),
                (54, 154, 51, 100),
                (154, 254, 101, 150),
                (254, 354, 151, 200),
                (354, 424, 201, 300),
                (424, 504, 301, 400),
                (504, 604, 401, 500)
            ],
            "co": [
                (0.0, 4.4, 0, 50),
                (4.4, 9.4, 51, 100),
                (9.4, 12.4, 101, 150),
                (12.4, 15.4, 151, 200),
                (15.4, 30.4, 201, 300),
                (30.4, 40.4, 301, 400),
                (40.4, 50.4, 401, 500)
            ],
            "so2": [
                (0, 35, 0, 50),
                (35, 75, 51, 100),
                (75, 185, 101, 150),
                (185, 304, 151, 200),
                (304, 604, 201, 300),
                (604, 804, 301, 400),
                (804, 1004, 401, 500)
            ],
            "no2": [
                (0, 53, 0, 50),
                (53, 100, 51, 100),
                (100, 360, 101, 150),
                (360, 649, 151, 200),
                (649, 1249, 201, 300),
                (1249, 1649, 301, 400),
                (1649, 2049, 401, 500)
            ],
            "o3": [
                (0.0, 0.054, 0, 50),
                (0.054, 0.070, 51, 100),
                (0.070, 0.085, 101, 150),
                (0.085, 0.105, 151, 200),
                (0.105, 0.200, 201, 300)
            ]
        }
    
    def calculate_aqi(self, pollutant: str, concentration: float) -> float:
        """Konversi konsentrasi polutan ke nilai AQI (dengan fallback aman)."""
        if pollutant not in self.breakpoints or concentration is None or np.isnan(concentration):
            return 0.0

        # 🔹 Konversi satuan O3 dari µg/m³ ke ppm sebelum perhitungan
        if pollutant == "o3":
            concentration = concentration / 2000.0

        # 🔹 Cari rentang breakpoint
        for c_low, c_high, i_low, i_high in self.breakpoints[pollutant]:
            if c_low <= concentration <= c_high:
                aqi = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
                return round(aqi, 2)

        # 🔹 Jika di bawah range terendah → kembalikan AQI minimum
        if concentration < self.breakpoints[pollutant][0][0]:
            return round(self.breakpoints[pollutant][0][2], 2)  # i_low pertama

        # 🔹 Jika di atas range tertinggi → kembalikan AQI maksimum (misal 500)
        if concentration > self.breakpoints[pollutant][-1][1]:
            return round(self.breakpoints[pollutant][-1][3], 2)  # i_high terakhir

        return 0.0

def save_forecast_to_db(forecasts_aqi):
    """
    Simpan hasil forecast (nilai AQI & konsentrasi) ke MySQL.
    Parameter: forecasts_aqi → dict hasil perhitungan dari pipeline
    """
    conn = create_connection()
    if not conn:
        print("Tidak bisa menyimpan data — koneksi MySQL gagal.")
        return
    
    cursor = conn.cursor()

    insert_query = """
        INSERT INTO forecast_aqi (pollutant, predicted_concentration, nilai_aqi)
        VALUES (%s, %s, %s)
    """

    try:
        for pollutant, values in forecasts_aqi.items():
            for row in values:
                cursor.execute(insert_query, (
                    pollutant,
                    row["predicted_concentration"],
                    row["nilai aqi"]
                ))
        conn.commit()
        print("✅ Data AQI berhasil disimpan ke database.")
    except Exception as e:
        print("❌ Gagal menyimpan ke database:", e)
    finally:
        cursor.close()
        conn.close()
