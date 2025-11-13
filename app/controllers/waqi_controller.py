from flask import Blueprint, jsonify
import requests, json
from datetime import datetime, timedelta
from DB.config import create_connection
from mysql.connector import Error


api_bp_web = Blueprint("api_bp_web", __name__)

class WaqiController:

    @staticmethod
    def get_breakpoints():
        return {
            "pm25": [(0.0, 12.0, 0, 50),
                     (12.0, 35.4, 51, 100),
                     (35.4, 55.4, 101, 150),
                     (55.4, 150.4, 151, 200),
                     (150.4, 250.4, 201, 300),
                     (250.4, 350.4, 301, 400),
                     (350.4, 500.4, 401, 500)],
            "pm10": [(0, 54, 0, 50),
                     (54, 154, 51, 100),
                     (154, 254, 101, 150),
                     (254, 354, 151, 200),
                     (354, 424, 201, 300),
                     (424, 504, 301, 400),
                     (504, 604, 401, 500)],
            "co": [(0.0, 4.4, 0, 50),
                   (4.4, 9.4, 51, 100),
                   (9.4, 12.4, 101, 150),
                   (12.4, 15.4, 151, 200),
                   (15.4, 30.4, 201, 300),
                   (30.4, 40.4, 301, 400),
                   (40.4, 50.4, 401, 500)],
            "so2": [(0, 35, 0, 50),
                    (35, 75, 51, 100),
                    (75, 185, 101, 150),
                    (185, 304, 151, 200),
                    (304, 604, 201, 300),
                    (604, 804, 301, 400),
                    (804, 1004, 401, 500)],
            "no2": [(0, 53, 0, 50),
                    (53, 100, 51, 100),
                    (100, 360, 101, 150),
                    (360, 649, 151, 200),
                    (649, 1249, 201, 300),
                    (1249, 1649, 301, 400),
                    (1649, 2049, 401, 500)],
            "o3": [(0.0, 0.054, 0, 50),
                   (0.054, 0.070, 51, 100),
                   (0.070, 0.085, 101, 150),
                   (0.085, 0.105, 151, 200),
                   (0.105, 0.200, 201, 300)]
        }

    @staticmethod
    def hitung_aqi(pollutant, concentration):
        bp = WaqiController.get_breakpoints().get(pollutant)
        if not bp or concentration is None:
            return 0
        for c_low, c_high, i_low, i_high in bp:
            if c_low <= concentration <= c_high:
                return round(((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low)
        return 0

    @staticmethod
    def fetch_WAQI_default():
        url = "https://airnet.waqi.info/airnet/sse/historic/daily/420154"
        try:
            r = requests.get(url, stream=True, timeout=15)
            if r.status_code != 200:
                return []
        except Exception:
            return []

        data_list = []
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                try:
                    json_part = line.replace("data: ", "")
                    parsed = json.loads(json_part)
                    data_list.append(parsed)
                except json.JSONDecodeError:
                    continue
        return data_list
    
    @staticmethod
    def prediksi_aqi_mobile():
        conn = create_connection()
        if conn is None:
            return {"error": "Koneksi gagal"}

        try:
            cursor = conn.cursor(dictionary=True)
            query = """
                SELECT pollutant, nilai_aqi
                FROM forecast_aqi
                WHERE pollutant IN ('pm25', 'pm10', 'no2', 'so2', 'co', 'o3')
                ORDER BY id DESC
            """
            cursor.execute(query)
            rows = cursor.fetchall()

            # Kelompokkan data per polutan
            grouped = {p: [] for p in ["co", "no2", "o3", "pm10", "pm25", "so2"]}
            for row in rows:
                p = row["pollutant"].lower()
                if len(grouped[p]) < 7:
                    grouped[p].append(row["nilai_aqi"])

            # 🔹 Mulai tanggal dari BESOK
            start_date = datetime.now() + timedelta(days=1)

            # Gabungkan jadi list per baris (tiap baris berisi semua polutan)
            result = []
            for i in range(7):
                tanggal_prediksi = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")

                result.append({
                    "aqi": grouped["pm25"][i] if i < len(grouped["pm25"]) else None,
                    "co": grouped["co"][i] if i < len(grouped["co"]) else None,
                    "no2": grouped["no2"][i] if i < len(grouped["no2"]) else None,
                    "o3": grouped["o3"][i] if i < len(grouped["o3"]) else None,
                    "pm10": grouped["pm10"][i] if i < len(grouped["pm10"]) else None,
                    "pm25": grouped["pm25"][i] if i < len(grouped["pm25"]) else None,
                    "so2": grouped["so2"][i] if i < len(grouped["so2"]) else None,
                    "tanggal": tanggal_prediksi
                })

            return result

        except Error as e:
            print("❌ Gagal ambil data:", e)
            return {"error": str(e)}

        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()


    @staticmethod
    def prediksi_aqi_web():
        
        conn = create_connection()
        if conn is None:
            return {"error": "Koneksi gagal"}

        try:
            cursor = conn.cursor(dictionary=True)
            query = """
                SELECT pollutant, nilai_aqi
                FROM forecast_aqi
                WHERE pollutant IN ('pm25', 'pm10', 'no2', 'so2', 'co', 'o3')
                ORDER BY id DESC
            """
            cursor.execute(query)
            rows = cursor.fetchall()

            # Kelompokkan data per polutan
            grouped = {p: [] for p in ["co", "no2", "o3", "pm10", "pm25", "so2"]}
            for row in rows:
                p = row["pollutant"].lower()
                if len(grouped[p]) < 7:
                    grouped[p].append(row["nilai_aqi"])

            # 🔹 Mulai tanggal dari BESOK
            start_date = datetime.now() + timedelta(days=1)

            # Gabungkan jadi list per baris (tiap baris berisi semua polutan)
            hasil_prediksi = []
            for i in range(7):
                tanggal_prediksi = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")

                hasil_prediksi.append({
                    "aqi": grouped["pm25"][i] if i < len(grouped["pm25"]) else None,
                    "co": grouped["co"][i] if i < len(grouped["co"]) else None,
                    "no2": grouped["no2"][i] if i < len(grouped["no2"]) else None,
                    "o3": grouped["o3"][i] if i < len(grouped["o3"]) else None,
                    "pm10": grouped["pm10"][i] if i < len(grouped["pm10"]) else None,
                    "pm25": grouped["pm25"][i] if i < len(grouped["pm25"]) else None,
                    "so2": grouped["so2"][i] if i < len(grouped["so2"]) else None,
                    "tanggal": tanggal_prediksi
                })

            return {"aqi": hasil_prediksi}

        except Error as e:
            print("❌ Gagal ambil data:", e)
            return {"error": str(e)}

        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()
    
    