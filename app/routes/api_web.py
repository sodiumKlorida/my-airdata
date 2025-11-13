from flask import Blueprint, render_template, jsonify
from controllers.aqicn_controller import AqicnController 
from controllers.bmkg_controller import BmkgController
from controllers.waqi_controller import WaqiController
from controllers.tips_controller import TipsController
from DB.config import get_forecast_data
from model_ds.function.air_quality_forecasting import AirQualityPipeline, AQICalculator, save_forecast_to_db
import traceback
import os

api_bp_web = Blueprint("try",__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "model_ds", "Model")

# =============================================
# WAQI FUNCTION
# =============================================
@api_bp_web.route('/waqi', methods=['GET'])
def get_waqi():
    try:
        waqi_data = WaqiController.fetch_WAQI_default()
        return jsonify({
            "success": True,
            "message": "Berhasil mendapatkan data WAQI",
            "data": waqi_data
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
# =============================================
# PREDIKSI AQI FUNCTION
# =============================================
@api_bp_web.route("/waqi/prediksi", methods=["GET"])
def prediksi_aqi_endpoint():
    try:
        hasil = WaqiController.prediksi_aqi_web()
        return jsonify(hasil)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
# =============================================
# WEB ROUTE
# =============================================
@api_bp_web.route("/")
def show_web():
    try:
        polutant = AqicnController.fetch_current_air()
        current_weather = BmkgController.fetch_current_weather_web()
        forecast_weather = BmkgController.fetch_forecast_weather_web()
        hasil = WaqiController.prediksi_aqi_web()
        tips = TipsController.fetch_Tips()
        
        return render_template(
            "dummy.html", 
            aqi=polutant,
            forecast=forecast_weather,
            current=current_weather,
            prediksi_aqi=hasil["aqi"],
            tips=tips
        )
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
@api_bp_web.route("/aqi/DB", methods=["GET"])
def api_get_forecast():
    """Endpoint untuk ambil data forecast dalam format JSON"""
    data = get_forecast_data()
    if "error" in data:
        return jsonify({"success": False, "error": data["error"]}), 500
    return jsonify({
        "success": True,
        "message": "Data forecast berhasil diambil dari database",
        "data": data
    })
    
# =============================================
# CURRENT WEATHER FUNCTION
# =============================================
def current_weather():
    try:
        list_weather = BmkgController.fetch_BMKG_default()
        current_weather = list_weather["data"][0].get("cuaca", [])[0][0]

        current = {
            "curah_hujan": current_weather.get("tp"),
            "kelembapan": current_weather.get("hu"),
            "suhu": current_weather.get("t"),
            "waktu": current_weather.get("local_datetime", current_weather.get("datetime")),
            "weatherdesc": current_weather.get("weather_desc"),
            "kecepatan_angin": current_weather.get("ws"),
            "ikon": current_weather.get("image")
        }

        return current

    except Exception as e:
        traceback.print_exc()
        return f"success: {False} error {str(e)}", 500
    
# =============================================
# CURRENT AQI FUNCTION
# =============================================
def current_aqi():
    try:
        data = AqicnController.fetch_AQICN_default()
        iaqi = data.get("data", {}).get("iaqi", {})

        aqi = {
            "pm25": iaqi.get("pm25", {}).get("v", 0),
            "no2": iaqi.get("no2", {}).get("v", 0),
            "co": iaqi.get("co", {}).get("v", 0),
            "pm10": iaqi.get("pm10", {}).get("v", 0),
            "so2": iaqi.get("so2", {}).get("v", 0),
            "o3": iaqi.get("o3", {}).get("v", 0),
        }

        # kirim data ke template (index.html atau dashboard.html)
        return aqi

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp_web.route("/forecast/train", methods=["GET"])
def forecast_train():
    try:
        pipeline = AirQualityPipeline(model_dir=MODEL_DIR)
        aqi_calc = AQICalculator()

        # Jalankan pipeline lengkap
        df_final, forecasts, scalers = pipeline.run_complete_pipeline(num_forecast_days=7)

        # Konversi ke AQI
        forecasts_aqi = {}
        for pollutant, values in forecasts.items():
            forecasts_aqi[pollutant] = [
                {
                    "predicted_concentration": round(float(v), 2),
                    "nilai aqi": aqi_calc.calculate_aqi(pollutant, v)
                }
                for v in values
            ]
        
        save_forecast_to_db(forecasts_aqi)


        return jsonify({
            "success": True,
            "message": "Retraining selesai dan prediksi berhasil dikonversi ke AQI. db",
            "data": forecasts_aqi,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    
    