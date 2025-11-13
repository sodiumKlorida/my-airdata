import traceback
from flask import Blueprint, jsonify, render_template 
from model_ds.function.air_quality_forecasting import AirQualityPipeline, AQICalculator, save_forecast_to_db
# from DB.config import get_forecast_data, get_forecast_data_new
import os
import requests

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "model_ds", "Model")

api_bp = Blueprint("aqi_model", __name__)

@api_bp.route("/forecast/train", methods=["GET"])
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
        
# @api_bp.route("/api/forecast/DB", methods=["GET"])
# def api_get_forecast():
#     """Endpoint untuk ambil data forecast dalam format JSON"""
#     data = get_forecast_data()

#     if "error" in data:
#         return jsonify({"success": False, "error": data["error"]}), 500

#     return jsonify({
#         "success": True,
#         "message": "Data forecast berhasil diambil dari database",
#         "data": data
#     })
    
# @api_bp.route("/api/forecast/DB/new", methods=["GET"])
# def api_get():
#     """Endpoint untuk ambil data forecast dalam format JSON"""
#     data = get_forecast_data_new()

#     if "error" in data:
#         return jsonify({"success": False, "error": data["error"]}), 500

#     return jsonify({
#         "success": True,
#         "message": "Data forecast berhasil diambil dari database",
#         "data": data
#     })

# @api_bp.route("/forecast/chart")
# def forecast_chart():
#     """Menampilkan halaman Chart Forecast AQI"""
#     # Tidak perlu fetch API di sini, data diambil via JS
#     return render_template("chart.html")

    
# # @api_bp.route("/api/forecast/pm25", methods=["GET"])
# # def pm25():
# #     """Endpoint untuk ambil data PM2.5 (7 hari terbaru)"""
# #     data = get_forecast_data()

# #     # 🔹 Jika ada error saat ambil data
# #     if "error" in data:
# #         return jsonify({"success": False, "error": data["error"]}), 500

# #     # 🔹 Filter hanya polutan PM2.5 (bisa jadi key-nya "pm2_5" atau "pm25")
# #     pm25_data = [item for item in data if item.get("pollutant", "").lower() in ("pm2.5", "pm25")]

# #     # 🔹 Urutkan berdasarkan tanggal (created_at)
# #     pm25_data.sort(key=lambda x: x.get("created_at", ""), reverse=True)

# #     # 🔹 Ambil hanya 7 data terbaru
# #     pm25_data = pm25_data[:7]

# #     # 🔹 Urutkan ulang agar tanggalnya naik (lama → baru)
# #     pm25_data = sorted(pm25_data, key=lambda x: x.get("created_at", ""))

# #     return jsonify({
# #         "success": True,
# #         "message": "Data PM2.5 (7 hari terbaru) berhasil diambil",
# #         "data": pm25_data
# #     })


# # @api_bp.route("/forecast/chart", methods=["GET"])
# # def forecast_chart_page():
# #     return render_template("chart.html")

# # @api_bp.route("/data", methods=["GET"])
# # def forecast():
# #     return render_template("forecast.html")



