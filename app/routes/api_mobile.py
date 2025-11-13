from flask import Blueprint, jsonify
from controllers.aqicn_controller import AqicnController
from controllers.bmkg_controller import BmkgController
from controllers.waqi_controller import WaqiController
from controllers.tips_controller import TipsController

api_bp_mobile = Blueprint('api', __name__, url_prefix='/api')

# =============================================
# COMBINED WEATHER AND AIR QUALITY + TIPS
# =============================================
@api_bp_mobile.route('/air/current', methods=['GET'])
def get_current():
    try:
        weather = BmkgController.fetch_current_weather()
        air = AqicnController.fetch_current_air()
        tips = TipsController.fetch_Tips()  
        return jsonify({
                "success": True, 
                "message": "Berhasil mendapatkan data", 
                "data": {
                    "weather": weather,
                    "air": air,
                    "tips":tips
                }
            }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
# =============================================
# COMBINED WEATHER AND AQI FORECAST
# =============================================   
@api_bp_mobile.route('/air/forecast', methods=['GET'])
def get_forecast():
    try:
        weather = BmkgController.fetch_forecast_weather()
        aqi = WaqiController.prediksi_aqi_mobile()
        return jsonify(
            {
                "success": True,
                "message": "Berhasil mendapatkan data",
                "data": {
                    "air": {
                        "weather": weather,
                        "aqi": aqi
                    }
                }
            }
            
            ), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

