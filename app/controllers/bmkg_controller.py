import requests
from datetime import datetime, timedelta
from flask import jsonify

class BmkgController:
    
    # ============================================= 
    # URL BMKG
    # =============================================
    @staticmethod    
    def url_BMKG():
        return 'https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4=35.78.14.1007'
    
     # ============================================= 
    # FETCH MAIN DATA BMKG
    # =============================================
    @staticmethod
    def fetch_BMKG_default():
        url = BmkgController.url_BMKG()
        response = requests.get(url)  
        data = response.json()        
        return data
    
    # ============================================= 
    # FETCH WEATHER CURRENT
    # =============================================   
    @staticmethod
    def fetch_current_weather():
       list_weather = BmkgController.fetch_BMKG_default() 
       current_weather = list_weather["data"][0].get("cuaca", [])[0][0]
       return {    
            "curah_hujan": current_weather.get("tp"), 
            "kelembapan": current_weather.get("hu"),
            "suhu": current_weather.get("t"),
            "waktu": current_weather.get("local_datetime", current_weather.get("datetime")),
            "weatherdesc": current_weather.get("weather_desc"),
            "kecepatan_angin": current_weather.get("ws"), 
       }
       
    @staticmethod
    def fetch_current_weather_web():
       list_weather = BmkgController.fetch_BMKG_default() 
       current_weather = list_weather["data"][0].get("cuaca", [])[0][0]
       return {    
            "curah_hujan": current_weather.get("tp"), 
            "kelembapan": current_weather.get("hu"),
            "suhu": current_weather.get("t"),
            "waktu": current_weather.get("local_datetime", current_weather.get("datetime")),
            "weatherdesc": current_weather.get("weather_desc"),
            "kecepatan_angin": current_weather.get("ws"), 
            "ikon": current_weather.get("image"), 
       }
    
    # ============================================= 
    # FORECAST WEATHER   
    # ============================================= 
    @staticmethod
    def fetch_forecast_weather():
        """Ambil daftar cuaca dari data BMKG untuk hari ini dan besok"""
        
        cuaca_list = []
        data_list = BmkgController.fetch_BMKG_default()  # ambil data default BMKG

        if data_list:
            cuaca_raw = data_list["data"][0].get("cuaca", [])

            # Hitung tanggal hari ini dan besok
            today = datetime.now().date()
            tomorrow = today + timedelta(days=1)

            for periode in cuaca_raw:
                for entry in periode:
                    # Ambil waktu entry sebagai datetime object
                    waktu_str = entry.get("local_datetime", entry.get("datetime"))
                    try:
                        waktu_dt = datetime.fromisoformat(waktu_str)
                    except Exception:
                        continue  # skip jika format waktu salah

                    # Ambil hanya hari ini atau besok
                    if waktu_dt.date() == today or waktu_dt.date() == tomorrow:
                        cuaca_list.append({
                            "kelembapan": entry.get("hu"),
                            "suhu": entry.get("t"),
                            "waktu": waktu_str,
                            "weatherdesc": entry.get("weather_desc"),
                        })
                        
        return cuaca_list 
    
    @staticmethod
    def fetch_forecast_weather_web():
        """Ambil daftar cuaca dari data BMKG untuk hari ini dan besok"""
        
        cuaca_list = []
        data_list = BmkgController.fetch_BMKG_default()  # ambil data default BMKG

        if data_list:
            cuaca_raw = data_list["data"][0].get("cuaca", [])

            # Hitung tanggal hari ini dan besok
            today = datetime.now().date()
            tomorrow = today + timedelta(days=1)

            for periode in cuaca_raw:
                for entry in periode:
                    # Ambil waktu entry sebagai datetime object
                    waktu_str = entry.get("local_datetime", entry.get("datetime"))
                    try:
                        waktu_dt = datetime.fromisoformat(waktu_str)
                    except Exception:
                        continue  # skip jika format waktu salah

                    # Ambil hanya hari ini atau besok
                    if waktu_dt.date() == today or waktu_dt.date() == tomorrow:
                        cuaca_list.append({
                            "kelembapan": entry.get("hu"),
                            "suhu": entry.get("t"),
                            "waktu": waktu_str,
                            "weatherdesc": entry.get("weather_desc"),
                            "ikon": entry.get("image"),
                        })
                        
        return cuaca_list 
    
    # ============================================= 
    # TIPS WEATHER RAIN
    # =============================================
    @staticmethod
    def tips_weather_hujan():
        data_weather = BmkgController.fetch_BMKG_default()
        weather = data_weather["data"][0].get("cuaca", [])[0][0].get("weather_desc")

        if weather:
            if weather == "Hujan":
                return "Cuaca sedang hujan, jangan lupa bawa payung!"
            if weather == "Hujan Ringan":
                return "Cuaca sedang hujan, jangan lupa bawa payung!"
            if weather == "Hujan Petir":
                return "Cuaca hujan petir, sebaiknya tetap di dalam ruangan!"

    # ============================================= 
    # TIPS WEATHER TEMPERATURE  
    # =============================================    
    def tips_weather_suhu():
        data_weather = BmkgController.fetch_BMKG_default()
        temperature = data_weather["data"][0].get("cuaca", [])[0][0].get("t")
        if temperature:
            if temperature >= 31:
                return "Cuaca panas, pastikan untuk tetap terhidrasi!"
            elif temperature <= 25:
                return "Cuaca dingin, kenakan pakaian hangat!"

    
        
