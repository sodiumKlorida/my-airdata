import requests

class AqicnController:
    
    # ============================================= 
    # URL AQICN
    # ============================================= 
    @staticmethod
    def url_AQICN():
        return 'https://api.waqi.info/feed/A420154/?token=8e6fff766a818b4aeba7e9b74e1a293da3f7c4dd'
    
    # =============================================
    # FETCH MAIN DATA AQICN
    # =============================================
    @staticmethod
    def fetch_AQICN_default():
        url = AqicnController.url_AQICN()
        response = requests.get(url)  
        data = response.json()        
        return data
    
    # ============================================= 
    # AQI CURRENT
    # ============================================= 
    @staticmethod
    def fetch_current_air(): 
        data = AqicnController.fetch_AQICN_default()
        iaqi = data.get("data", {}).get("iaqi", {})

        return {
            "aqi": iaqi.get("pm25", {}).get("v"),
            "pm25": iaqi.get("pm25", {}).get("v"),
            "no2": iaqi.get("no2", {}).get("v"),
            "co": iaqi.get("co", {}).get("v"),
            "pm10": iaqi.get("pm10", {}).get("v"),
            "so2": iaqi.get("so2", {}).get("v"),
            "o3": iaqi.get("o3", {}).get("v"),
        }
    
    # ============================================= 
    # TIPS AIR QUALITY
    # ============================================= 
    @staticmethod
    def tips_air():
        data = AqicnController.fetch_AQICN_default()
        iaqi = data.get("data", {}).get("iaqi", {})
        pm25 = iaqi.get("pm25", {}).get("v")
        
        if pm25:
            if pm25 <= 50:
                return "Kualitas udara baik, tetap jaga kesehatan!"
            elif pm25 <= 100:
                return "Kualitas udara sedang, sebaiknya gunakan masker."
            elif pm25 <= 150:
                return "Kualitas udara tidak sehat untuk kelompok sensitif, gunakan masker."
            elif pm25 <= 200:
                return "Kualitas udara tidak sehat, hindari aktivitas luar ruangan."
            elif pm25 <= 300:
                return "Kualitas udara sangat tidak sehat, tetap di dalam ruangan."
            else:
                return "Bahaya! Kualitas udara sangat buruk, segera cari tempat aman."
            


    