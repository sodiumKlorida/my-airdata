from controllers.bmkg_controller import BmkgController
from controllers.aqicn_controller import AqicnController

class TipsController:
    @staticmethod
    def fetch_Tips():
        tips = []

        # CUACA HUJAN
        if (BmkgController.tips_weather_hujan() is not None):
            tips.append(BmkgController.tips_weather_hujan())
        
        # SUHU UDARA PANAS ATAU DINGIN
        if (BmkgController.tips_weather_suhu() is not None):
            tips.append(BmkgController.tips_weather_suhu())
        
        # KUALITAS UDARA
        if (AqicnController.tips_air() is not None):
            tips.append(AqicnController.tips_air())
        
        tips_result = " ".join(tips)
        
        return tips_result 