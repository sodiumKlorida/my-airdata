import mysql.connector
from mysql.connector import Error

def create_connection():  
    """Koneksi ke MySQL lokal."""
    try:
        connection = mysql.connector.connect(
            host="",   # tanpa https://
            user="",
            password="",
            database="",
            port=3306
        )
        
        if connection.is_connected():
            print("✅ Koneksi ke MySQL berhasil")
        return connection
    except Error as e:
        print("❌ Gagal konek ke MySQL:", e)
        return None

def get_forecast_data():
    """Ambil 7 data terakhir untuk 6 polutan utama dan ubah ke format per-baris."""
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

        # Kelompokkan 7 data terakhir per polutan
        grouped = {}
        for row in rows:
            p = row["pollutant"].lower()
            if p not in grouped:
                grouped[p] = []
            if len(grouped[p]) < 7:
                grouped[p].append(row["nilai_aqi"])

        # Ubah jadi format per baris (index ke-0, ke-1, dst)
        result = []
        for i in range(7):
            entry = {}
            for pollutant in ["co", "no2", "o3", "pm10", "pm25", "so2"]:
                values = grouped.get(pollutant, [])
                if len(values) > i:
                    entry[pollutant] = values[i]
            result.append(entry)

        return {"aqi": result}

    except Error as e:
        print("❌ Gagal ambil data:", e)
        return {"error": str(e)}

    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
