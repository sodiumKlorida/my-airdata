document.addEventListener("DOMContentLoaded", async () => {
    const ctx = document.getElementById("aqiChart").getContext("2d");
    let aqiChart; // simpan instance Chart agar bisa diupdate tema

    try {
        const response = await fetch("/aqi/DB");
        const result = await response.json();

        if (!result.success) {
            alert("Gagal memuat data dari API");
            return;
        }

        const aqiData = result.data.aqi;

        // 🔹 Tanggal dimulai dari BESOK
        const startDate = new Date();
        startDate.setDate(startDate.getDate() + 1);

        // 🔹 Generate tanggal otomatis
        const labels = aqiData.map((_, i) => {
            const nextDate = new Date(startDate);
            nextDate.setDate(startDate.getDate() + i);
            return nextDate.toLocaleDateString("id-ID", {
                day: "2-digit",
                month: "short",
            });
        });

        // 🔹 Dataset untuk 6 polutan
        const datasets = [
            { label: "CO",   data: aqiData.map(d => d.co),   borderColor: "rgb(255, 99, 132)" },
            { label: "NO₂",  data: aqiData.map(d => d.no2),  borderColor: "rgb(54, 162, 235)" },
            { label: "O₃",   data: aqiData.map(d => d.o3),   borderColor: "rgb(255, 206, 86)" },
            { label: "PM10", data: aqiData.map(d => d.pm10), borderColor: "rgb(75, 192, 192)" },
            { label: "PM2.5",data: aqiData.map(d => d.pm25), borderColor: "rgb(153, 102, 255)" },
            { label: "SO₂",  data: aqiData.map(d => d.so2),  borderColor: "rgb(255, 159, 64)" }
        ];

        // 🔹 Fungsi buat tema chart
        const getChartColors = () => {
            const isDark = document.documentElement.classList.contains("dark");
            return {
                text: isDark ? "#e5e7eb" : "#374151",   // teks
                grid: isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)",
                title: isDark ? "#f3f4f6" : "#111827",
                background: isDark ? "#1f2937" : "#f3f4f6" // warna area chart
            };
        };

        const colors = getChartColors();

        // 🔹 Buat chart awal
        aqiChart = new Chart(ctx, {
            type: "line",
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                backgroundColor: colors.background,
                scales: {
                    x: {
                        ticks: { color: colors.text },
                        grid: { color: colors.grid }
                    },
                    y: {
                        beginAtZero: true,
                        ticks: { color: colors.text },
                        grid: { color: colors.grid },
                        title: {
                            display: true,
                            text: "Konsentrasi Polutan (µg/m³)",
                            color: colors.text
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: "bottom",
                        labels: { color: colors.text }
                    },
                    title: {
                        display: true,
                        text: "Perbandingan 6 Polutan Berdasarkan Prediksi Tanggal",
                        color: colors.title
                    }
                },
                elements: {
                    line: { tension: 0.3 },
                    point: { radius: 4 }
                }
            }
        });

        // 🔹 Update tema chart saat toggle dark mode ditekan
        const themeToggle = document.getElementById("toggle-theme");
        if (themeToggle) {
            themeToggle.addEventListener("click", () => {
                setTimeout(() => {
                    const newColors = getChartColors();
                    const { scales, plugins } = aqiChart.options;

                    // update warna chart tanpa reset data
                    scales.x.ticks.color = newColors.text;
                    scales.x.grid.color = newColors.grid;
                    scales.y.ticks.color = newColors.text;
                    scales.y.grid.color = newColors.grid;
                    scales.y.title.color = newColors.text;
                    plugins.title.color = newColors.title;
                    plugins.legend.labels.color = newColors.text;

                    aqiChart.options.backgroundColor = newColors.background;
                    aqiChart.update();
                }, 300); // beri jeda sedikit agar class .dark sempat berubah
            });
        }

    } catch (error) {
        console.error("Error:", error);
        alert("Terjadi kesalahan saat mengambil data.");
    }
});
