document.addEventListener("DOMContentLoaded", function () {
  const container = document.getElementById("weather-animation");
  const desc = document.getElementById("weather-condition").textContent.trim().toLowerCase();

  // Tentukan animasi berdasarkan kondisi cuaca
  let animasi = "cloudy.json"; // default

  if (desc.includes("cerah")) {
    animasi = "sun.json";
  } else if (desc.includes("berawan")) {
    animasi = "cloudy.json";
  } else if (desc.includes("hujan ringan")) {
    animasi = "rain_clouds.json";
  } else if (desc.includes("hujan")) {
    animasi = "rain_clouds.json";
  } else if (desc.includes("hujan petir") || desc.includes("petir")) {
    animasi = "storm.json";
  } else if (desc.includes("kabut")) {
    animasi = "foggy.json";
  }

  // Render animasi
  container.innerHTML = `
    <lottie-player
      src="/static/animations/${animasi}"
      background="transparent"
      speed="1"
      style="width: 100%; height: 100%;"
      loop
      autoplay>
    </lottie-player>
  `;
});
