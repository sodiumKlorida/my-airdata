document.addEventListener("DOMContentLoaded", function () {
  const mobileCard = document.getElementById("mobile-card");
  if (!mobileCard) return;

  // Ambil data dari elemen desktop agar sinkron
  const suhu = document.querySelector("#desktop-suhu")?.dataset.value || "--";
  const angin = document.querySelector("#desktop-angin")?.dataset.value || "--";
  const aqi = document.querySelector("#desktop-aqi")?.dataset.value || "--";
  const weatherDesc = document.querySelector("#desktop-weather")?.dataset.desc || "Berawan";

  // Fungsi menentukan animasi cuaca
  function getWeatherIcon(desc) {
    const d = desc.toLowerCase();
    if (d.includes("cerah"))
      return `<lottie-player src="/static/animations/sun.json" background="transparent" speed="1" style="width:100px;height:100px;" loop autoplay></lottie-player>`;
    if (d.includes("berawan"))
      return `<lottie-player src="/static/animations/cloudy.json" background="transparent" speed="1" style="width:100px;height:100px;" loop autoplay></lottie-player>`;
    if (d.includes("hujan"))
      return `<lottie-player src="/static/animations/rain_clouds.json" background="transparent" speed="1" style="width:100px;height:100px;" loop autoplay></lottie-player>`;
    if (d.includes("petir"))
      return `<lottie-player src="/static/animations/storm.json" background="transparent" speed="1" style="width:100px;height:100px;" loop autoplay></lottie-player>`;
    return `<lottie-player src="/static/animations/cloudy.json" background="transparent" speed="1" style="width:100px;height:100px;" loop autoplay></lottie-player>`;
  }

  // Data kartu yang akan berganti
  const cards = [
    {
      title: "Suhu",
      icon: "🌡️",
      value: `${suhu}°C`
    },
    {
      title: "Angin",
      icon: "🍃",
      value: `${angin} Km/j`
    },
    // {
    //   title: "AQI",
    //   icon: "🍃",
    //   value: `${angin} Km/j`
    // },
    {
      title: weatherDesc,
      iconHTML: getWeatherIcon(weatherDesc),
      value: ""
    }
  ];

  let index = 0;

  function updateCard() {
    const card = cards[index];
    mobileCard.classList.add("opacity-0", "transition", "duration-500");

    setTimeout(() => {
      mobileCard.innerHTML = `
        <p class="text-lg font-semibold text-gray-700 dark:text-gray-200 mb-2">${card.title}</p>
        ${card.iconHTML || `<span class='text-5xl'>${card.icon}</span>`}
        <p class="text-4xl font-bold text-gray-900 dark:text-gray-100 mt-2">${card.value}</p>
      `;
      mobileCard.classList.remove("opacity-0");
    }, 300);

    index = (index + 1) % cards.length;
  }

  updateCard();
  setInterval(updateCard, 4000);
});
