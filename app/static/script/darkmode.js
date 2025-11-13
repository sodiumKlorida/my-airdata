document.addEventListener("DOMContentLoaded", function () {
  const toggleButton = document.getElementById("toggle-theme");
  const switchIcon = document.getElementById("switch-icon");
  const iconSun = document.getElementById("icon-sun");
  const iconMoon = document.getElementById("icon-moon");

  // Cek tema yang tersimpan
  if (localStorage.theme === "dark" || 
     (!("theme" in localStorage) && window.matchMedia("(prefers-color-scheme: dark)").matches)) {
    document.documentElement.classList.add("dark");
    iconSun.classList.add("hidden");
    iconMoon.classList.remove("hidden");
    switchIcon.classList.add("translate-x-6");
  }

  // Event klik tombol
  toggleButton.addEventListener("click", () => {
    const html = document.documentElement;
    const isDark = html.classList.contains("dark");

    html.classList.toggle("dark");
    localStorage.theme = isDark ? "light" : "dark";

    // Animasi dan ikon
    switchIcon.classList.toggle("translate-x-6");
    iconSun.classList.toggle("hidden");
    iconMoon.classList.toggle("hidden");
  });
});
