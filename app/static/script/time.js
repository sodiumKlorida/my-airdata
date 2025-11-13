// ======================
// Update Tanggal & Waktu
// ======================
function updateDateTime() {
  const tanggalEl = document.getElementById('tanggal');
  const waktuEl = document.getElementById('waktu');

  const now = new Date();

  // Format tanggal: Jumat, 17 Oktober 2025
  const tanggal = now.toLocaleDateString('id-ID', {
    weekday: 'long',
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  });

  // Format waktu: 13:45:12
  const waktu = now.toLocaleTimeString('id-ID', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });

  tanggalEl.textContent = tanggal;
  waktuEl.textContent = waktu;
}

// Jalankan setiap detik
setInterval(updateDateTime, 1000);

// Jalankan pertama kali saat halaman dimuat
updateDateTime();
