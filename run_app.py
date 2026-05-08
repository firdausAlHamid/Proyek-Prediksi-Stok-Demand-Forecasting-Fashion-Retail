"""
run_app.py - Deployment script untuk menjalankan Streamlit + optional ngrok tunnel
"""
import subprocess, threading, time, socket

def is_port_in_use(port: int) -> bool:
    """Cek apakah port sudah dipakai (Streamlit sudah jalan)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0

def run_streamlit():
    subprocess.Popen([
        "streamlit", "run", "app.py",
        "--server.port=8501",
        "--server.headless=true",
        "--server.address=0.0.0.0"
    ])

if __name__ == "__main__":
    if is_port_in_use(8501):
        print("[INFO] Streamlit sudah berjalan di port 8501, skip start ulang.")
    else:
        print("[INFO] Menjalankan Streamlit di port 8501 ...")
        threading.Thread(target=run_streamlit, daemon=True).start()
        time.sleep(5)

    try:
        from pyngrok import ngrok, conf
        # Set auth token ngrok
        conf.get_default().auth_token = "2wwLDRxdsjQ7vXKu3QR5Nj76sIK_2UjrRZBzSBz2yVNJEojAu"
        # Tutup tunnel lama jika ada
        for tunnel in ngrok.get_tunnels():
            ngrok.disconnect(tunnel.public_url)
        url = ngrok.connect(8501)
        print(f"\n[OK] Dashboard ONLINE! Buka URL ini:\n     {url.public_url}\n")
    except ImportError:
        print("[INFO] pyngrok tidak terinstall. Dashboard lokal: http://localhost:8501")
    except Exception as e:
        print(f"[WARN] ngrok error: {e}")
        print("[INFO] Dashboard lokal tetap bisa diakses di: http://localhost:8501")

    # Jaga proses tetap hidup
    print("[INFO] Tekan Ctrl+C untuk menghentikan.")
    while True:
        time.sleep(60)
