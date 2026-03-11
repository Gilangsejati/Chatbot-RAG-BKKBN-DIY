# src/voice.py
import os
from .config import WHISPER_MODEL

try:
    import whisper
except Exception:
    whisper = None


class VoiceNote:
    def __init__(self, model_name: str = WHISPER_MODEL):
        """
        Inisialisasi model Whisper. Jika library whisper tidak tersedia,
        akan melempar RuntimeError saat digunakan.
        """
        if whisper is None:
            raise RuntimeError("whisper package tidak terinstall. Jalankan: pip install -U openai-whisper")
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_path: str) -> str:
        """
        Transkrip file audio menjadi teks.
        Pastikan ffmpeg terinstall dan audio_path valid.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file tidak ditemukan: {audio_path}")
        # whisper mengembalikan dict hasil, ambil key 'text'
        result = self.model.transcribe(audio_path)
        return result.get("text", "")


# Untuk test cepat jalankan: python -m src.voice
if __name__ == "__main__":
    # contoh test: letakkan file audio di data/test_audio.mp3
    test_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "test_audio.mp3")
    print("Test audio path:", test_file)
    try:
        vn = VoiceNote()
        text = vn.transcribe(test_file)
        print("Hasil transkripsi:\n", text)
    except Exception as e:
        print("Error:", e)
        print("\nCatatan:\n - Pastikan paket 'openai-whisper' terinstall (pip install -U openai-whisper)\n - Pastikan ffmpeg terinstall dan dapat dipanggil dari terminal (cek: ffmpeg -version)\n - Jika file audio tidak ada, ubah variabel test_file di bagian __main__")
