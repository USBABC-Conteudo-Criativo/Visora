import os
import sys  # Adicione esta linha para importar o módulo sys
import cv2
import concurrent.futures
from vosk import Model, KaldiRecognizer
import numpy as np
import threading
from datetime import datetime
import json
from pathlib import Path
import wave
import re
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QFrame, QMenuBar, QMenu,
    QStatusBar, QFileDialog, QProgressBar, QToolBar,
    QSizePolicy, QScrollArea, QDoubleSpinBox, QSpinBox, QMessageBox, QListWidget, QListWidgetItem, QInputDialog, QSlider, QGraphicsDropShadowEffect
)
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve, QUrl
from PyQt6.QtGui import QAction, QIcon, QImage, QPixmap, QColor
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from pydub import AudioSegment
from pydub.silence import split_on_silence
import requests
import subprocess

class VideoProcessor:
    def __init__(self):
        self.report = {
            "video_original": "",
            "tempo_cortes": 45,
            "total_segmentos": 0,
            "segmentos": [],
            "transcricoes": [],
            "highlights": [],
            "video_compilado": "",
            "logs": []
        }
        self.vosk_model = None
        self.word_corrections = {
            "entao": "então", "entaum": "então", "eh": "é",
            "tb": "também", "tambem": "também", "voce": "você",
            "vc": "você", "tá": "está", "ta": "está",
            "pra": "para", "q": "que", "pq": "porque",
            "n": "não", "nao": "não", "ne": "né",
            "vamo": "vamos", "agnt": "a gente", "cmg": "comigo",
            "ñ": "não", "p": "para", "num": "não",
            "ai": "aí", "hj": "hoje", "qdo": "quando",
            "kd": "cadê", "tbm": "também", "pq": "porque",
            "oq": "o que", "mto": "muito", "td": "tudo",
            "nd": "nada", "qnt": "quanto", "qndo": "quando",
            "msm": "mesmo", "vlw": "valeu", "vc": "você",
            "blz": "beleza", "flw": "falou", "mt": "muito",
            "qm": "quem", "tdo": "tudo", "sla": "sei lá",
            "bom": "bom", "meu": "meu", "minha": "minha",
            "aqui": "aqui", "agora": "agora", "então": "então",
            "acho": "acho", "isso": "isso", "esse": "esse",
            "esta": "esta", "este": "este", "isso": "isso",
            "aquilo": "aquilo", "aquele": "aquele",
            "pro": "para o", "pros": "para os", "pras": "para as",
            "prum": "para um", "numa": "em uma", "num": "em um",
        }

    def initialize_vosk(self):
        if self.vosk_model is None:
            model_path = "model"
            if os.path.exists(model_path):
                print("Carregando modelo Vosk...")
                self.vosk_model = Model(model_path=model_path)
            else:
                print("Modelo Vosk não encontrado. Baixando...")
                # Apenas indicando que o modelo precisa ser baixado manualmente
                Path("model").mkdir(exist_ok=True)
                self.report["logs"].append("Modelo Vosk não encontrado. Por favor, baixe o modelo pt-BR em https://alphacephei.com/vosk/models e extraia para a pasta 'model'")
                return False
        return True

    def save_log(self, output_folder):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file = Path(output_folder) / "process_log.txt"
        with open(log_file, "w", encoding="utf-8") as log:
            log.write(f"Processamento Iniciado: {timestamp}\n")
            for entry in self.report["logs"]:
                log.write(f"{entry}\n")
        self.report["logs"].append(f"Log salvo em {log_file}")
        return log_file

    def extract_audio(self, video_path, output_folder):
        try:
            Path(output_folder).mkdir(parents=True, exist_ok=True)
            audio_path = Path(output_folder) / "audio.wav"
            command = [
                'ffmpeg',
                '-i', str(video_path),
                '-acodec', 'pcm_s16le',
                '-ac', '1',
                '-ar', '16000',
                '-y',
                str(audio_path)
            ]
            subprocess.run(command, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            self.report["logs"].append(f"Áudio extraído com sucesso: {audio_path}")
            return str(audio_path)
        except subprocess.CalledProcessError as e:
            self.report["logs"].append(f"Erro ao extrair áudio: {str(e)}")
            return None

    def improve_transcript(self, text):
        if not text:
            return ""
        words = text.split()
        corrected_words = []
        for word in words:
            word_lower = word.lower()
            if word_lower in self.word_corrections:
                if word[0].isupper():
                    corrected_words.append(self.word_corrections[word_lower].capitalize())
                else:
                    corrected_words.append(self.word_corrections[word_lower])
            else:
                corrected_words.append(word)
        improved_text = " ".join(corrected_words)
        improved_text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), improved_text)
        if improved_text and improved_text[0].isalpha():
            improved_text = improved_text[0].upper() + improved_text[1:]
        sentences = re.split(r'([.!?])', improved_text)
        result = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and sentences[i+1] in ['.', '!', '?']:
                result.append(sentences[i] + sentences[i+1])
                i += 2
            else:
                result.append(sentences[i])
                i += 1
        final_text = " ".join(result).strip()
        final_text = re.sub(r'\s+', ' ', final_text)
        final_text = re.sub(r'\s([,.!?:;])', r'\1', final_text)
        return final_text

    def generate_transcript(self, audio_path):
        try:
            if not self.initialize_vosk():
                return {"text": "Modelo de reconhecimento não encontrado. Baixe o modelo pt-BR.", "words": []}

            recognizer = KaldiRecognizer(self.vosk_model, 16000)
            recognizer.SetWords(True)
            with wave.open(audio_path, "rb") as wf:
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
                    raise ValueError("O áudio deve ser mono, 16-bit PCM e 16kHz.")
                transcript = ""
                word_results = []
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        transcript += result.get("text", "") + " "
                        if "result" in result:
                            word_results.extend(result["result"])
                final_result = json.loads(recognizer.FinalResult())
                transcript += final_result.get("text", "")
                if "result" in final_result:
                    word_results.extend(final_result["result"])
            improved_transcript = self.improve_transcript(transcript.strip())
            self.report["logs"].append(f"Transcrição gerada: {len(improved_transcript.split())} palavras")
            return {"text": improved_transcript, "words": word_results}
        except Exception as e:
            self.report["logs"].append(f"Erro na transcrição: {str(e)}")
            return {"text": "Erro ao gerar transcrição.", "words": []}

    def segment_video(self, filename, output_folder, segment_time):
        try:
            Path(output_folder).mkdir(parents=True, exist_ok=True)
            output_pattern = str(Path(output_folder) / "segment%d.mp4")
            command = [
                'ffmpeg',
                '-i', filename,
                '-f', 'segment',
                '-segment_time', str(segment_time),
                '-c', 'copy',
                '-reset_timestamps', '1',
                '-y',
                output_pattern
            ]
            subprocess.run(command, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            segment_files = sorted([f for f in os.listdir(output_folder) if f.startswith("segment") and f.endswith(".mp4")])
            self.report["segmentos"] = [str(Path(output_folder) / f) for f in segment_files]
            self.report["total_segmentos"] = len(segment_files)
            self.report["logs"].append(f"Vídeo segmentado em {len(segment_files)} partes")
        except subprocess.CalledProcessError as e:
            self.report["logs"].append(f"Erro ao segmentar vídeo: {str(e)}")

    def _format_srt_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_remainder = seconds % 60
        milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(seconds_remainder):02d},{milliseconds:03d}"

    def generate_srt(self, transcript_data, output_path, filename):
        try:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            srt_file = Path(output_path) / f"{filename}.srt"
            if not transcript_data.get("words"):
                with open(srt_file, "w", encoding="utf-8") as f:
                    start_time = "00:00:00,000"
                    word_count = len(transcript_data["text"].split())
                    duration_seconds = max(word_count * 0.5, 5)  # Mais tempo para menos palavras
                    end_time = self._format_srt_time(duration_seconds)
                    f.write("1\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{transcript_data['text'].strip()}\n\n")
                self.report["logs"].append(f"Legendas estimadas geradas em {srt_file}")
                return str(srt_file)
            phrases = []
            current_phrase = []
            current_start = transcript_data["words"][0]["start"]
            for word in transcript_data["words"]:
                current_phrase.append(word)
                if len(current_phrase) >= 6 and (
                    word["word"].endswith('.') or
                    word["word"].endswith('!') or
                    word["word"].endswith('?') or
                    word["word"].endswith(',')
                ):
                    phrase_text = " ".join(w["word"] for w in current_phrase)
                    improved_phrase = self.improve_transcript(phrase_text)
                    phrases.append({
                        "start": current_start,
                        "end": word["end"],
                        "text": improved_phrase
                    })
                    if len(transcript_data["words"]) > transcript_data["words"].index(word) + 1:
                        current_start = transcript_data["words"][transcript_data["words"].index(word) + 1]["start"]
                    current_phrase = []
                elif len(current_phrase) >= 8:
                    phrase_text = " ".join(w["word"] for w in current_phrase)
                    improved_phrase = self.improve_transcript(phrase_text)
                    phrases.append({
                        "start": current_start,
                        "end": word["end"],
                        "text": improved_phrase
                    })
                    if len(transcript_data["words"]) > transcript_data["words"].index(word) + 1:
                        current_start = transcript_data["words"][transcript_data["words"].index(word) + 1]["start"]
                    current_phrase = []
            if current_phrase:
                phrase_text = " ".join(w["word"] for w in current_phrase)
                improved_phrase = self.improve_transcript(phrase_text)
                phrases.append({
                    "start": current_start,
                    "end": current_phrase[-1]["end"],
                    "text": improved_phrase
                })
            with open(srt_file, "w", encoding="utf-8") as f:
                for i, phrase in enumerate(phrases, 1):
                    start_time = self._format_srt_time(phrase["start"])
                    end_time = self._format_srt_time(phrase["end"])
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{phrase['text']}\n\n")
            self.report["logs"].append(f"Legendas sincronizadas geradas em {srt_file}")
            return str(srt_file)
        except Exception as e:
            self.report["logs"].append(f"Erro ao gerar legendas: {str(e)}")
            return None

    def extract_main_topic(self, text):
        if not text:
            return "Segmento sem texto"
        corrected_text = self.improve_transcript(text)
        filler_words = [
            "né", "então", "assim", "bem", "tipo", "como", "aí", "só", "agora",
            "aqui", "hoje", "esse", "essa", "isso", "este", "esta", "isto", "é", "foi"
        ]
        words = corrected_text.lower().split()
        filtered_words = [w for w in words if w not in filler_words and len(w) > 2]
        topic_triggers = [
            "importante", "atenção", "urgente", "notícia", "destaque", "lembrar",
            "observação", "dica", "conselho", "segredo", "tutorial", "como fazer",
            "passo a passo", "aprenda", "descubra", "veja", "saiba", "confira"
        ]
        topic_sentences = [s for s in corrected_text.split('.') if any(trigger in s.lower() for trigger in topic_triggers)]
        if topic_sentences:
            topic = topic_sentences[0].strip()
            if len(topic) > 80:
                topic = " ".join(topic.split()[:10]) + "..."
            return topic
        word_count = {}
        for word in filtered_words:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
        top_words = [word for word, _ in sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:5] if len(word) > 2]
        if top_words:
            for sentence in corrected_text.split('.'):
                if any(word in sentence.lower() for word in top_words):
                    topic = sentence.strip()
                    if len(topic) > 80:
                        topic = " ".join(topic.split()[:10]) + "..."
                    return topic
            return " ".join(top_words).capitalize()
        first_sentence = corrected_text.split('.')[0].strip() if corrected_text else text
        if len(first_sentence.split()) > 3:
            if len(first_sentence) > 80:
                first_sentence = " ".join(first_sentence.split()[:10]) + "..."
            return first_sentence
        return corrected_text[:80].strip() + "..." if len(corrected_text) > 80 else corrected_text

    def score_highlight(self, transcript):
        keywords = ["importante", "atenção", "urgente", "notícia", "destaque"]
        score = sum(transcript.lower().count(word) * 10 for word in keywords)
        return score

    def save_highlight_clip(self, segment_path, transcript_data, output_folder):
        try:
            Path(output_folder).mkdir(parents=True, exist_ok=True)
            topic = self.extract_main_topic(transcript_data["text"].strip())
            safe_topic = ''.join(c for c in topic if c.isalnum() or c.isspace() or c in '.,!?-_')
            safe_topic = safe_topic.rstrip('.,!?-_').strip()
            clip_name = f"{safe_topic}.mp4"
            output_file = Path(output_folder) / clip_name
            command = [
                'ffmpeg',
                '-i', str(segment_path),
                '-c', 'copy',
                '-y',
                str(output_file)
            ]
            subprocess.run(command, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            self.report["logs"].append(f"Highlight salvo: {output_file}")
            self.generate_srt(transcript_data, output_folder, safe_topic)
            return str(output_file)
        except subprocess.CalledProcessError as e:
            self.report["logs"].append(f"Erro ao salvar highlight: {str(e)}")
            return None

    def merge_highlights(self, highlight_clips, output_folder, theme_name="video_tematico"):
        try:
            Path(output_folder).mkdir(parents=True, exist_ok=True)
            output_file = Path(output_folder) / f"{theme_name}.mp4"
            if not highlight_clips:
                self.report["logs"].append("Nenhum highlight para compilar")
                return None
            if len(highlight_clips) == 1:
                shutil.copy2(highlight_clips[0], str(output_file))
                return str(output_file)
            else:
                temp_list_file = "file_list.txt"
                with open(temp_list_file, "w", encoding="utf-8") as f:
                    for clip in highlight_clips:
                        f.write(f"file '{clip}'\n")
                command = [
                    'ffmpeg',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', temp_list_file,
                    '-c', 'copy',
                    '-y',
                    str(output_file)
                ]
                subprocess.run(command, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    os.remove(temp_list_file)
                except:
                    pass
            self.report["logs"].append(f"Highlights compilados em {output_file}")
            return str(output_file)
        except Exception as e:
            self.report["logs"].append(f"Erro ao compilar highlights: {str(e)}")
            return None

    def remove_silence(self, video_path, output_path):
        try:
            # Primeiro precisamos extrair o áudio
            temp_audio = "temp_audio.wav"
            extract_command = [
                'ffmpeg',
                '-i', video_path,
                '-q:a', '0',
                '-map', 'a',
                '-y',
                temp_audio
            ]
            subprocess.run(extract_command, check=True, creationflags=subprocess.CREATE_NO_WINDOW)

            # Processar o áudio com pydub
            audio = AudioSegment.from_file(temp_audio, format="wav")
            chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40, keep_silence=100)

            if not chunks:
                self.report["logs"].append("Nenhum silêncio detectado para remover.")
                os.remove(temp_audio)
                return video_path

            # Concatenar os chunks sem silêncio
            processed_audio = chunks[0]
            for chunk in chunks[1:]:
                processed_audio += chunk

            # Salvar o áudio processado
            processed_audio_path = "processed_audio.wav"
            processed_audio.export(processed_audio_path, format="wav")

            # Substituir o áudio no vídeo
            temp_output = "temp_output.mp4"
            replace_command = [
                'ffmpeg',
                '-i', video_path,
                '-i', processed_audio_path,
                '-c:v', 'copy',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',
                '-y',
                temp_output
            ]
            subprocess.run(replace_command, check=True, creationflags=subprocess.CREATE_NO_WINDOW)

            # Mover para o destino final
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            final_command = [
                'ffmpeg',
                '-i', temp_output,
                '-c', 'copy',
                '-y',
                output_path
            ]
            subprocess.run(final_command, check=True, creationflags=subprocess.CREATE_NO_WINDOW)

            # Limpar arquivos temporários
            try:
                os.remove(temp_audio)
                os.remove(processed_audio_path)
                os.remove(temp_output)
            except:
                pass

            self.report["logs"].append(f"Silêncios removidos e vídeo salvo em {output_path}")
            return output_path
        except Exception as e:
            self.report["logs"].append(f"Erro ao remover silêncios: {str(e)}")
            return None

    def export_video(self, video_path, output_path, format):
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            command = [
                'ffmpeg',
                '-i', video_path,
                '-c:v', 'copy',
                '-c:a', 'copy',
                '-y',
                output_path
            ]
            subprocess.run(command, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            self.report["logs"].append(f"Vídeo exportado para {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            self.report["logs"].append(f"Erro ao exportar vídeo: {str(e)}")
            return None

    def download_video(self, url, output_path):
        try:
            # Verificar se o yt-dlp está instalado
            if not shutil.which("yt-dlp"):
                self.report["logs"].append("yt-dlp não encontrado. Por favor, instale-o para baixar vídeos.")
                return None

            # Crie o diretório de saída se não existir
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Use yt-dlp para baixar o vídeo
            command = [
                'yt-dlp',
                '-f', 'best',
                '-o', output_path,
                url
            ]
            subprocess.run(command, check=True, creationflags=subprocess.CREATE_NO_WINDOW)

            # Verifique se o arquivo existe
            if os.path.exists(output_path):
                self.report["logs"].append(f"Vídeo baixado e salvo em {output_path}")
                return output_path
            else:
                self.report["logs"].append("Falha ao baixar o vídeo")
                return None
        except Exception as e:
            self.report["logs"].append(f"Erro ao baixar vídeo: {str(e)}")
            return None

    def create_story_format(self, video_path, output_path):
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Usar FFmpeg para redimensionar para formato story
            command = [
                'ffmpeg',
                '-i', video_path,
                '-vf', 'scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2',
                '-c:a', 'copy',
                '-y',
                output_path
            ]
            subprocess.run(command, check=True, creationflags=subprocess.CREATE_NO_WINDOW)

            self.report["logs"].append(f"Vídeo em formato story salvo em {output_path}")
            return output_path
        except Exception as e:
            self.report["logs"].append(f"Erro ao criar vídeo em formato story: {str(e)}")
            return None

class VideoPlayer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_widget = QVideoWidget(self)
        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)

        # Configurar volume padrão
        self.audio_output.setVolume(0.5)

        self.play_pause_button = QPushButton("Play/Pause")
        self.play_pause_button.clicked.connect(self.toggle_play_pause)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.media_player.stop)

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(50)
        self.volume_slider.valueChanged.connect(self.set_volume)

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setRange(0, 100)
        self.timeline_slider.sliderMoved.connect(self.set_position)

        layout = QVBoxLayout(self)
        layout.addWidget(self.video_widget)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.play_pause_button)
        controls_layout.addWidget(self.stop_button)
        layout.addLayout(controls_layout)

        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("Volume:"))
        volume_layout.addWidget(self.volume_slider)
        layout.addLayout(volume_layout)

        layout.addWidget(self.timeline_slider)
        self.setLayout(layout)

        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)

    def set_video(self, video_path):
        if video_path and os.path.exists(video_path):
            self.media_player.setSource(QUrl.fromLocalFile(video_path))
            self.media_player.play()
        else:
            print(f"Arquivo de vídeo não encontrado: {video_path}")

    def toggle_play_pause(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def set_position(self, position):
        if self.media_player.duration() > 0:
            self.media_player.setPosition(int(self.media_player.duration() * position / 100))

    def set_volume(self, volume):
        self.audio_output.setVolume(volume / 100)

    def position_changed(self, position):
        if self.media_player.duration() > 0:
            self.timeline_slider.setValue(int(position * 100 / self.media_player.duration()))

    def duration_changed(self, duration):
        self.timeline_slider.setRange(0, duration)

class VideoProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VISORIA PRO V3")
        self.setGeometry(100, 100, 1200, 800)
        self.processor = VideoProcessor()
        self.setup_ui()

    def setup_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # Menu
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("Arquivo")
        edit_menu = menu_bar.addMenu("Editar")
        help_menu = menu_bar.addMenu("Ajuda")

        open_action = QAction("Abrir", self)
        open_action.triggered.connect(self.select_video)
        file_menu.addAction(open_action)

        export_action = QAction("Exportar", self)
        export_action.triggered.connect(self.export_video)
        file_menu.addAction(export_action)

        exit_action = QAction("Sair", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Layout esquerdo (principal)
        left_layout = QVBoxLayout()

        # Área de seleção de vídeo
        self.video_path = ""
        video_select_layout = QHBoxLayout()
        video_select_layout.addWidget(QLabel("Escolha o vídeo:"))
        self.video_select_button = QPushButton("Selecionar Vídeo")
        self.video_select_button.clicked.connect(self.select_video)
        video_select_layout.addWidget(self.video_select_button)
        left_layout.addLayout(video_select_layout)

        # Configurações
        cut_time_layout = QHBoxLayout()
        cut_time_layout.addWidget(QLabel("Tempo de Corte (s):"))
        self.corte_entry = QSpinBox()
        self.corte_entry.setRange(10, 300)
        self.corte_entry.setValue(45)
        cut_time_layout.addWidget(self.corte_entry)
        left_layout.addLayout(cut_time_layout)

        # Player de vídeo
        self.video_player = VideoPlayer()
        left_layout.addWidget(self.video_player)

        # Botões de processamento
        process_layout = QHBoxLayout()
        self.process_button = QPushButton("Processar")
        self.process_button.clicked.connect(self.process_video)
        process_layout.addWidget(self.process_button)

        self.download_button = QPushButton("Baixar Vídeo")
        self.download_button.clicked.connect(self.download_video)
        process_layout.addWidget(self.download_button)

        self.export_button = QPushButton("Exportar")
        self.export_button.clicked.connect(self.export_video)
        process_layout.addWidget(self.export_button)

        left_layout.addLayout(process_layout)

        # Barra de progresso
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        left_layout.addWidget(self.progress_bar)

        # Layout direito (painel lateral)
        right_layout = QVBoxLayout()
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        right_widget.setFixedWidth(300)  # Reduzido pela metade

        # Lista de segmentos
        segments_group = QFrame()
        segments_group.setFrameShape(QFrame.Shape.StyledPanel)
        segments_layout = QVBoxLayout(segments_group)
        segments_layout.addWidget(QLabel("<h3>Segmentos</h3>"))

        self.segments_list = QListWidget()
        self.segments_list.itemClicked.connect(self.play_segment)
        segments_layout.addWidget(self.segments_list)

        right_layout.addWidget(segments_group)

        # Painel de transcrição
        transcript_group = QFrame()
        transcript_group.setFrameShape(QFrame.Shape.StyledPanel)
        transcript_layout = QVBoxLayout(transcript_group)
        transcript_layout.addWidget(QLabel("<h3>Transcrição</h3>"))

        self.transcript_area = QScrollArea()
        self.transcript_widget = QLabel("Nenhuma transcrição disponível")
        self.transcript_widget.setWordWrap(True)
        self.transcript_area.setWidget(self.transcript_widget)
        self.transcript_area.setWidgetResizable(True)
        transcript_layout.addWidget(self.transcript_area)

        right_layout.addWidget(transcript_group)

        # Adicionar layouts ao layout principal
        main_layout.addLayout(left_layout, 7)
        main_layout.addWidget(right_widget, 3)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Pronto")

        # Configurações iniciais
        self.output_folder = ""
        self.current_segment = ""
        self.transcripts = {}
        self.highlights = []
        self.update_ui_state(False)

        # Aplicar estilo
        self.apply_styles()

    def apply_styles(self):
        # Estilo moderno
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QFrame {
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: white;
            }
            QPushButton {
                background-color: #2980b9;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
            QPushButton:pressed {
                background-color: #1c6ea4;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2980b9;
                width: 10px;
            }
            QLabel {
                color: #333;
            }
            QScrollArea {
                border: none;
            }
        """)

    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar Vídeo", "", "Arquivos de Vídeo (*.mp4 *.avi *.mkv *.mov);;Todos os Arquivos (*)"
        )
        if file_path:
            self.video_path = file_path
            self.video_player.set_video(file_path)
            self.processor.report["video_original"] = file_path
            self.status_bar.showMessage(f"Vídeo carregado: {os.path.basename(file_path)}")
            self.update_ui_state(True)
            self.create_thumbnail(file_path)

    def create_thumbnail(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Não foi possível abrir o vídeo: {video_path}")
                return

            # Capturar o frame do meio do vídeo
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                print("Não foi possível capturar o frame para thumbnail")
                return

            # Redimensionar e salvar temporariamente
            height, width = frame.shape[:2]
            new_width = 200
            new_height = int(height * (new_width / width))
            resized = cv2.resize(frame, (new_width, new_height))

            # Converter BGR para RGB
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # Criar QImage e QPixmap
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            # Criar e adicionar um item à lista com a thumbnail
            item = QListWidgetItem(f"Vídeo Original")
            item.setData(Qt.ItemDataRole.UserRole, video_path)
            item.setIcon(QIcon(pixmap))
            self.segments_list.addItem(item)

        except Exception as e:
            print(f"Erro ao criar thumbnail: {e}")

    def update_ui_state(self, video_loaded=False):
        self.process_button.setEnabled(video_loaded)
        self.export_button.setEnabled(video_loaded and self.processor.report["segmentos"])

    def process_video(self):
        if not self.video_path:
            QMessageBox.warning(self, "Aviso", "Por favor, selecione um vídeo primeiro.")
            return

        # Configurar pasta de saída
        video_dir = os.path.dirname(self.video_path)
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.output_folder = os.path.join(video_dir, f"{video_name}_processado")

        # Configurar parâmetros
        self.processor.report["tempo_cortes"] = self.corte_entry.value()

        # Iniciar processamento em thread separada
        self.processing_thread = ProcessingThread(self.processor, self.video_path, self.output_folder)
        self.processing_thread.progress_update.connect(self.update_progress)
        self.processing_thread.segment_ready.connect(self.add_segment)
        self.processing_thread.finished.connect(self.processing_finished)

        # UI updates
        self.status_bar.showMessage("Processando vídeo...")
        self.process_button.setEnabled(False)
        self.segments_list.clear()
        self.transcript_widget.setText("Processando...")

        # Iniciar thread
        self.processing_thread.start()

    def update_progress(self, progress, message):
        self.progress_bar.setValue(progress)
        self.status_bar.showMessage(message)

    def add_segment(self, segment_path, transcript_text, thumbnail_path):
        try:
            # Armazenar transcrição
            self.transcripts[segment_path] = transcript_text

            # Criar item com thumbnail
            item = QListWidgetItem(os.path.basename(segment_path))
            item.setData(Qt.ItemDataRole.UserRole, segment_path)

            # Adicionar thumbnail se existir
            if thumbnail_path and os.path.exists(thumbnail_path):
                pixmap = QPixmap(thumbnail_path)
                item.setIcon(QIcon(pixmap))

            self.segments_list.addItem(item)
        except Exception as e:
            print(f"Erro ao adicionar segmento: {e}")

    def play_segment(self, item):
        segment_path = item.data(Qt.ItemDataRole.UserRole)
        if segment_path and os.path.exists(segment_path):
            self.video_player.set_video(segment_path)
            self.current_segment = segment_path

            # Atualizar transcrição
            if segment_path in self.transcripts:
                self.transcript_widget.setText(self.transcripts[segment_path])
                self.transcript_area.ensureWidgetVisible(self.transcript_widget)
            else:
                self.transcript_widget.setText("Sem transcrição disponível")

    def processing_finished(self):
        self.status_bar.showMessage("Processamento concluído")
        self.process_button.setEnabled(True)
        self.update_ui_state(True)

    def download_video(self):
        url, ok = QInputDialog.getText(self, "Baixar Vídeo", "Insira a URL do vídeo:")
        if ok and url:
            video_dir = os.path.expanduser("~/Downloads")
            output_path = os.path.join(video_dir, "video_baixado.mp4")

            # Iniciar download em thread separada
            self.download_thread = DownloadThread(self.processor, url, output_path)
            self.download_thread.progress_update.connect(self.update_progress)
            self.download_thread.finished.connect(self.download_finished)

            self.status_bar.showMessage("Baixando vídeo...")
            self.download_button.setEnabled(False)
            self.download_thread.start()

    def download_finished(self, success, video_path):
        if success:
            self.video_path = video_path
            self.video_player.set_video(video_path)
            self.status_bar.showMessage(f"Vídeo baixado: {os.path.basename(video_path)}")
            self.update_ui_state(True)
            self.create_thumbnail(video_path)
        else:
            QMessageBox.warning(self, "Erro", "Falha ao baixar o vídeo.")
        self.download_button.setEnabled(True)

    def export_video(self):
        if not self.processor.report["segmentos"]:
            QMessageBox.warning(self, "Aviso", "Nenhum segmento processado para exportar.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Exportar Vídeo", "", "Arquivo MP4 (*.mp4);;Arquivo AVI (*.avi)"
        )

        if file_path:
            # Selecionar segmento atual ou compilar
            if self.current_segment and os.path.exists(self.current_segment):
                source_video = self.current_segment
            else:
                source_video = self.processor.report["segmentos"][0]

            self.processor.export_video(source_video, file_path, os.path.splitext(file_path)[1])
            self.status_bar.showMessage(f"Vídeo exportado: {file_path}")

class ProcessingThread(QThread):
    progress_update = pyqtSignal(int, str)
    segment_ready = pyqtSignal(str, str, str)

    def __init__(self, processor, video_path, output_folder):
        super().__init__()
        self.processor = processor
        self.video_path = video_path
        self.output_folder = output_folder

    def run(self):
        try:
            # Criar pasta de saída
            Path(self.output_folder).mkdir(parents=True, exist_ok=True)

            # Extrair áudio e processar em segmentos
            self.progress_update.emit(10, "Extraindo áudio...")
            audio_path = self.processor.extract_audio(self.video_path, self.output_folder)

            if not audio_path:
                self.progress_update.emit(0, "Falha ao extrair áudio")
                return

            # Inicializar reconhecimento de voz
            self.progress_update.emit(20, "Inicializando reconhecimento de voz...")
            if not self.processor.initialize_vosk():
                self.progress_update.emit(0, "Modelo de reconhecimento não encontrado")
                return

            # Segmentar vídeo
            self.progress_update.emit(30, "Segmentando vídeo...")
            self.processor.segment_video(
                self.video_path,
                self.output_folder,
                self.processor.report["tempo_cortes"]
            )

            # Processar segmentos
            total_segments = len(self.processor.report["segmentos"])
            if total_segments == 0:
                self.progress_update.emit(0, "Nenhum segmento gerado")
                return

            # Processar cada segmento
            for i, segment_path in enumerate(self.processor.report["segmentos"]):
                segment_name = os.path.basename(segment_path)
                segment_progress = 40 + (50 * i // total_segments)
                self.progress_update.emit(segment_progress, f"Processando segmento {i+1}/{total_segments}")

                # Extrair áudio do segmento
                segment_audio = self.processor.extract_audio(
                    segment_path,
                    os.path.join(self.output_folder, f"segment_audio_{i}")
                )

                if segment_audio:
                    # Gerar transcrição
                    transcript_data = self.processor.generate_transcript(segment_audio)

                    # Criar SRT
                    self.processor.generate_srt(
                        transcript_data,
                        os.path.join(self.output_folder, "subtitles"),
                        f"segment{i}"
                    )

                    # Gerar thumbnail
                    thumbnail_path = self.create_thumbnail(segment_path, i)

                    # Emitir sinal para atualizar UI
                    self.segment_ready.emit(
                        segment_path,
                        transcript_data["text"],
                        thumbnail_path
                    )

            self.progress_update.emit(100, "Processamento concluído")

        except Exception as e:
            self.progress_update.emit(0, f"Erro: {str(e)}")

    def create_thumbnail(self, video_path, index):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None

            # Capturar frame do meio
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                return None

            # Redimensionar
            height, width = frame.shape[:2]
            new_width = 200
            new_height = int(height * (new_width / width))
            resized = cv2.resize(frame, (new_width, new_height))

            # Salvar thumbnail
            thumbnail_dir = os.path.join(self.output_folder, "thumbnails")
            Path(thumbnail_dir).mkdir(parents=True, exist_ok=True)
            thumbnail_path = os.path.join(thumbnail_dir, f"thumb_{index}.jpg")
            cv2.imwrite(thumbnail_path, resized)

            return thumbnail_path
        except Exception as e:
            print(f"Erro ao criar thumbnail: {e}")
            return None

class DownloadThread(QThread):
    progress_update = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, processor, url, output_path):
        super().__init__()
        self.processor = processor
        self.url = url
        self.output_path = output_path

    def run(self):
        try:
            self.progress_update.emit(20, "Iniciando download...")
            video_path = self.processor.download_video(self.url, self.output_path)

            if video_path and os.path.exists(video_path):
                self.progress_update.emit(100, "Download concluído")
                self.finished.emit(True, video_path)
            else:
                self.progress_update.emit(0, "Falha ao baixar vídeo")
                self.finished.emit(False, "")
        except Exception as e:
            self.progress_update.emit(0, f"Erro: {str(e)}")
            self.finished.emit(False, "")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoProcessorApp()
    window.show()
    sys.exit(app.exec())

