import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QFileDialog, QSlider)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl, Qt


def helper_set_slider_int_value(slider, x):
    slider.tracking = True
    slider.value = int(x)
    slider.sliderPosition = int(x)
    slider.update()
    slider.repaint()


class VideoProcessorApp(QWidget):
    def __init__(self):
        super().__init__()

        self.valid_input_duration = False
        self.valid_output_duration = False

        # Main layout
        layout = QVBoxLayout()

        # 'Add Video' button
        self.btnAddVideo = QPushButton("Add Video")
        self.btnAddVideo.clicked.connect(self.open_file)
        layout.addWidget(self.btnAddVideo)

        # Input Video Widget
        self.inputVideoWidget = QVideoWidget()
        self.inputVideoPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.inputVideoPlayer.setVideoOutput(self.inputVideoWidget)
        self.inputVideoPlayer.durationChanged.connect(self.set_valid_input_duration)
        self.inputVideoPlayer.positionChanged.connect(self.set_input_slider_position)
        self.inputVideoPlayer.mediaStatusChanged.connect(self.input_check_media_status)

        self.inputVideoControls = QHBoxLayout()
        self.toggleInputButton = QPushButton("▶️")
        self.toggleInputButton.clicked.connect(self.toggle_input_video)
        self.inputVideoControls.addWidget(self.toggleInputButton)

        self.inputVideoSlider = QSlider(Qt.Horizontal)
        self.inputVideoSlider.valueChanged.connect(self.set_input_position)
        self.inputVideoControls.addWidget(self.inputVideoSlider)

        self.clearInputButton = QPushButton("Clear")
        self.clearInputButton.clicked.connect(self.clear_input_video)
        self.inputVideoControls.addWidget(self.clearInputButton)

        layout.addWidget(self.inputVideoWidget)
        layout.addLayout(self.inputVideoControls)

        # 'Process' button
        self.btnProcess = QPushButton("Process Video")
        self.btnProcess.clicked.connect(self.process_video)
        self.btnProcess.setEnabled(False)
        layout.addWidget(self.btnProcess)

        # Output Video Widget
        self.outputVideoWidget = QVideoWidget()
        self.outputVideoPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.outputVideoPlayer.setVideoOutput(self.outputVideoWidget)
        self.outputVideoPlayer.durationChanged.connect(self.set_valid_output_duration)
        self.outputVideoPlayer.positionChanged.connect(self.set_output_slider_position)
        self.outputVideoPlayer.mediaStatusChanged.connect(self.output_check_media_status)

        self.outputVideoControls = QHBoxLayout()
        self.toggleOutputButton = QPushButton("▶️")
        self.toggleOutputButton.clicked.connect(self.toggle_output_video)
        self.outputVideoControls.addWidget(self.toggleOutputButton)

        self.outputVideoSlider = QSlider(Qt.Horizontal)
        self.outputVideoSlider.valueChanged.connect(self.set_output_position)
        self.outputVideoControls.addWidget(self.outputVideoSlider)

        self.saveOutputButton = QPushButton("Save")
        self.saveOutputButton.clicked.connect(self.save_output_video)
        self.outputVideoControls.addWidget(self.saveOutputButton)

        layout.addWidget(self.outputVideoWidget)
        layout.addLayout(self.outputVideoControls)

        self.outputVideoWidget.hide()  # Hide output video widget until video is processed

        self.setLayout(layout)
        self.setWindowTitle("Video Processor App")
        self.setGeometry(100, 100, 800, 600)

    # == VIDEO Status check

    @staticmethod
    def check_media_status(video_widget, status):
        if status == QMediaPlayer.EndOfMedia:
            video_widget.stop()  # Stop the player
            video_widget.play()  # Start the player again from the beginning

    def input_check_media_status(self, status):
        self.check_media_status(self.inputVideoPlayer, status)

    def output_check_media_status(self, status):
        self.check_media_status(self.outputVideoPlayer, status)

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_name:
            self.inputVideoPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(file_name)))
            self.input_video_play()
            self.btnProcess.setEnabled(True)

    # === VIDEO Duration check

    def set_valid_input_duration(self, duration):
        self.valid_input_duration = bool(duration)

    def set_valid_output_duration(self, duration):
        self.valid_output_duration = bool(duration)

    # === Video actions

    @staticmethod
    def set_position(video_widget, position_percent):
        position_ms = int(video_widget.duration() / 100 * position_percent)
        video_widget.setPosition(position_ms)

    @staticmethod
    def video_stop(video_widget, btn_widget):
        video_widget.pause()
        btn_widget.setText("▶️")

    @staticmethod
    def video_play(video_widget, btn_widget):
        video_widget.play()
        btn_widget.setText("⏸️")

    # === Slider position

    @staticmethod
    def set_slider_position(valid_duration, slider_widget, video_widget, position_ms):
        if not valid_duration:
            slider_widget.setSliderPosition(0)
        else:
            video_duration = video_widget.duration()
            position_percent = (position_ms / video_duration) * 100
            slider_widget.setSliderPosition(int(position_percent))

    def set_input_slider_position(self, position_ms):
        self.set_slider_position(self.valid_input_duration, self.inputVideoSlider, self.inputVideoPlayer, position_ms)

    def set_output_slider_position(self, position_ms):
        self.set_slider_position(self.valid_output_duration, self.inputVideoSlider, self.inputVideoPlayer, position_ms)

    def set_input_position(self, position_percent):
        self.set_position(self.inputVideoPlayer, position_percent)

    # === INPUT video ===

    def input_video_play(self):
        self.video_play(self.inputVideoPlayer, self.toggleInputButton)

    def input_video_stop(self):
        self.video_stop(self.inputVideoPlayer, self.toggleInputButton)

    def toggle_input_video(self):
        if self.inputVideoPlayer.state() == QMediaPlayer.PlayingState:
            self.input_video_stop()
        else:
            self.input_video_play()

    def clear_input_video(self):
        self.inputVideoPlayer.setMedia(QMediaContent())
        self.btnProcess.setEnabled(False)
        self.set_input_position(0)

    # == OUTPUT video ===

    def process_video(self):
        # Placeholder for video processing logic
        processed_video_url = self.inputVideoPlayer.currentMedia().canonicalUrl()  # Mock the processing
        self.outputVideoPlayer.setMedia(QMediaContent(processed_video_url))
        self.outputVideoWidget.show()
        self.toggleOutputButton.setText("▶️")
        self.input_video_play()

    def output_video_play(self):
        self.video_play(self.outputVideoPlayer, self.toggleOutputButton)

    def output_video_stop(self):
        self.video_stop(self.outputVideoPlayer, self.toggleOutputButton)

    def toggle_output_video(self):
        if self.outputVideoPlayer.state() == QMediaPlayer.PlayingState:
            self.output_video_stop()
        else:
            self.output_video_play()

    def set_output_position(self, position_percent):
        self.set_position(self.outputVideoPlayer, position_percent)

    def save_output_video(self):
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if fileName:
            # Implement video saving logic here
            print("Save the video to:", fileName)

# Uncomment below to run the application
app = QApplication(sys.argv)
ex = VideoProcessorApp()
ex.show()
sys.exit(app.exec_())
