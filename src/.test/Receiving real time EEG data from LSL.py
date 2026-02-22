import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from PySide6.QtGui import QPalette, QColor
from pylsl import StreamInlet, resolve_byprop
import sys
import csv
import time

class LSLTimeScope(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Live Scope + Terminal Logger")

        # --- SETTINGS ---
        self.SAMPLING_RATE = 250
        self.TIME_WINDOW = 10 
        self.AMPLITUDE_SCALE = 100 

        # --- LSL SETUP ---
        print("Resolving LSL stream...")
        streams = resolve_byprop("type", "EEG", timeout=5)
        if not streams:
            print("Error: No EEG stream found. Ensure Program 1 is running.")
            sys.exit()
            
        self.inlet = StreamInlet(streams[0])
        info = self.inlet.info()
        self.CHANNEL_COUNT = info.channel_count()
        
        # --- RECORDING & PRINT STATE ---
        self.is_recording = False
        self.csv_file = None
        self.csv_writer = None
        self.start_lsl_time = None 
        self.print_counter = 0 # To control terminal scroll speed

        # --- GUI LAYOUT ---
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        self.record_btn = QtWidgets.QPushButton("Start Recording")
        self.record_btn.setCheckable(True)
        self.record_btn.setStyleSheet("background-color : #4CAF50; color: white; height: 50px; font-weight: bold;")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.layout.addWidget(self.record_btn)

        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)

        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.setup_plot()

        self.MAX_POINTS = self.SAMPLING_RATE * self.TIME_WINDOW 
        self.t_vec = np.arange(self.MAX_POINTS) / self.SAMPLING_RATE
        self.data_buffer = np.zeros((self.MAX_POINTS, self.CHANNEL_COUNT))
        self.sample_index = 0

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_all)
        self.timer.start(40) 

    def setup_plot(self):
        palette = self.palette()
        self.plot_widget.setBackground(palette.color(QPalette.ColorRole.Window))
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.setYRange(0, self.CHANNEL_COUNT)
        self.plot_item.setXRange(0, self.TIME_WINDOW)
        self.curves = [self.plot_item.plot(pen=pg.mkPen(QColor(palette.color(QPalette.ColorRole.WindowText)), width=1)) 
                       for _ in range(self.CHANNEL_COUNT)]

    def toggle_recording(self):
        if self.record_btn.isChecked():
            self.is_recording = True
            self.start_lsl_time = None 
            self.record_btn.setText("STOP RECORDING")
            self.record_btn.setStyleSheet("background-color : #ff4c4c; color: white; font-weight: bold; height: 50px;")
            
            filename = f"EEG_Capture_{time.strftime('%Y%m%d-%H%M%S')}.csv"
            self.csv_file = open(filename, mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            header = ['Time_Seconds'] + [f'CH{i+1}' for i in range(self.CHANNEL_COUNT)]
            self.csv_writer.writerow(header)
            self.status_bar.showMessage(f"Recording to: {filename}")
        else:
            self.stop_recording_logic()

    def stop_recording_logic(self):
        self.is_recording = False
        self.record_btn.setText("Start Recording")
        self.record_btn.setStyleSheet("background-color : #4CAF50; color: white; font-weight: bold; height: 50px;")
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.status_bar.showMessage("File Saved Successfully")

    def update_all(self):
        while True:
            sample, timestamp = self.inlet.pull_sample(timeout=0.0)
            if sample is None:
                break
            
            # --- TERMINAL PRINTING LOGIC ---
            # Only print every 25th sample (~10 times per second) so it's readable
            self.print_counter += 1
            if self.print_counter % 25 == 0:
                # Create a string: "CH1: 12.3 | CH2: -5.4 | ..."
                print_str = " | ".join([f"CH{i+1}: {val:7.2f}" for i, val in enumerate(sample)])
                print(f"[LIVE] {print_str}")

            # --- CSV LOGGING ---
            if self.is_recording:
                if self.start_lsl_time is None:
                    self.start_lsl_time = timestamp
                self.csv_writer.writerow([f"{(timestamp - self.start_lsl_time):.4f}"] + sample)

            # --- DATA BUFFER ---
            idx = self.sample_index % self.MAX_POINTS
            self.data_buffer[idx, :] = sample
            self.sample_index += 1

        # --- PLOT REFRESH ---
        N = 5 
        for i, curve in enumerate(self.curves):
            offset = self.CHANNEL_COUNT - i - 0.5
            curve.setData(self.t_vec[::N], self.data_buffer[::N, i] / self.AMPLITUDE_SCALE + offset)

    def closeEvent(self, event):
        self.stop_recording_logic()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = LSLTimeScope()
    window.resize(1024, 600)
    window.show()
    sys.exit(app.exec())