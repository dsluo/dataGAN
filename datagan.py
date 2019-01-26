from pathlib import Path

from PyQt5 import QtCore
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QLabel, QVBoxLayout, QProgressBar, \
    QComboBox, QLineEdit, QMessageBox

from esrgan.esrgan import ESRGAN


# noinspection PyArgumentList,PyUnresolvedReferences,PyCallByClass
class MainWidget(QWidget):
    DEFAULT_MODEL_PATH = (Path(__file__).parent / 'esrgan/models').absolute()

    def __init__(self):
        super().__init__()

        self.setMinimumWidth(400)

        self.inputs = None
        self.output_directory = None
        self.model = str(self.DEFAULT_MODEL_PATH / 'RRDB_ESRGAN_x4.pth')

        self.setWindowTitle('dataGAN')

        self.layout = QVBoxLayout()

        self.input_select = QPushButton('Select files', self)
        self.input_select.clicked.connect(self.select_inputs)
        self.input_label = QLabel('Input file(s)')

        self.output_select = QPushButton('Select Directory', self)
        self.output_select.clicked.connect(self.select_output)
        self.output_label = QLabel('Output Directory')

        self.device = QComboBox(self)
        self.device.addItem('CUDA')
        self.device.addItem('CPU')
        self.device.setCurrentIndex(0)
        self.device_label = QLabel('Device')

        self.scale = QLineEdit()
        self.scale.setValidator(QIntValidator(bottom=1))
        self.scale.setText('4')
        self.scale_label = QLabel('Scale')

        self.model_select = QPushButton(self.model, self)
        self.model_select.clicked.connect(self.select_model)
        self.model_label = QLabel('Neural Network Model')

        self.process = QPushButton('Process', self)
        self.process.clicked.connect(self.upscale)
        self.process.setDisabled(True)
        self.result_label = QLabel('Select an input and output.')

        self.progress = QProgressBar(self)
        self.progress.setValue(0)

        self.error = QMessageBox()
        self.error.setText('An error occurred while running.')
        self.error.setWindowTitle('Error')
        self.error.setIcon(QMessageBox.Critical)

        self.layout.addWidget(self.input_label)
        self.layout.addWidget(self.input_select)
        self.layout.addWidget(self.output_label)
        self.layout.addWidget(self.output_select)
        self.layout.addWidget(self.device_label)
        self.layout.addWidget(self.device)
        self.layout.addWidget(self.scale_label)
        self.layout.addWidget(self.scale)
        self.layout.addWidget(self.model_label)
        self.layout.addWidget(self.model_select)
        self.layout.addWidget(self.process)
        self.layout.addWidget(self.result_label)
        self.layout.addWidget(self.progress)

        self.setLayout(self.layout)

        self.worker = None
        self.thread = None

    def select_inputs(self):
        self.inputs, _ = QFileDialog.getOpenFileNames(self, 'Open File', str(Path.home()))
        if len(self.inputs) == 1:
            name = Path(self.inputs[0]).name
            self.input_select.setText(name)
        elif len(self.inputs) == 0:
            self.input_select.setText('Select files')
        else:
            self.input_select.setText(f'{len(self.inputs)} files')

        self.process.setDisabled(not (self.inputs and self.output_directory))

    def select_output(self):
        self.output_directory = QFileDialog.getExistingDirectory(self, 'Output Directory', str(Path.home()))
        self.output_select.setText(self.output_directory)
        self.process.setDisabled(not (self.inputs and self.output_directory))

    def select_model(self):
        self.model, _ = QFileDialog.getOpenFileName(self, 'Open File', str(self.DEFAULT_MODEL_PATH),
                                                    'NN Models (*.pth);;All Files (*)')
        self.model_select.setText(self.model)

    def upscale(self):
        try:
            device = self.device.currentText().lower()
            scale = int(self.scale.text())
            self.worker = Worker(self.inputs, self.output_directory, scale, device, self.model)
            self.thread = QtCore.QThread()
            self.worker.moveToThread(self.thread)
            self.worker.finished_one.connect(self.incr_progress)
            self.worker.finished.connect(self.thread.quit)

            self.progress.setRange(0, len(self.inputs))
            self.progress.setValue(0)

            self.disable_widgets(True)
            self.thread.started.connect(self.worker.process_esrgan)
            self.thread.finished.connect(self.enable_widgets)

            self.thread.start()
        except Exception as ex:
            self.error.setDetailedText(str(ex))
            self.error.show()

    def disable_widgets(self, disable):
        for widget in (self.input_select, self.output_select, self.device, self.scale, self.process):
            widget.setDisabled(disable)

    @QtCore.pyqtSlot()
    def enable_widgets(self):
        self.disable_widgets(False)

    @QtCore.pyqtSlot()
    def incr_progress(self):
        self.progress.setValue(self.progress.value() + 1)


# noinspection PyArgumentList
class Worker(QtCore.QObject):
    OUTPUT_FORMAT = '{file.stem}-upscaled{file.suffix}'

    finished = QtCore.pyqtSignal()
    finished_one = QtCore.pyqtSignal()

    def __init__(self, inputs, output_directory, upscale, device, model_path):
        super().__init__()
        self.inputs = inputs
        self.output_directory = output_directory
        self.esrgan = ESRGAN(model_path, device, upscale)

    @QtCore.pyqtSlot()
    def process_esrgan(self):
        for input in self.inputs:
            input_file = Path(input)
            output = Path(self.output_directory) / self.OUTPUT_FORMAT.format(file=input_file)
            output = str(output.absolute())
            self.esrgan.upscale(input, output)
            self.finished_one.emit()
        self.finished.emit()


if __name__ == '__main__':
    app = QApplication([])
    app.setApplicationName('dataGAN')

    main = MainWidget()
    main.show()

    app.exec_()
