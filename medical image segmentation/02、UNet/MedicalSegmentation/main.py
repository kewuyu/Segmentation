import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog,QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from inference import infer
from inference import process
class ImageProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('UNet Image Processor')
        self.setGeometry(300, 300, 1000, 500)  # 调整窗口大小

        self.mainLayout = QVBoxLayout()

        self.imagesLayout = QHBoxLayout()

        self.originalImageLabel = QLabel(self)
        self.imagesLayout.addWidget(self.originalImageLabel)

        self.processedImageLabel = QLabel(self)
        self.imagesLayout.addWidget(self.processedImageLabel)

        self.mainLayout.addLayout(self.imagesLayout)

        self.btnLoad = QPushButton('Load Image', self)
        self.btnLoad.clicked.connect(self.loadImage)
        self.mainLayout.addWidget(self.btnLoad)

        self.setLayout(self.mainLayout)

    def loadImage(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*);;JPEG (*.jpeg;*.jpg);;PNG (*.png)", options=options)
        if fileName:
            self.displayImage(fileName)

    def displayImage(self, path):
        overlay_image, original_image = infer(path)

        self.showImageOnLabel(original_image, self.originalImageLabel)
        self.showImageOnLabel(overlay_image, self.processedImageLabel)

    def showImageOnLabel(self, image, label):
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)

        # 调整图像大小以适应标签
        pixmap = QPixmap.fromImage(qImg)
        resizedPixmap = pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio)

        label.setPixmap(resizedPixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessor()
    ex.show()
    sys.exit(app.exec_())