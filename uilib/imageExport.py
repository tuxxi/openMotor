from matplotlib import pyplot as plt
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QDialogButtonBox
from .views.ImageExporter_ui import Ui_ImageExporter

class ImageExportMenu(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.ui = Ui_ImageExporter()
        self.ui.setupUi(self)
        self.simRes = None
        self.preferences = None
        self.ui.buttonBox.accepted.connect(self.exportImage)

        self.ui.independent.setupChecks(False)
        self.ui.dependent.setupChecks(True)

    def exportImage(self):
        if len(self.ui.independent.getSelectedChannels()) != 1 or len(self.ui.dependent.getSelectedChannels()) == 0:
            QApplication.instance().outputMessage("You must select an independent channel and at least one dependent channel.")
            return
        xChannel = self.ui.independent.getSelectedChannels()[0]
        yChannels = self.ui.dependent.getSelectedChannels()
        xAxisUnit = self.preferences.getUnit(self.simRes.channels[xChannel].unit)
        path = QFileDialog.getSaveFileName(None, 'Save Image', '', 'Image Files (*.png)')[0]
        if path is not None and path != '':
            if path[-4:] != '.png':
                path += '.png'
            legend = []
            for channelName in yChannels:
                channel = self.simRes.channels[channelName]
                yUnit = self.preferences.getUnit(channel.unit)
                plt.plot(self.simRes.channels[xChannel].getData(xAxisUnit), channel.getData(yUnit))
                if channel.valueType in (int, float):
                    if yUnit != '':
                        legend.append(channel.name + ' - ' + yUnit)
                    else:
                        legend.append(channel.name)
                elif channel.valueType in (list, tuple):
                    if yUnit != '':
                        for i in range(len(channel.getData()[0])):
                            legend.append(channel.name + ' - Grain ' + str(i + 1) + ' - ' + yUnit)
                    else:
                        for i in range(len(channel.getData()[0])):
                            legend.append(channel.name + ' - Grain ' + str(i + 1))
            plt.legend(legend, ncol=1, bbox_to_anchor=(1.04, 1))
            plt.xlabel(self.simRes.channels[xChannel].name + ' - ' + xAxisUnit)

            plt.title(self.simRes.getDesignation())
            plt.savefig(path, bbox_inches="tight")
            plt.clf()

    def setPreferences(self, pref):
        self.preferences = pref

    def open(self):
        self.ui.buttonBox.button(QDialogButtonBox.Ok).setEnabled(self.simRes is not None)
        self.show()

    def acceptSimResult(self, simRes):
        self.simRes = simRes
