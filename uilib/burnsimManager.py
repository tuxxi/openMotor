import xml.etree.ElementTree as ET

from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QFileDialog, QApplication, QMessageBox

import motorlib.grains
import motorlib.units
import motorlib.propellant
import motorlib.motor

# BS type -> oM class for all grains we can import
SUPPORTED_GRAINS = {
    '1': motorlib.grains.BatesGrain,
    '2': motorlib.grains.DGrain,
    '3': motorlib.grains.MoonBurner,
    '5': motorlib.grains.CGrain,
    '6': motorlib.grains.XCore,
    '7': motorlib.grains.Finocyl
}

# BS type -> label for grains we know about but can't import
UNSUPPORTED_GRAINS = {
    '4': 'Star',
    '8': 'Tablet',
    '9': 'Pie Segment'
}

# oM class -> BS type for grains we can export
EXPORT_TYPES = {
    motorlib.grains.BatesGrain: '1',
    motorlib.grains.EndBurningGrain: '1',
    motorlib.grains.DGrain: '2',
    motorlib.grains.MoonBurner: '3',
    motorlib.grains.CGrain: '5',
    motorlib.grains.XCore: '6',
    motorlib.grains.Finocyl: '7'
}

# Attributes for the root element of the BSX file
bsxMotorAttrib = {
    'Name': '',
    'DiameterMM': '0',
    'Length': '0',
    'Delays': '0',
    'HardwareWeight': '0',
    'MFGCode': '',
    'ThrustMethod': '1',
    'ThrustCoefGiven': '1.2',
    'UnitsLinear': '1'
}

def inToM(value):
    """Converts a string containing a value in inches to a float of meters"""
    return motorlib.units.convert(float(value), 'in', 'm')

def mToIn(value):
    """Converts a float containing meters to a string of inches"""
    return str(motorlib.units.convert(value, 'm', 'in'))

class BurnsimManager(QObject):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.fileManager = app.fileManager

    def showImportMenu(self):
        """Open a dialog to pick the file to load and load it. Returns true if they load something, false otherwise"""
        if self.fileManager.unsavedCheck():
            path = QFileDialog.getOpenFileName(None, 'Import BurnSim motor', '', 'BurnSim Motor Files (*.bsx)')[0]
            if path != '' and path is not None:
                return self.importFile(path)
        return False

    def showExportMenu(self):
        """Open a dialog to pick the file to save to and dump the BSX version of the current motor to it"""
        path = QFileDialog.getSaveFileName(None, 'Export BurnSim motor', '', 'BurnSim Motor Files (*.bsx)')[0]
        if path == '' or path is None:
            return
        if path[-4:] != '.bsx':
            path += '.bsx'
        motor = self.fileManager.getCurrentMotor()
        self.exportFile(path, motor)

    def showWarning(self, text):
        """Show a dialog displaying some text"""
        msg = QMessageBox()
        msg.setText(text)
        msg.setWindowTitle("Warning")
        msg.exec_()

    def importFile(self, path):
        """Opens the BSX file located at path, generates a motor from it, and starts motor history there"""
        motor = motorlib.motor.Motor()
        motor.config.setProperties(self.app.preferencesManager.preferences.general.getProperties())
        tree = ET.parse(path)
        root = tree.getroot()
        errors = ''
        propSet = False
        for child in root:
            if child.tag == 'Nozzle':
                motor.nozzle.setProperty('throat', inToM(child.attrib['ThroatDia']))
                motor.nozzle.setProperty('exit', inToM(child.attrib['ExitDia']))
                motor.nozzle.setProperty('efficiency', float(child.attrib['NozzleEfficiency']) / 100)
                motor.nozzle.setProperty('divAngle', 15)
                motor.nozzle.setProperty('convAngle', 45)
                errors += 'Nozzle angles not specified, assumed to be 15° and 45°.\n'
            if child.tag == 'Grain':
                if child.attrib['Type'] in SUPPORTED_GRAINS:
                    motor.grains.append(SUPPORTED_GRAINS[child.attrib['Type']]())
                    motor.grains[-1].setProperty('diameter', inToM(child.attrib['Diameter']))
                    motor.grains[-1].setProperty('length', inToM(child.attrib['Length']))

                    grainType = child.attrib['Type']

                    if child.attrib['EndsInhibited'] == '1':
                        motor.grains[-1].setProperty('inhibitedEnds', 'Top')
                    elif child.attrib['EndsInhibited'] == '2':
                        motor.grains[-1].setProperty('inhibitedEnds', 'Both')

                    if grainType in ('1', '3', '7'): # Grains with core diameter
                        motor.grains[-1].setProperty('coreDiameter', inToM(child.attrib['CoreDiameter']))

                    if grainType == '2': # D grain specific properties
                        motor.grains[-1].setProperty('slotOffset', inToM(child.attrib['EdgeOffset']))

                    elif grainType == '3': # Moonburner specific properties
                        motor.grains[-1].setProperty('coreOffset', inToM(child.attrib['CoreOffset']))

                    elif grainType == '5': # C grain specific properties
                        motor.grains[-1].setProperty('slotWidth', inToM(child.attrib['SlotWidth']))
                        radius = motor.grains[-1].getProperty('diameter') / 2
                        motor.grains[-1].setProperty('slotOffset', radius - inToM(child.attrib['SlotDepth']))

                    elif grainType == '6': # X core specific properties
                        motor.grains[-1].setProperty('slotWidth', inToM(child.attrib['SlotWidth']))
                        motor.grains[-1].setProperty('slotLength', inToM(child.attrib['CoreDiameter']) / 2)

                    elif grainType == '7': # Finocyl specific properties
                        motor.grains[-1].setProperty('finWidth', inToM(child.attrib['FinWidth']))
                        motor.grains[-1].setProperty('finLength', inToM(child.attrib['FinLength']))
                        motor.grains[-1].setProperty('numFins', int(child.attrib['FinCount']))

                    if not propSet: # Use propellant numbers from the forward grain
                        impProp = child.find('Propellant')
                        propellant = motorlib.propellant.Propellant()
                        propellant.setProperty('name', impProp.attrib['Name'])
                        ballN = float(impProp.attrib['BallisticN'])
                        ballA = float(impProp.attrib['BallisticA']) * 1/(6895**ballN)
                        propellant.setProperty('n', ballN)
                        # Conversion only does in/s to m/s, the rest is handled above
                        ballA = motorlib.units.convert(ballA, 'in/(s*psi^n)', 'm/(s*Pa^n)')
                        propellant.setProperty('a', ballA)
                        density = motorlib.units.convert(float(impProp.attrib['Density']), 'lb/in^3', 'kg/m^3')
                        propellant.setProperty('density', density)
                        propellant.setProperty('k', float(impProp.attrib['SpecificHeatRatio']))
                        impMolarMass = impProp.attrib['MolarMass']
                        # If the user has entered 0, override it to match the default propellant.
                        if impMolarMass == '0':
                            propellant.setProperty('m', 23.67)
                        else:
                            propellant.setProperty('m', float(impMolarMass))
                        # Burnsim doesn't provide this property. Set it to match the default propellant.
                        propellant.setProperty('t', 3500)
                        motor.propellant = propellant
                        propSet = True

                else:
                    if child.attrib['Type'] in UNSUPPORTED_GRAINS:
                        errors += "File contains a "
                        errors += UNSUPPORTED_GRAINS[child.attrib['Type']]
                        errors += " grain, which can't be imported.\n"
                    else:
                        errors += "File contains an unknown grain of type " + child.attrib['Type'] + '.\n'

            if child.tag == 'TestData':
                errors += "\nFile contains test data, which is not imported."

        if errors != '':
            QApplication.instance().outputMessage(errors + '\nThe rest of the motor will be imported.')

        self.fileManager.startFromMotor(motor)
        return True

    def exportFile(self, path, motor):
        """Takes a path to a bsx file and motor object and dumps the BSX version of the motor to the file"""
        errors = ''

        outMotor = ET.Element('Motor')
        outMotor.attrib = bsxMotorAttrib

        outNozzle = ET.SubElement(outMotor, 'Nozzle')
        outNozzle.attrib['ThroatDia'] = mToIn(motor.nozzle.getProperty('throat'))
        outNozzle.attrib['ExitDia'] = mToIn(motor.nozzle.getProperty('exit'))
        outNozzle.attrib['NozzleEfficiency'] = str(int(motor.nozzle.getProperty('efficiency') * 100))
        outNozzle.attrib['AmbientPressure'] = '14.7'

        for gid, grain in enumerate(motor.grains):
            if type(grain) in EXPORT_TYPES:
                outGrain = ET.SubElement(outMotor, 'Grain')
                outGrain.attrib['Type'] = EXPORT_TYPES[type(grain)]
                outGrain.attrib['Propellant'] = motor.propellant.getProperty('name')
                outGrain.attrib['Diameter'] = mToIn(grain.getProperty('diameter'))
                outGrain.attrib['Length'] = mToIn(grain.getProperty('length'))

                if isinstance(grain, motorlib.grains.EndBurningGrain):
                    outGrain.attrib['CoreDiameter'] = '0'
                    outGrain.attrib['EndsInhibited'] = '1'
                else:
                    ends = grain.getProperty('inhibitedEnds')
                    if ends == 'Neither':
                        outGrain.attrib['EndsInhibited'] = '0'
                    elif ends in ('Top', 'Bottom'):
                        outGrain.attrib['EndsInhibited'] = '1'
                    else:
                        outGrain.attrib['EndsInhibited'] = '2'
                    # Grains with core diameter
                    if type(grain) in (motorlib.grains.BatesGrain, motorlib.grains.Finocyl, motorlib.grains.MoonBurner):
                        outGrain.attrib['CoreDiameter'] = mToIn(grain.getProperty('coreDiameter'))

                    if isinstance(grain, motorlib.grains.DGrain):
                        outGrain.attrib['EdgeOffset'] = mToIn(grain.getProperty('slotOffset'))

                    elif isinstance(grain, motorlib.grains.MoonBurner):
                        outGrain.attrib['CoreOffset'] = mToIn(grain.getProperty('coreOffset'))

                    elif isinstance(grain, motorlib.grains.CGrain):
                        outGrain.attrib['SlotWidth'] = mToIn(grain.getProperty('slotWidth'))
                        radius = motor.grains[-1].getProperty('diameter') / 2
                        outGrain.attrib['SlotDepth'] = mToIn(grain.getProperty('slotOffset') - radius)

                    elif isinstance(grain, motorlib.grains.XCore):
                        outGrain.attrib['SlotWidth'] = mToIn(grain.getProperty('slotWidth'))
                        outGrain.attrib['CoreDiameter'] = mToIn(2 * grain.getProperty('slotLength'))

                    elif isinstance(grain, motorlib.grains.Finocyl):
                        outGrain.attrib['FinCount'] = str(grain.getProperty('numFins'))
                        outGrain.attrib['FinLength'] = mToIn(grain.getProperty('finLength'))
                        outGrain.attrib['FinWidth'] = mToIn(grain.getProperty('finWidth'))

                outProp = ET.SubElement(outGrain, 'Propellant')
                outProp.attrib['Name'] = motor.propellant.getProperty('name')
                ballA = motor.propellant.getProperty('a')
                ballN = motor.propellant.getProperty('n')
                ballA = motorlib.units.convert(ballA * (6895**ballN), 'm/(s*Pa^n)', 'in/(s*psi^n)')
                outProp.attrib['BallisticA'] = str(ballA)
                outProp.attrib['BallisticN'] = str(ballN)
                density = str(motorlib.units.convert(motor.propellant.getProperty('density'), 'kg/m^3', 'lb/in^3'))
                outProp.attrib['Density'] = density
                outProp.attrib['SpecificHeatRatio'] = str(motor.propellant.getProperty('k'))
                outProp.attrib['MolarMass'] = str(motor.propellant.getProperty('m'))
                outProp.attrib['CombustionTemp'] = '0' # Unclear if this is used anyway
                ispStar = motor.propellant.getCStar() / 9.80665
                outProp.attrib['ISPStar'] = str(ispStar)
                # Add empty notes section
                ET.SubElement(outProp, 'Notes')

            else:
                errors += "Can't export grain #" + str(gid + 1) + " because it has type " + grain.geomName + ".\n"
        # Add empty notes section
        ET.SubElement(outMotor, 'MotorNotes')

        if errors != '':
            QApplication.instance().outputMessage(errors + '\nThe rest of the motor will be exported.')

        with open(path, 'wb') as outFile:
            outFile.write(ET.tostring(outMotor))
