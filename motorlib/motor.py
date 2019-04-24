from . import grain
from . import grainTypes
from . import nozzle
from . import propellant
from . import geometry
from . import units
from . import simulationResult, simAlert, simAlertLevel, simAlertType
from . import endBurningGrain

import math
import numpy as np

class motor():
    def __init__(self):
        self.grains = []
        self.propellant = propellant()
        self.nozzle = nozzle.nozzle()

    def getDict(self):
        motorData = {}
        motorData['nozzle'] = self.nozzle.getProperties()
        motorData['propellant'] = self.propellant.getProperties()
        motorData['grains'] = [{'type': grain.geomName, 'properties': grain.getProperties()} for grain in self.grains]
        return motorData

    def loadDict(self, dictionary):
        self.nozzle.setProperties(dictionary['nozzle'])
        self.propellant.setProperties(dictionary['propellant'])
        self.grains = []
        for entry in dictionary['grains']:
            self.grains.append(grainTypes[entry['type']]())
            self.grains[-1].setProperties(entry['properties'])

    def calcKN(self, r, burnoutWebThres = 0.00001):
        surfArea = sum([gr.getSurfaceAreaAtRegression(reg) * int(gr.isWebLeft(reg, burnoutWebThres)) for gr, reg in zip(self.grains, r)])
        nozz = self.nozzle.getThroatArea()
        return surfArea / nozz

    def calcIdealPressure(self, r, kn = None, burnoutWebThres = 0.00001):
        k = self.propellant.getProperty('k')
        t = self.propellant.getProperty('t')
        m = self.propellant.getProperty('m')
        p = self.propellant.getProperty('density')
        a = self.propellant.getProperty('a')
        n = self.propellant.getProperty('n')
        if kn is None:
            kn = self.calcKN(r, burnoutWebThres)
        num = kn * p * a
        exponent = 1/(1 - n)
        denom = ((k/((8314/m)*t))*((2/(k+1))**((k+1)/(k-1))))**0.5
        return (num/denom) ** exponent

    def calcForce(self, r, casePressure = None, ambientPressure = 101325, burnoutWebThres = 0.00001):
        k = self.propellant.getProperty('k')
        t_a = self.nozzle.getThroatArea()
        e_a = self.nozzle.getExitArea()

        p_a = ambientPressure
        if casePressure is None:
            p_c = self.calcIdealPressure(r, None, burnoutWebThres)
        else:
            p_c = casePressure

        if p_c == 0:
            return 0

        p_e = self.nozzle.getExitPressure(k, p_c)

        t1 = (2*(k**2))/(k-1)
        t2 = (2/(k+1))**((k+1)/(k-1))
        t3 = 1 - ((p_e/p_c) ** ((k-1)/k))

        sr = (t1 * t2 * t3) ** 0.5

        f = self.nozzle.props['efficiency'].getValue() * t_a * p_c * sr + (p_e - p_a) * e_a
        if np.isnan(f):
            f = 0

        return f

    def calcErosiveFraction(self, G, r_0, grain):
        """Calculate erosive burn rate fraction for some dx of grain
        Based on modified Mukunda and Paul model
        Erosive fraction is defined r/r_0,
        therefore total burn rate including erosive fraction is r * r_0
        """
        G_in_imperial = units.convert(G, 'kg/(m^2*s)', 'lb/(in^2*s)')
        rho = self.propellant.getProperty('density')
        mu = 1e-4 # self.propellant.getProperty('mu')
        d_0 = grain.getPortHydraulicDiameter(r_0)
        Re_0 = rho * r_0 * d_0 / mu # Reynolds' Number.
        g = (G / (rho * r_0)) * (Re_0 * 1000) ** -0.125 # mass flux ratio modified for size effects
        g_th = 35.0 # mass flux threshold: if g is below this value, no erosive effects are considered
        return 1.0 + 0.023*(g ** 0.8 - g_th ** 0.8) * np.heaviside(g - g_th, 0)

    def calcSteadyStateBurnRate(self, simRes):
        # r = aP**n
        return self.propellant.getProperty('a') * (simRes.channels['pressure'].getLast() ** self.propellant.getProperty('n'))

    def runSimulation(self, preferences = None, callback = None):
        if preferences is not None:
            ambientPressure = preferences.general.getProperty('ambPressure')
            burnoutWebThres = preferences.general.getProperty('burnoutWebThres')
            burnoutThrustThres = preferences.general.getProperty('burnoutThrustThres')
            ts = preferences.general.getProperty('timestep')
            erosive = preferences.general.getProperty('erosive')
            erosive_dx = preferences.general.getProperty('erosive_dx')

        else:
            ambientPressure = 101325
            burnoutWebThres = 0.00001
            burnoutThrustThres = 0.1
            ts = 0.01
            erosive = False
            erosive_dx = 0.001

        simRes = simulationResult(self)

        # Checkdx for geometry errors
        if len(self.grains) == 0:
            simRes.addAlert(simAlert(simAlertLevel.ERROR, simAlertType.CONSTRAINT, 'Motor must have at least one propellant grain.', 'Motor'))
        for gid, grain in enumerate(self.grains):
            if type(grain) is endBurningGrain and gid != 0: # Endburners have to be at the foward end
                simRes.addAlert(simAlert(simAlertLevel.ERROR, simAlertType.CONSTRAINT, 'End burning grains must be the forward-most grain in the motor.', 'Grain ' + str(gid + 1)))
            for alert in grain.getGeometryErrors():
                alert.location = 'Grain ' + str(gid + 1)
                simRes.addAlert(alert)
        for alert in self.nozzle.getGeometryErrors():
            simRes.addAlert(alert)

        # If any geometry errors occurred, stop simulation and return an empty sim with errors
        if len(simRes.getAlertsByLevel(simAlertLevel.ERROR)) > 0:
            return simRes

        # Generate coremaps for perforated grains
        for grain in self.grains:
            grain.simulationSetup(preferences)

        # Setup initial values
        perGrainReg = [0 for _ in self.grains]

        # At t=0, the motor hasn't yet ignited
        simRes.channels['time'].addData(0)
        simRes.channels['kn'].addData(0)
        simRes.channels['pressure'].addData(0)
        simRes.channels['force'].addData(0)
        simRes.channels['mass'].addData([grain.getVolumeAtRegression(0) * self.propellant.getProperty('density') for grain in self.grains])
        simRes.channels['massFlow'].addData([0 for grain in self.grains])
        simRes.channels['massFlux'].addData([0 for grain in self.grains])

        # At t = ts, the motor has ignited
        simRes.channels['time'].addData(ts)
        simRes.channels['kn'].addData(self.calcKN(perGrainReg, burnoutWebThres))
        simRes.channels['pressure'].addData(self.calcIdealPressure(perGrainReg, None, burnoutWebThres))
        simRes.channels['force'].addData(self.calcForce(perGrainReg, None, ambientPressure, burnoutWebThres))
        simRes.channels['mass'].addData([grain.getVolumeAtRegression(0) * self.propellant.getProperty('density') for grain in self.grains])
        simRes.channels['massFlow'].addData([0 for grain in self.grains])
        simRes.channels['massFlux'].addData([0 for grain in self.grains])

        # Check port/throat ratio and add a warning if it is large enough
        aftPort = self.grains[-1].getPortArea(0)
        if aftPort is not None:
            minAllowed = 2 # TODO: Make the threshold configurable
            ratio = aftPort / geometry.circleArea(self.nozzle.props['throat'].getValue())
            if ratio < minAllowed:
                desc = 'Initial port/throat ratio of ' + str(round(ratio, 3)) + ' was less than ' + str(minAllowed)
                simRes.addAlert(simAlert(simAlertLevel.WARNING, simAlertType.CONSTRAINT, desc, 'N/A'))

        # Perform timesteps
        while simRes.channels['force'].getLast() > burnoutThrustThres * 0.01 * simRes.channels['force'].getMax(): # 0.01 to convert to a percentage
            mf = 0
            perGrainMass = [0 for _ in self.grains]
            perGrainMassFlow = [0 for _ in self.grains]
            perGrainMassFlux = [0 for _ in self.grains]

            prev_rates = {} # erosive burn rate of previous time step, per dx

            for gid, grain in enumerate(self.grains):
                if grain.getWebLeft(perGrainReg[gid]) > burnoutWebThres:
                    r_0 = ts * self.calcSteadyStateBurnRate(simRes) # steady state regression for this time step

                    if erosive:
                        len_ = grain.getProperty('length')
                        # per dx regressions: one total regression for each dx
                        perDxRegressions = [r_0 for _ in np.linspace(0, len_, len_/erosive_dx + 1)]
                        # mass flux for previous dx
                        perDxMassFlux = [0 for _ in np.linspace(0, len_, len_/erosive_dx + 1)]

                        # in order to calculate the total mass flux and regression for the grain, we split it into
                        # sections and iterate, stepping down by each dx every time to reach
                        # a new "section".
                        # note that this is very slow!
                        for idx, dx in enumerate(np.linspace(0, len_, len_/erosive_dx + 1)):

                            # get total regression for this dx
                            total_reg = perDxRegressions[idx]
                            # get previous burn rate for this dx
                            prev_rate = prev_rates.get(idx, r_0)

                            perDxMassFlux[idx] = grain.getMassFlux(mf, ts, total_reg, prev_rate, dx,
                                                                   self.propellant.getProperty('density'), erosive_dx)

                            # update mass flow
                            mf += grain.getFaceArea(total_reg) * erosive_dx * self.propellant.getProperty('density')

                            # update regression with the erosive rate
                            n_e = self.calcErosiveFraction(perDxMassFlux[idx], r_0, grain)  # erosive burn fraction
                            r_e = r_0 * n_e  # r = r_0 * n_e
                            perDxRegressions[idx] += r_e
                            prev_rates[idx] = r_e

                        massflux = perDxMassFlux[-1]

                    else:
                        # Find the mass flux through the grain based on the mass flow fed into from grains above it
                        massflux = grain.getPeakMassFlux(mf, ts, perGrainReg[gid], r_0,
                                                         self.propellant.getProperty('density'))

                    perGrainMassFlux[gid] = massflux
                    perGrainMass[gid] = grain.getVolumeAtRegression(perGrainReg[gid]) * self.propellant.getProperty('density') # Find the mass of the grain after regression
                    mf += (simRes.channels['mass'].getLast()[gid] - perGrainMass[gid]) / ts # Add the change in grain mass to the mass flow

                    perGrainReg[gid] += r_0 # Apply the regression

                perGrainMassFlow[gid] = mf

            # regression finished, add data points;
            simRes.channels['mass'].addData(perGrainMass)
            simRes.channels['massFlow'].addData(perGrainMassFlow)
            simRes.channels['massFlux'].addData(perGrainMassFlux)

            # Calculate KN
            simRes.channels['kn'].addData(self.calcKN(perGrainReg, burnoutWebThres))

            # Calculate Pressure
            simRes.channels['pressure'].addData(self.calcIdealPressure(perGrainReg, simRes.channels['kn'].getLast(), burnoutWebThres))

            # Calculate force
            simRes.channels['force'].addData(self.calcForce(perGrainReg, simRes.channels['pressure'].getLast(), ambientPressure, burnoutWebThres))

            simRes.channels['time'].addData(simRes.channels['time'].getLast() + ts)

            if callback is not None:
                progress = max([g.getWebLeft(r) / g.getWebLeft(0) for g,r in zip(self.grains, perGrainReg)]) # Grain with the largest percentage of its web left
                if callback(1 - progress): # If the callback returns true, it is time to cancel
                    return simRes

        simRes.channels['time'].addData(simRes.channels['time'].getLast() + ts)
        simRes.channels['kn'].addData(0)
        simRes.channels['pressure'].addData(0)
        simRes.channels['force'].addData(0)
        simRes.channels['mass'].addData([grain.getVolumeAtRegression(0) * self.propellant.getProperty('density') for grain in self.grains])
        simRes.channels['massFlow'].addData([0 for grain in self.grains])
        simRes.channels['massFlux'].addData([0 for grain in self.grains])

        simRes.success = True

        return simRes
