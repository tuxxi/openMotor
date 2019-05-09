from . import grainTypes
from . import nozzle
from . import propellant
from . import geometry
from . import simulationResult, simAlert, simAlertLevel, simAlertType
from . import endBurningGrain

import numpy as np
from copy import copy

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

    def calcKn(self, r, burnoutWebThres=0.00001):
        surfArea = np.sum((gr.getSurfaceAreaAtRegression(reg) for gr, reg in zip(self.grains, r)
                           if gr.isWebLeft(reg, burnoutWebThres)))
        nozzArea = self.nozzle.getThroatArea()
        return surfArea / nozzArea

    def calcKnFromSlices(self, regs, steadyReg, burnoutWebThres=0.00001, erosive_dx=0.001):
        """Calculate Kn from regressions of each grain slice, used in erosive burning simulation"""

        def getFaceAreas(gid, grain):
            """Get areas of top and bottom face of grain, based on erosive regressions"""
            topArea = grain.getFaceArea(list(regs[gid].values())[0])
            bottomArea = grain.getFaceArea(list(regs[gid].values())[-1])

            # set areas to zero if they are inhibited
            if grain.props['inhibitedEnds'].getValue == 'Top':
                topArea = 0
            if grain.props['inhibitedEnds'].getValue == 'Bottom':
                bottomArea = 0
            elif grain.props['inhibitedEnds'].getValue == 'Both':
                topArea, bottomArea = 0, 0
            return topArea, bottomArea

        surfArea = 0
        for gid, grain in enumerate(self.grains):
            topArea, bottomArea = getFaceAreas(gid, grain)

            portArea = 0
            # dict of step lengths, used as filter for burned out sections
            dxDict = self._createDxLengthDict(grain, steadyReg, erosive_dx)

            burnedOutSlices = 0
            for idx, reg in regs[gid].items():
                if grain.isWebLeft(reg, burnoutWebThres): # does web exist for this slice?
                    portArea += grain.getCorePerimeter(reg) * dxDict[idx]
                else:
                    burnedOutSlices += 1

            if burnedOutSlices == len(regs[gid].values()):
                topArea = bottomArea = portArea = 0

            surfArea += portArea + topArea + bottomArea

        nozzArea = self.nozzle.getThroatArea()
        return surfArea / nozzArea

    @staticmethod
    def _createDxLengthDict(grain, reg, erosive_dx):
        """Creates a dict of lengths of each step along the grain in the format: {idx: len}. idx is the step
        index and len is the new step length adjusted for grain length regression: as the grain faces regress,
        the overall length decreases and invalidates some steps. This dict is used to filter the steps"""

        def stepLength(dx, idx):
            ret = erosive_dx
            if offsetFromEnd < dx < regressedLen + offsetFromEnd:
                # check if regression falls between two slices
                if idx <= burnedOut or idx >= num_points - burnedOut - 1:
                    ret = lenBurnedOut - offsetFromEnd

            else: # this section is outside of grain length boundaries, so it burned out.
                ret = 0

            return ret

        regressedLen = grain.getRegressedLength(reg)
        len_ = grain.getProperty('length')
        offsetFromEnd = (len_ - regressedLen) / 2 # we assume the regression of length is even on both faces
        num_points = len_ / erosive_dx
        burnedOut = np.ceil(offsetFromEnd / erosive_dx)  # how many slices have burned out at each end
        lenBurnedOut = burnedOut * erosive_dx

        return {idx: stepLength(dx, idx) for idx, dx in enumerate(np.linspace(0, len_, num_points + 1))}

    def calcIdealPressure(self, r, kn = None, burnoutWebThres = 0.00001, burnrate=None):
        k = self.propellant.getProperty('k')
        t = self.propellant.getProperty('t')
        m = self.propellant.getProperty('m')
        p = self.propellant.getProperty('density')
        a = self.propellant.getProperty('a')
        n = self.propellant.getProperty('n')
        if kn is None:
            kn = self.calcKn(r, burnoutWebThres)

        if burnrate is None:
            num = kn * p * a
            exponent = 1/(1 - n)
            denom = ((k/((8314/m)*t))*((2/(k+1))**((k+1)/(k-1))))**0.5
            return (num/denom) ** exponent
        else:
            num = (k * 8314 / m * t) ** 0.5
            denom = k * ((2 / (k + 1)) ** ((k + 1) / (k - 1))) ** 0.5
            cstar = num / denom

            return p * kn * burnrate * cstar

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

    # TODO: fix magnitude (seems too small?)
    def calcErosiveFraction(self, G, r_0, reg, grain):
        """Calculate erosive burn rate fraction for some dx of grain
        Based on modified Mukunda and Paul model
        Erosive fraction is defined as r/r_0, where r is total burn rate, r_0 is steady-state burn rate
        """
        rho = self.propellant.getProperty('density')
        mu = self.propellant.getProperty('mu')
        d_0 = grain.getCharacteristicLength(reg)
        Re_0 = (rho * r_0 * d_0) / mu           # Reynolds' Number.
        g_0 = G / (rho * r_0)                   # mass flux ratio
        g = g_0 * (Re_0 / 1000)**-0.125         # modified for size effects
        g_th = 35.0 # mass flux threshold: if g is below this value, no erosive effects are considered
        return 1.0 + 0.023*(g**0.8 - g_th**0.8) * np.heaviside(g-g_th, 0)

    def calcSteadyStateBurnRate(self, simRes):
        # r = aP**n
        return self.propellant.getProperty('a') * (simRes.channels['pressure'].getLast() ** self.propellant.getProperty('n'))

    def runSimulation(self, preferences = None, callback = None):
        if preferences is not None:
            ambientPressure = preferences.general.getProperty('ambPressure')
            burnoutWebThres = preferences.general.getProperty('burnoutWebThres')
            burnoutThrustThres = preferences.general.getProperty('burnoutThrustThres')
            dt = preferences.general.getProperty('timestep')
            erosive = preferences.general.getProperty('erosive')
            erosive_dx = preferences.general.getProperty('erosive_dx')

        else:
            ambientPressure = 101325
            burnoutWebThres = 0.00001
            burnoutThrustThres = 0.1
            dt = 0.01
            erosive = False
            erosive_dx = 0.001

        simRes = simulationResult(self)

        # Check for geometry errors
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
        # At t=0, the motor hasn't yet ignited
        simRes.channels['time'].addData(0)
        simRes.channels['kn'].addData(0)
        simRes.channels['pressure'].addData(0)
        simRes.channels['force'].addData(0)
        simRes.channels['mass'].addData([grain.getVolumeAtRegression(0) * self.propellant.getProperty('density')
                                         for grain in self.grains])
        simRes.channels['massFlow'].addData([0 for _ in self.grains])
        simRes.channels['massFlux'].addData([0 for _ in self.grains])

        # At t = ts, the motor has ignited
        simRes.channels['time'].addData(dt)
        simRes.channels['kn'].addData(self.calcKn([0 for _ in self.grains], burnoutWebThres))
        simRes.channels['pressure'].addData(self.calcIdealPressure([0 for _ in self.grains], None, burnoutWebThres))
        simRes.channels['force'].addData(self.calcForce([0 for _ in self.grains], None, ambientPressure, burnoutWebThres))
        simRes.channels['mass'].addData([grain.getVolumeAtRegression(0) * self.propellant.getProperty('density')
                                         for grain in self.grains])
        simRes.channels['massFlow'].addData([0 for _ in self.grains])
        simRes.channels['massFlux'].addData([0 for _ in self.grains])

        # Check port/throat ratio and add a warning if it is large enough
        aftPort = self.grains[-1].getPortArea(0)
        if aftPort is not None:
            minAllowed = 2 # TODO: Make the threshold configurable
            ratio = aftPort / geometry.circleArea(self.nozzle.props['throat'].getValue())
            if ratio < minAllowed:
                desc = 'Initial port/throat ratio of ' + str(round(ratio, 3)) + ' was less than ' + str(minAllowed)
                simRes.addAlert(simAlert(simAlertLevel.WARNING, simAlertType.CONSTRAINT, desc, 'N/A'))

        perGrainReg = [0 for _ in self.grains]  # total regression per grain: for non-erosive use
        totalSteadyStateReg = 0

        # erosive params
        prev_rates = [{} for _ in self.grains]  # erosive burn rate of previous time step: per grain, per dx.
        perDxRegressions = [{} for _ in self.grains]  # total regression: per grain, per dx

        # Perform timesteps until thrust is below thrust threshold percentage
        while simRes.channels['force'].getLast() > burnoutThrustThres * 0.01 * simRes.channels['force'].getMax():
            totalMassFlow = 0
            perGrainMass = [0 for _ in self.grains]
            perGrainMassFlow = [0 for _ in self.grains]
            perGrainMassFlux = [0 for _ in self.grains]

            r_0 = self.calcSteadyStateBurnRate(simRes)  # steady state burn rate
            reg = r_0 * dt
            totalSteadyStateReg += reg
            rho = self.propellant.getProperty('density')

            # calculate regression and other per-grain parameters
            for gid, grain in enumerate(self.grains):
                if grain.getWebLeft(perGrainReg[gid]) > burnoutWebThres: # grain has not burned out yet.

                    if erosive:
                        len_ = grain.getProperty('length')
                        num_points = len_ / erosive_dx

                        # dict of step lengths, used as filter for burned out sections
                        dxDict = self._createDxLengthDict(grain, r_0, erosive_dx)

                        nonErosiveMFs = [0 for _ in np.linspace(0, len_, num_points + 1)]
                        totalMass = 0
                        # to calculate the total mass flux and regression for the grain, we split it into
                        # sections and iterate, stepping down by length dx and calculating mass flux and regression for
                        # each section.
                        # note that this is _very_ slow!
                        for idx, dx in enumerate(np.linspace(0, len_, num_points + 1)):

                            if idx not in perDxRegressions[gid]:
                                perDxRegressions[gid][idx] = 0

                            prev_reg = perDxRegressions[gid][idx]
                            prev_rate = prev_rates[gid].get(idx, r_0)

                            if dxDict[idx] > 0: # this section has not burned out yet, so we will still calculate MF
                                nonErosiveMFs[idx] = grain.getMassFlux(totalMassFlow, dt, prev_reg, reg, dx, rho)

                                # scale burning SA to only this length dx
                                BurningSA = grain.getCorePerimeter(prev_reg) * dxDict[idx]
                                totalMassFlow += BurningSA * rho * prev_rate
                                totalMass += grain.getVolumeAtRegression(prev_reg) * dxDict[idx] * rho

                            # erosive burn fraction using non-erosive mass flux
                            n_e = self.calcErosiveFraction(nonErosiveMFs[idx], r_0, prev_reg, grain)
                            r_tot = r_0 * n_e  # r = r_0 * n_e
                            perDxRegressions[gid][idx] += r_tot * dt
                            prev_rates[gid][idx] = r_tot

                        perGrainMassFlux[gid] = np.max(nonErosiveMFs) # just record the max. mass flux
                        perGrainMass[gid] = totalMass

                        # TODO ugly hack: use the head end regression to ensure the motor burns out completely.
                        perGrainReg[gid] = list(perDxRegressions[gid].values())[0]

                    else:
                        # Find the mass flux through the grain based on the mass flow fed into from grains above it
                        perGrainMassFlux[gid] = grain.getPeakMassFlux(totalMassFlow, dt, perGrainReg[gid], reg, rho)
                        perGrainReg[gid] += reg  # Apply the regression

                        totalMassFlow += grain.getSurfaceAreaAtRegression(reg) * rho * r_0

                    perGrainMassFlow[gid] = totalMassFlow

            # add data points to simulation result
            simRes.channels['mass'].addData(perGrainMass)
            simRes.channels['massFlow'].addData(perGrainMassFlow)
            simRes.channels['massFlux'].addData(perGrainMassFlux)

            # Calculate Pressure and KN
            if erosive:
                burnrate = np.max(list(prev_rates[-1].values())) # use the max burn rate of the bottom grain
                kn = self.calcKnFromSlices(perDxRegressions, totalSteadyStateReg, burnoutWebThres, erosive_dx)
                pressure = self.calcIdealPressure(perGrainReg, kn, burnoutWebThres, burnrate)
                non_erosive_kn = self.calcKn(perGrainReg, burnoutWebThres)
            else:
                kn = self.calcKn(perGrainReg, burnoutWebThres)
                pressure = self.calcIdealPressure(perGrainReg, simRes.channels['kn'].getLast(), burnoutWebThres)
            simRes.channels['pressure'].addData(pressure)
            simRes.channels['kn'].addData(kn)

            # Calculate force
            simRes.channels['force'].addData(self.calcForce(perGrainReg, simRes.channels['pressure'].getLast(), ambientPressure, burnoutWebThres))

            simRes.channels['time'].addData(simRes.channels['time'].getLast() + dt)

            if callback is not None:
                progress = np.max([g.getWebLeft(r) / g.getWebLeft(0) for g, r in zip(self.grains, perGrainReg)]) # Grain with the largest percentage of its web left
                if callback(1 - progress): # If the callback returns true, it is time to cancel
                    return simRes

        # simulation finished
        simRes.success = True

        return simRes
