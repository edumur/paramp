# This Python file uses the following encoding: utf-8

# Copyright (C) 2016 Dumur Étienne
# etienne.dumur@gmail.com

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
import scipy.constants as cst
from scipy.optimize import minimize_scalar

class Find(object):



    def find_max_gain(self, scale='log', R0=50.):
        """
        Return the maximum power gain.
        Numerically estimated.

        Parameters
        ----------
        scale: {log, linear}, optional
            The power reflection will be returned in log or linear scale.

        Raises
        ------
        ValueError
            If the parameters are not in the good type.
        """

        if scale not in ('log', 'linear'):
            raise ValueError("Parameter 'scale' must be 'log' or 'linear'")


        y = abs(self.reflection(self.find_resonance_frequency(R0)))**2.

        if scale.lower() == 'log':
            return 10.*np.log10(y)
        elif scale.lower() == 'linear':
            return y



    def find_reflection_fwhm(self, R0=50.):
        """
        Return the half width at half maximum in Hz of the power reflection.
        Numerically estimated.
        """

        def func(f, half_max, R0):
            return abs(-abs(self.reflection(f, R0))**2. + half_max)

        f0 = self.find_resonance_frequency(R0)
        half_max = (  abs(self.reflection(f0, R0))**2.\
                    + abs(self.reflection(f0+100e9, R0))**2.)/2.
        df = minimize_scalar(func, bounds=(1., 100e9),
                                   method='bounded',
                                   args=(half_max, R0)).x

        return abs(f0 - df)*2.



    def find_angular_resonance_frequency(self, R0=50.):
        """
        Return the angular resonance frequency in rad.Hz of the power reflection.
        Numerically estimated.
        """

        return self.find_resonance_frequency(R0)*2.*np.pi



    def find_resonance_frequency(self, R0=50.):
        """
        Return the resonance frequency in Hz of the power reflection.
        Numerically estimated.
        """

        def func(f, R0):
            return -abs(self.reflection(f, R0))**2.

        return minimize_scalar(func,
		                       args=(R0,),
							   bounds=(1., 100e9),
							   method='bounded').x



    def find_1db_deviation_power(self, f, R0=50., unit='dBm'):
        """
        Find the lowest power at which the gain changes of at least one dB.
        Please note that this change can be +/- 1dB.
        Return the time average power, not the RMS power.

        Parameters
        ----------
        f : float, np.ndarray
            The frequency in hertz.
        R0 : float, optional
            The characteristic impedance of the incoming line. Assumed to be
            losses line so real and 50 ohm.
        unit : {'dBm', 'rad'} string, optional
            Unit in which the result is returned
        """

        # First we backup the initial phi_s value
        backup_phi_s = self.phi_s

        # We look for the reflection power at very low input power
        self.phi_s = 0.00001
        reflection_low_power =  20.*np.log10(abs(self.reflection(f)))

        # We look for the maximum possible reflection power
        def func1(phi, f):
            self.phi_s = phi
            return  -abs(self.reflection(f))

        reflection_optimum_power = minimize_scalar(func1,
                                                   args=(f,),
                                                   bounds=(0.00001, 5.),
                                                   method='bounded').x

        self.phi_s = reflection_optimum_power
        reflection_optimum_power =  20.*np.log10(abs(self.reflection(f)))

        # If the maximum power is greater than the reflection at low power
        # the 1db compression point should be looked in a certain range of phi_s
        # +/- 1 because we looked at the 1dB deviation
        if reflection_optimum_power > reflection_low_power + 0.99:
            max_bound = self.phi_s
            condition = reflection_low_power + 1.
        else:
            max_bound = 5.
            condition = reflection_low_power - 1.

        # Looking for the 1dB deviation point
        def func2(phi, f, condition):
            self.phi_s = phi
            return  abs(20.*np.log10(abs(self.reflection(f))) - condition)**2.


        result =  minimize_scalar(func2,
        	                      args=(f, condition),
        						  bounds=(0.00001, max_bound),
        						  method='bounded').x

        # Setting back the backup input power
        self.phi_s = backup_phi_s

        if unit.lower() == 'dbm':
            # From rad to watt
            result = (cst.hbar/2./cst.e*result*2.*np.pi*f)**2./R0/2.
            # Return dBm
            return 10.*np.log10(result/1e-3)
        elif unit.lower() == 'rad':
            return result
        else:
            raise ValueError("'unit' parameter must be 'dbm' or 'rad'.")
