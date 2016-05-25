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
from scipy.optimize import minimize_scalar

class Find(object):



    def find_max_gain(self, scale='log'):
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


        y = abs(self.reflection(self.find_resonance_frequency()))**2.

        if scale.lower() == 'log':
            return 10.*np.log10(y)
        elif scale.lower() == 'linear':
            return y



    def find_reflection_fwhm(self):
        """
        Return the half width at half maximum in Hz of the power reflection.
        Numerically estimated.
        """

        def func(f,half_max):
            return abs(-abs(self.reflection(f))**2. + half_max)

        f0 = self.find_resonance_frequency()
        half_max = (  abs(self.reflection(f0))**2.\
                    + abs(self.reflection(f0+100e9))**2.)/2.
        df = minimize_scalar(func, bounds=(1., 100e9),
                                   method='bounded',
                                   args=[half_max]).x

        return abs(f0 - df)*2.



    def find_angular_resonance_frequency(self):
        """
        Return the angular resonance frequency in rad.Hz of the power reflection.
        Numerically estimated.
        """

        return self.find_resonance_frequency()*2.*np.pi



    def find_resonance_frequency(self):
        """
        Return the resonance frequency in Hz of the power reflection.
        Numerically estimated.
        """

        def func(f):
            return -abs(self.reflection(f))**2.

        return minimize_scalar(func, bounds=(1., 100e9), method='bounded').x
