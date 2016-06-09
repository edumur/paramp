# This Python file uses the following encoding: utf-8

# Implementation of the Pumpistor model of a flux pumped SQUID in the
# three wave mixing degenerate case ω_p = ω_s + ω_i.
#
# Based on an article of K. M. Sundqvist et al:
# "The pumpistor: A linearized model of a flux-pumped superconducting
# quantum interference device for use as a negative-resistance parametric
# amplifier"
# APL 109 102603 (2013)

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
from scipy.special import jv

class JPA(object):



    def __init__(self, I_c, phi_s, phi_dc, phi_ac, theta_p,
                 theta_s=0., f_p=None):
        """
        Implementation of the Pumpistor model of a flux pumped SQUID in the
        three wave mixing case ω_p = ω_s + ω_i.

        Based on an article of K. M. Sundqvist et al:
        "The pumpistor: A linearized model of a flux-pumped superconducting
        quantum interference device for use as a negative-resistance parametric
        amplifier"
        APL 109 102603 (2013)

        ...

        Attributes
        ----------
        I_c : float
            Critical current of the SQUID in ampere.
        phi_s : float
            Amplitude of the signal in rad.
        phi_dc : float
            DC amplitude of the pump in Φ0 unit.
        phi_ac : float
            AC amplitude of the pump in Φ0 unit.
        theta_p : float
            Phase of the pump in rad.
        theta_s : float, optional
            Phase of the pump in rad, default is zero which implies that that\
            the signal phase is the reference.
        f_p : float, optional
            Pump frequency. If None we assume  f_p = 2*f_s.

        Raises
        ------
        ValueError
            If the parameters are not in the good type.
        """

        if type(I_c) is not float:
            raise ValueError('I_c parameter must be float type.')
        if type(phi_s) is not float:
            raise ValueError('phi_s parameter must be float type')
        if type(phi_dc) is not float:
            raise ValueError('phi_dc parameter must be float type.')
        if type(phi_ac) is not float:
            raise ValueError('phi_ac parameter must be float type.')
        if type(theta_p) is not float:
            raise ValueError('theta_p parameter must be float type.')
        if type(theta_s) is not float:
            raise ValueError('theta_s parameter must be float type.')

        self.I_c     = I_c
        self.phi_s   = phi_s
        self.phi_dc  = phi_dc
        self.phi_ac  = phi_ac
        self.theta_p = theta_p
        self.theta_s = theta_s
        self.f_p = f_p



    def F(self):
        """
        Return the dimensionless DC flux amplitude.
        """

        return np.pi*self.phi_dc



    def delta_f(self):
        """
        Return the dimensionless AC flux amplitude.
        """

        return np.pi*self.phi_ac



    def delta_theta(self):
        """
        Return the phase difference between the signal and the pump as
        2θ_s - θ_p.
        If θ_s has not been specified, it is 0 by default.
        """

        return 2.*self.theta_s - self.theta_p



    def josephson_inductance(self):
        """
        Return the Josephson inductance in henry.
        """

        return cst.hbar*self.phi_s\
               /2./cst.e/self.I_c/abs(np.cos(self.F()))/2./jv(1., self.phi_s)



    def pumpistor_inductance(self, f=None):
        """
        Return the pumpistor inductance.
        In the case of the non-degenerate case, a parent class must provide
        a external_impedance method returning the impedance of the electrical
        environment seen by the SQUID.

        Parameters
        ----------
        f : float, np.ndarray, optional
            Signal frequency in hertz.
            Is required in the non-degenerate case but optional for the
            degenerate one.
        """

        # If f_p is None, we return the pumpistor inductance of the
        # degenerate case.
        if self.f_p is None:

            return -2.*np.exp(1j*self.delta_theta())/self.delta_f()\
                   *cst.hbar/2./cst.e/self.I_c/abs(np.sin(self.F()))\
                   *self.phi_s/(2.*jv(1., self.phi_s)\
                                - 2.*np.exp(2j*self.delta_theta())*jv(3., self.phi_s))
        else:

            if f is None:
                raise ValueError('In the non-degenerate case, the pumpistor needs the signal frequency.')

            return cst.h/2./cst.e/np.pi/self.I_c/np.sin(self.F())**2./self.delta_f()**2.\
                   *(- 2.*np.cos(self.F())\
                     + 1j*cst.h/2./cst.e/np.pi/self.I_c\
                         *2.*np.pi*(self.f_p - f)\
                         *self.external_impedance(self.f_p - f).conjugate())



    def squid_inductance(self, f=None):
        """
        Return the squid inductance which is simply the parallel sum of the
        pumpistor and the Josephson indutance.
        """

        return 1./(  1./self.josephson_inductance()\
                   + 1./self.pumpistor_inductance(f))



    def pumpistor_impedance(self, f):
        """
        Return the pumpistor impedance.

        Parameters
        ----------
        f : float, np.ndarray
            The frequency in hertz.

        Raises
        ------
        ValueError
            If the parameters are not in the good type.
        """

        if type(f) not in (float, np.ndarray):
            raise ValueError('f parameter must be float or np.ndarray type.')

        return 1j*f*2.*np.pi*self.pumpistor_inductance(f)



    def josephson_impedance(self, f):
        """
        Return the Josephson impedance.

        Parameters
        ----------
        f : float, np.ndarray
            The frequency in hertz.

        Raises
        ------
        ValueError
            If the parameters are not in the good type.
        """

        if type(f) not in (float, np.ndarray):
            raise ValueError('f parameter must be float or np.ndarray type.')

        return 1j*f*2.*np.pi*self.josephson_inductance()



    def squid_impedance(self, f):
        """
        Return the squid impedance.

        Parameters
        ----------
        f : float, np.ndarray
            The frequency in hertz.

        Raises
        ------
        ValueError
            If the parameters are not in the good type
        """

        if type(f) not in (float, np.ndarray):
            raise ValueError('f parameter must be float or np.ndarray type')

        return 1j*f*2.*np.pi*self.squid_inductance(f)



    def squid_reflection(self, f, z0=50.):
        """
        Return the reflection of the SQUID.

        Parameters
        ----------
        f : float, np.ndarray
            The frequency in hertz.
        z0 : float, optional
            The characteristic impedance of the transmission line connected to
            the SQUID, default is 50. Ω.

        Raises
        ------
        ValueError
            If the parameters are not in the good type
        """

        if type(f) not in (float, np.ndarray):
            raise ValueError('f parameter must be float or np.ndarray type')

        return  (self.squid_impedance(f) - z0)\
               /(self.squid_impedance(f) + z0)
