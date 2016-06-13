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
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.import numpy as np

import numpy as np

from JPA import JPA
from LJPA import LJPA

class Klopfenstein_discretization(JPA):



    def __init__(self, f_p, C, L_s, I_c, phi_s, phi_dc, phi_ac, theta_p, theta_s):

        JPA.__init__(self, I_c, phi_s, phi_dc, phi_ac, theta_p, theta_s)

        self.C   = C
        self.L_s = L_s
        self.f_p = f_p



    def reflection_discretization(self, f, z_ext, ll, cl, n, l, zl, as_theory):
        """
        Return the reflection of the Klopfenstein taper.
        To do so, the taper impedance is discretised in n sections.
        Each of these sections is modeled as a lossless transmission line.
        At the end the LJPA is modeled as a lumped element to ground and very
        high impedance (1e99 ohm) to the line.

        Parameters
        ----------
        f : float
            Signal frequency in GHz.
        z_ext : float
            Impedance seen by the pumpistor at the idler frequency
        ll : np.ndarray
            Inductance per unit length along the taper length in henry per meter
        cl : np.ndarray
            Capacitance per unit length along the taper length in farad per meter
        n : float
            Number of discret elements used to model the taper line.
        l : float
            Length of the taper in meter
        zl : float
            Load impedance
        as_theory : bool
            If true use the load impedance of the characteristic impeance
            calculation to try to mimic the theoritical reflection.
            Use this parameter to test if this method can correctly mimic
            the theoritical expectation.
        """

        o = 2.*np.pi*f

        # Calculate the chain of matrices representing the Klopfenstein tapper
        a = o*np.sqrt(ll*cl)*l/(n - 1.)
        m = np.array([np.cos(a), 1j*np.sqrt(ll/cl)*np.sin(a), 1j/np.sqrt(ll/cl)*np.sin(a), np.cos(a)])

        # Give the good shape to the array for the matrix product
        m = m.transpose()
        m.shape = (n, 2, 2)

        # Calculate the chain matrix prodcut
        M = reduce(np.dot, m)

        # Calculate the LJPA impedance
        z_ljpa = 1./(1j*self.C*o + 1./(1j*o*(self.L_s + self.squid_inductance(f, z_ext))))

        # We end the chain by two elements:
        # 1 - a load impedance to the ground
        # 2 - a huge impedance to the circuit
        if as_theory:
            M = np.dot(M, np.array([[1., 0.],[1./zl, 1.]]))
        else:
            M = np.dot(M, np.array([[1., 0.],[1./z_ljpa, 1.]]))

        M = np.dot(M, np.array([[1., 1e99],[0., 1.]]))

        # Compute the reflection from the array elements
        a = M.item(0)
        b = M.item(1)
        c = M.item(2)
        d = M.item(3)

        return (a + b/50. - c*50. - d)/(a + b/50. + c*50. + d)



    def external_discretization(self, f, ll, cl, n, l, zl, as_theory):
        """
        Return the impedance of the electrical environment seen by the SQUID.
        We assume the circuit to be 50 ohm matched.

        Parameters
        ----------
        f : float
            Signal frequency in hertz.
        n : float, optional
            Number of discret elements used to model the taper line.
        R0 : float, optional
            The characteristic impedance of the incoming line. Assumed to be
            50 ohm.
        as_theory : bool, optional
            If true use the load impedance of the characteristic impeance
            calculation to try to mimic the theoritical reflection.
            Use this parameter to test if this method can correctly mimic
            the theoritical expectation.
        """

        # The idler angular frequency
        o = 2.*np.pi*(self.f_p - f)

        # Calculate the chain of matrices representing the Klopfenstein tapper
        a = o*np.sqrt(ll*cl)*l/(n - 1.)
        m = np.array([np.cos(a), 1j*np.sqrt(ll/cl)*np.sin(a), 1j/np.sqrt(ll/cl)*np.sin(a), np.cos(a)])

        # Give the good shape to the array for the matrix product
        m = m.transpose()
        m.shape = (n, 2, 2)

        # Calculate the chain matrix prodcut
        M = reduce(np.dot, m)

        # We end the chain by two elements:
        # 1 - a load impedance to the ground
        # 2 - a huge impedance to the circuit
        M = np.dot(M, np.array([[1., 0.],[1./50., 1.]]))
        M = np.dot(M, np.array([[1., 1e99],[0., 1.]]))

        # We start the chain by two elements:
        # 1 - The stray inductance
        # 2 - The resonator capacitance to ground
        M = np.dot(np.array([[1., 1j*o*self.L_s], [0., 1.]]), M)
        M = np.dot(np.array([[1., 0.], [1j*o*self.C, 1.]]), M)

        # Return the z11 impedance element
        return M.item(0)/M.item(2)



def reflection_discretization(param):

    f, z_ext, ll, cl, n, l, zl, C, L_s, I_c, phi_s, phi_dc, phi_ac, theta_p, theta_s, as_theory, f_p = param

    a = Klopfenstein_discretization(f_p, C, L_s, I_c, phi_s, phi_dc, phi_ac, theta_p, theta_s)

    return a.reflection_discretization(f, z_ext, ll, cl, n, l, zl, as_theory)



def external_discretization(param):

    f, ll, cl, n, l, zl, C, L_s, I_c, phi_s, phi_dc, phi_ac, theta_p, theta_s, as_theory, f_p = param

    a = Klopfenstein_discretization(f_p, C, L_s, I_c, phi_s, phi_dc, phi_ac, theta_p, theta_s)

    return a.external_discretization(f, ll, cl, n, l, zl, as_theory)
