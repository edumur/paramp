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



    def matrix_chain(self, arg, z):
        """
        Return the ABCD matrix of the taper
        """

        # Calculate the chain of matrices representing the Klopfenstein tapper
        m = np.array([np.cos(arg), 1j*z*np.sin(arg), 1j/z*np.sin(arg), np.cos(arg)])

        # Give the good shape to the array for the matrix product
        m = m.transpose()
        m.shape = (len(z), 2, 2)

        # Calculate the chain matrix prodcut
        return reduce(np.dot, m)


    def reflection_discretization(self, f, z_ext, z, prod, zl, as_theory):
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
        zl : float
            Load impedance
        as_theory : bool
            If true use the load impedance of the characteristic impedance
            calculation to try to mimic the theoretical reflection.
            Use this parameter to test if this method can correctly mimic
            the theoretical expectation.
        """

        o = 2.*np.pi*f

        # Obain the taper ABCD matrix
        M = self.matrix_chain(o*prod, z)

        # Calculate the LJPA impedance
        z_ljpa = 1./(1j*self.C*o + 1./(1j*o*(self.L_s + self.squid_inductance(f, z_ext))))

        # We end the chain by two elements:
        # 1 - a load impedance to the ground
        # 2 - a huge impedance to the circuit
        if as_theory:
            M = np.dot(M, np.array([[1., 0.],[1./zl, 1.]]))
        else:
            M = np.dot(M, np.array([[1., 0.],[1./z_ljpa, 1.]]))


        # Compute the reflection from the array elements
        a = M.item(0)
        b = M.item(1)
        c = M.item(2)
        d = M.item(3)

        return (a + b/50. - c*50. - d)/(a + b/50. + c*50. + d)



    def external_discretization(self, f, z, prod, zl, as_theory, simple_ext):
        """
        Return the impedance of the electrical environment seen by the SQUID.
        We assume the circuit to be 50 ohm matched.

        Parameters
        ----------
        f : float
            Signal frequency in hertz.
        R0 : float, optional
            The characteristic impedance of the incoming line. Assumed to be
            50 ohm.
        as_theory : bool, optional
            If true use the load impedance of the characteristic impedance
            calculation to try to mimic the theoretical reflection.
            Use this parameter to test if this method can correctly mimic
            the theoretical expectation.
        simple_ext : bool, optional
            If true replace the impedance of the taper and the 50 matched
            impedance by the load impedance of the taper (zl). Should be close
            to real case and should be faster (close to twice faster).
        """

        o = 2.*np.pi*f

        if simple_ext:

            # Replace the taper and the 50ohm matched impedance by the impedance
            # of the taper end.
            z = 1j*self.L_s*o + 1./(1./zl + 1j*o*self.C)

        else:

            # Obain the taper ABCD matrix
            M = self.matrix_chain(o*prod, z)

            # We end the chain by two elements:
            # 1 - a load impedance to the ground
            M = np.dot(M, np.array([[1., 0.],[1./50., 1.]]))

            # We start the chain by two elements:
            # 1 - The stray inductance
            # 2 - The resonator capacitance to ground
            M = np.dot(np.array([[1., 0.], [1j*o*self.C, 1.]]), M)
            M = np.dot(np.array([[1., 1j*o*self.L_s], [0., 1.]]), M)

            # Return the z11 impedance element
            z = M.item(0)/M.item(2)

        return z



def reflection_discretization(param):

    f, z_ext, z, prod, zl, C, L_s, I_c, phi_s, phi_dc, phi_ac, theta_p, theta_s, as_theory, f_p = param

    a = Klopfenstein_discretization(f_p, C, L_s, I_c, phi_s, phi_dc, phi_ac, theta_p, theta_s)

    return a.reflection_discretization(f, z_ext, z, prod, zl, as_theory)



def external_discretization(param):

    f, z, prod, zl, C, L_s, I_c, phi_s, phi_dc, phi_ac, theta_p, theta_s, as_theory, simple_ext,  f_p = param

    a = Klopfenstein_discretization(f_p, C, L_s, I_c, phi_s, phi_dc, phi_ac, theta_p, theta_s)

    return a.external_discretization(f, z, prod, zl, as_theory, simple_ext)
