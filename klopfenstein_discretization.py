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

from LJPA import LJPA



def inter_func(param):
    """
    Intermediate function used because Pool function doesn't accept
    multiparameters function.
    """
    
    return matrix_product(*param)



def matrix_product(f, beta, z, n, l, zl,
                    C, L_s, I_c, phi_s, phi_dc, phi_ac, theta_p, theta_s,
                    as_theory):

    # Create the LJPA instance needed for the LJPA impedance calculation
    ljpa = LJPA(C, L_s, I_c, phi_s, phi_dc, phi_ac, theta_p, theta_s)

    # Calculate the chain of matrices representing the Klopfenstein tapper
    a = 2.*np.pi*f*np.sqrt(beta)*l/(n - 1.)
    m = np.array([np.cos(a), 1j*z*np.sin(a), 1j/z*np.sin(a), np.cos(a)])

    # Give the good shape to the array for the matrix product
    m = m.transpose()
    m.shape = (n, 2, 2)

    # Calculate the chain matrix prodcut
    M = reduce(np.dot, m)

    # We end the chain by two elements:
    # 1 - a load impedance to the ground
    # 2 - a huge impedance to the circuit
    if as_theory:
        M = M.dot(np.array([[1., 0.],[1./zl, 1.]]))
    else:
        M = M.dot(np.array([[1., 0.],[1./ljpa.impedance(f), 1.]]))

    M = M.dot(np.array([[1., 1e99],[0., 1.]]))

    # Compute the reflection from the array elements
    a = M.item(0)
    b = M.item(1)
    c = M.item(2)
    d = M.item(3)

    return (a + b/50. - c*50. - d)/(a + b/50. + c*50. + d)
