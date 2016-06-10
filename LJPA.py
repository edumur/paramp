# This Python file uses the following encoding: utf-8

# Implementation of the Pumpistor model of a flux pumped SQUID in the
# three wave mixing degenerate case ω_p = ω_s + ω_i and ω_p = ω_s.
#
# Based on an article from K. M. Sundqvist et al:
# "The pumpistor: A linearized model of a flux-pumped superconducting
# quantum interference device for use as a negative-resistance parametric
# amplifier"
# APL 109 102603 (2013),
# and on an article from J. Y. Mutus et al:
# "Design and characterization of a lumped element single-ended
# superconducting microwave parametric amplifier with on-chip
# flux bias line"
# APL 103 122602 (2013)

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
from scipy.optimize import minimize

from JPA import JPA
from find import Find


class LJPA(JPA, Find):



    def __init__(self, C, L_s, I_c, phi_s, phi_dc, phi_ac, theta_p,
                 theta_s = 0., f_p=None):
        """
        Implementation of the LJPA in the three wave mixing degenerate case
        ω_p = ω_s + ω_i and ω_p = ω_s.

        Based on an article of K. M. Sundqvist et al:
        "Design and characterization of a lumped element single-ended
        superconducting microwave parametric amplifier with on-chip
        flux bias line"
        APL 103 122602 (2013)

        ...

        Attributes
        ----------
        C : float
            Resonator capacitance in farad.
        L_s : float
            Resonator stray inductance in henry.
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

        if type(C) is not float:
            raise ValueError('C parameter must be float type.')
        if type(L_s) is not float:
            raise ValueError('L_s parameter must be float type')

        JPA.__init__(self, I_c, phi_s, phi_dc, phi_ac, theta_p, theta_s, f_p)

        self.C   = C
        self.L_s = L_s



    def external_impedance(self, f, R0=50.):
        """
        Return the impedance of the electrical environment seen by the SQUID.
        We assume the circuit to be 50 ohm matched.

        Parameters
        ----------
        f : float, np.ndarray, optional
            Signal frequency in hertz.
            Is required in the non-degenerate case but optional for the
            degenerate one.
        R0 : float, optional
            The characteristic impedance of the incoming line. Assumed to be
            50 ohm.
        """

        o = 2.*np.pi*f

        return 1./(1j*o*self.L_s + 1./(1j*o*self.C + 1./R0))



    def impedance(self, f):
        """
        Return the impedance of the resonator formed by the SQUID, the stray
        inductance and the capacitance.

        Parameters
        ----------
        f : float, np.ndarray
            The frequency in Hz
        """

        o = 2.*np.pi*f

        return 1./(1j*self.C*o + 1./(1j*o*(self.L_s + self.squid_inductance(f))))


    def angular_resonance_frequency(self, f=None):
        """
        Return the angular resonance frequency in rad.Hz of the resonator formed
        by the SQUID, the stray inductance and the capacitance.

        Parameters
        ----------
        f : float, np.ndarray, optional
            Signal frequency in hertz.
            Is required in the non-degenerate case but optional for the
            degenerate one.
        """

        return 1./np.sqrt(self.C*(self.L_s + self.squid_inductance(f).real))


    def resonance_frequency(self, f=None):
        """
        Return the resonance frequency in Hz of the resonator formed by the
        SQUID, the stray inductance and the capacitance.

        Parameters
        ----------
        f : float, np.ndarray, optional
            Signal frequency in hertz.
            Is required in the non-degenerate case but optional for the
            degenerate one.
        """

        return 1./np.sqrt(self.C*(  self.L_s\
                                  + self.squid_inductance(f).real))/2./np.pi



    def equivalent_resistance(self, f=None):
        """
        Return the resistance in ohm of the equivalente resonator formed by the
        SQUID, the stray inductance and the capacitance.

        Parameters
        ----------
        f : float, np.ndarray, optional
            Signal frequency in hertz.
            Is required in the non-degenerate case but optional for the
            degenerate one.
        """

        a = self.squid_inductance(f).real
        b = self.squid_inductance(f).imag

        o0 = self.angular_resonance_frequency(f)

        return -o0*(self.L_s + a)**2./b



    def equivalent_capacitance(self, f=None):
        """
        Return the capacitance in farad of the equivalente resonator formed by
        the SQUID, the stray inductance and the capacitance.

        Parameters
        ----------
        f : float, np.ndarray, optional
            Signal frequency in hertz.
            Is required in the non-degenerate case but optional for the
            degenerate one.
        """

        a = self.squid_inductance(f).real
        b = self.squid_inductance(f).imag

        return self.C/2.*(3. - (self.L_s + a)**2./(b**2. + (self.L_s + a)**2.))



    def equivalent_inductance(self, f=None):
        """
        Return the inductance in henry of the equivalente resonator formed by
        the SQUID, the stray inductance and the capacitance.

        Parameters
        ----------
        f : float, np.ndarray, optional
            Signal frequency in hertz.
            Is required in the non-degenerate case but optional for the
            degenerate one.
        """

        a = self.squid_inductance(f).real
        b = self.squid_inductance(f).imag

        return 2.*(self.L_s + a)/(3. - 1./(1. + (b/(self.L_s + a))**2.))



    def equivalent_impedance(self, f):
        """
        Return the impedance of the equivalente resonator formed by the SQUID,
        the stray inductance and the capacitance.

        Parameters
        ----------
        f : float, np.ndarray
            Signal frequency in hertz.
        """

        o = 2.*np.pi*f

        return 1./(1./self.equivalent_resistance(f)\
                   + 1j*o*self.equivalent_capacitance(f)\
                   + 1./1j/o/self.equivalent_inductance(f))



    def equivalent_angular_resonance_frequency(self, f=None):
        """
        Return the angular resonance frequency in rad.Hz of the equivalent_capacitance
        resonator formed by the SQUID, the stray inductance and the capacitance.

        Parameters
        ----------
        f : float, np.ndarray, optional
            Signal frequency in hertz.
            Is required in the non-degenerate case but optional for the
            degenerate one.
        """

        return 1./np.sqrt(self.equivalent_capacitance(f)\
                          *self.equivalent_inductance(f))/2./np.pi



    def equivalent_resonance_frequency(self, f=None):
        """
        Return the resonance frequency in Hz of the equivalent resonator formed
        by the SQUID, the stray inductance and the capacitance.

        Parameters
        ----------
        f : float, np.ndarray, optional
            Signal frequency in hertz.
            Is required in the non-degenerate case but optional for the
            degenerate one.
        """

        return 1./np.sqrt(self.equivalent_capacitance(f)\
                          *self.equivalent_inductance(f))/2./np.pi



    def internal_quality_factor(self, f=None):
        """
        Return the internal quality factor (Qi) of the equivalente resonator
        formed by the SQUID, the stray inductance and the capacitance.
        Since there is not dissipation in the model, Qi is related to the
        flux pumped SQUID more than losses.

        Parameters
        ----------
        f : float, np.ndarray, optional
            Signal frequency in hertz.
            Is required in the non-degenerate case but optional for the
            degenerate one.
        """

        a = self.squid_inductance(f).real
        b = self.squid_inductance(f).imag

        return -(self.L_s+a)/2./b*(3 - (self.L_s+a)**2./(b**2.+(self.L_s+a)**2.))



    def coupling_quality_factor(self, f=None, R0=50.):
        """
        Return the coupling quality factor (Qc) of the equivalente resonator
        formed by the SQUID, the stray inductance and the capacitance.

        Parameters
        ----------
        f : float, np.ndarray, optional
            Signal frequency in hertz.
            Is required in the non-degenerate case but optional for the
            degenerate one.
        R0 : float, optional
            The characteristic impedance of the incoming line. Assumed to be
            losses line so real and 50 ohm.
        """

        return R0*np.sqrt( self.equivalent_capacitance(f)\
                           /self.equivalent_inductance(f))



    def total_quality_factor(self, f=None, R0=50.):
        """
        Return the total quality factor (Q0) of the equivalente resonator
        formed by the SQUID, the stray inductance and the capacitance.

        Parameters
        ----------
        f : float, np.ndarray, optional
            Signal frequency in hertz.
            Is required in the non-degenerate case but optional for the
            degenerate one.
        R0 : float, optional
            The characteristic impedance of the incoming line. Assumed to be
            losses line so real and 50 ohm.
        """

        return 1./(  1./self.internal_quality_factor(f)\
                   + 1./self.coupling_quality_factor(f, R0))



    def optimized_squid_inductance_imag(self, f=None, R0=50.):
        """
        Return the imaginary part of the SQUID inductance for which the
        internal quality factor equalizes the coupling quality factor.

        Parameters
        ----------
        f : float, np.ndarray, optional
            Signal frequency in hertz.
            Is required in the non-degenerate case but optional for the
            degenerate one.
        R0 : float, optional
            The characteristic impedance of the incoming line. Assumed to be
            losses line so real and 50 ohm.
        """

        a = self.squid_inductance(f).real

        return (R0 - np.sqrt(R0**2. - 4.*np.sqrt((self.L_s + a)**3./self.C)))/2.



    def optimized_resonator_capacitance(self, f=None, R0=50.):
        """
        Return the resonator capacitance for which the internal quality factor
        equalizes the coupling quality factor.

        Parameters
        ----------
        f : float, np.ndarray, optional
            Signal frequency in hertz.
            Is required in the non-degenerate case but optional for the
            degenerate one.
        R0 : float, optional
            The characteristic impedance of the incoming line. Assumed to be
            losses line so real and 50 ohm.
        """

        a = self.squid_inductance(f).real
        b = self.squid_inductance(f).imag

        return (self.L_s + a)**3./b**2./(b - R0)**2.



    def optimized_LJPA(self, f0, Qc, BW=None,
                             R0=50.,
                             fixed=[None],
                             weight={'f0':10., 'Qc=Qi':1., 'Qc':1., 'BW':1.},
                             update_parameters=False,
                             full_output=False,
                             verbose=False,
                             method='Nelder-Mead',
                             bounds=None):
        """
        Optimized the different parameters of the LJPA to reached a target
        frequency and coupling quality factor.
        This is done by minimizing the relative error of three values:
            1 - the resonance frequency,
            2 - the coupling quality factor,
            3 - the absolute difference between the coupling and internal
                quality factor.
            4 (optional) - the bandwidth.

        Work only in the degenerate case !

        Parameters
        ----------
        f0 : float
            Target resonance frequency in GHz.
        Qc : float
            Target coupling quality factor.
        update_parameters : bool, optional
            If the differents parameters found after the optimization are set
            to be the parameters of the LPJA instance.
        full_output : bool, optional
            To return all optional output, if not, return just the parameters.
        verbose : bool, optional
            To print parameters, targets value and least square value during
            optimization.
        method : str, optional
            Type of solver. Should be one of
                'Nelder-Mead' - default
                'Powell'
                'CG'
                'BFGS'
                'Newton-CG'
                'L-BFGS-B'
                'TNC'
                'COBYLA'
                'SLSQP'
                'dogleg'
                'trust-ncg'

        Return
        ----------
        x : np.ndarray
            For :
            full_output == False:
                The solution of the optimization [phi_ac, I_c, L_s, C].
            full_output == True:
                x : ndarray
                    The solution of the optimization.
                success : bool
                    Whether or not the optimizer exited successfully.
                status : int
                    Termination status of the optimizer. Its value depends on
                    the underlying solver. Refer to message for details.
                message : str
                    Description of the cause of the termination.
                fun, jac, hess: ndarray
                    Values of objective function, its Jacobian and its Hessia
                    if (available). The Hessians may be approximations, see the
                    documentation of the function in question.
                hess_inv : object
                    Inverse of the objective function’s Hessian; may be an
                    approximation. Not available for all solvers. The type of
                    this attribute may be either np.ndarray or
                    scipy.sparse.linalg.LinearOperator.
                nfev, njev, nhev : int
                    Number of evaluations of the objective functions and of its
                    Jacobian and Hessian.
                nit : int
                    Number of iterations performed by the optimizer.
                maxcv : float
                    The maximum constraint violation.
        """

        def func(x, f0, Qc, R0, names):

            x = abs(x)

            for value, name in zip(x, names):
                if name == 'phi_ac':
                    self.phi_ac = value
                elif name == 'phi_dc':
                    self.phi_dc = value
                elif name == 'I_c':
                    self.I_c = value
                elif name == 'L_s':
                    self.L_s = value
                elif name == 'C':
                    self.C = value

            current_f0 = self.find_resonance_frequency(R0)
            current_Qc = self.coupling_quality_factor(current_f0, R0)
            current_Qi = self.internal_quality_factor(current_f0)

            if BW is not None:
                current_BW = self.find_reflection_fwhm()
                relative_error_BW = ((current_BW - BW)/BW*weight['BW'])**2.
            else:
                relative_error_BW = 0.

            y =  np.sum(((current_f0 - f0)/f0*weight['f0'])**2.\
                      + ((current_Qi + current_Qc)/Qc*weight['Qc=Qi'])**2.\
                      + ((Qc  - current_Qc)/Qc*weight['Qc'])**2.\
                      + relative_error_BW)

            if verbose:
                print '     ----------'
                print 'Parameters:'
                if len(names) != 5:
                    print '    Fixed:'
                    if 'phi_ac' not in names:
                        print '        phi_ac = '+str(round(self.phi_ac, 3))+ ' phi_0'
                    if 'phi_dc' not in names:
                        print '        phi_dc = '+str(round(self.phi_dc, 3))+ ' phi_0'
                    if 'I_c' not in names:
                        print '        I_c = '+str(round(self.I_c*1e6, 3))+ ' uA'
                    if 'L_s' not in names:
                        print '        L_s = '+str(round(self.L_s*1e12, 3))+ ' pH'
                    if 'C' not in names:
                        print '        C = '+str(round(self.C*1e12, 3))+ ' pF'
                print '    Optimized:'
                if 'phi_ac' in names:
                    print '        phi_ac = '+str(round(self.phi_ac, 3))+ ' phi_0'
                if 'phi_dc' in names:
                    print '        phi_dc = '+str(round(self.phi_dc, 3))+ ' phi_0'
                if 'I_c' in names:
                    print '        I_c = '+str(round(self.I_c*1e6, 3))+ ' uA'
                if 'L_s' in names:
                    print '        L_s = '+str(round(self.L_s*1e12, 3))+ ' pH'
                if 'C' in names:
                    print '        C = '+str(round(self.C*1e12, 3))+ ' pF'
                print '        '
                print 'Results:'
                print '    f_0 = '+str(round(current_f0/1e9, 3))+' GHz, weight: '+str(weight['f0'])
                print '    Q_c = '+str(round(current_Qc, 3))+', weight: '+str(weight['Qc'])
                print '    Q_i = '+str(round(current_Qi, 3))+', weight: '+str(weight['Qc=Qi'])
                if BW is not None:
                    print '    BW = '+str(round(current_BW/1e6, 3))+' MHz, weight: '+str(weight['BW'])
                print ''
                print 'Least square:'
                print '    '+str(y)
                print ''

            return y

        # Get a list of variables parameters name and value
        params_name  = ['phi_ac', 'phi_dc', 'I_c', 'L_s', 'C']
        params_value = [self.phi_ac, self.phi_dc, self.I_c, self.L_s, self.C]

        values = []
        names  = []
        for param_name, param_value in zip(params_name, params_value):
            if param_name not in fixed:
                names.append(param_name)
                values.append(param_value)

        # Store a backup before the minimization
        backups = params_value

        results = minimize(func,
                           values,
                           args=(f0, Qc, R0, names),
                           method=method,
                           bounds=bounds)

        if not update_parameters:

            self.phi_ac = backups[0]
            self.phi_dc = backups[1]
            self.I_c    = backups[2]
            self.L_s    = backups[3]
            self.C      = backups[4]

        if full_output:
            return results
        else:
            return results.x



    def reflection(self, f, z0=50.):
        """
        Return the reflection of the LJPA.

        Parameters
        ----------
        f : float, np.ndarray
            The frequency in hertz.
        z0 : float, optional
            The characteristic impedance of the transmission line connected to
            the SQUID, default is 50 Ω.

        Raises
        ------
        ValueError
            If the parameters are not in the good type
        """

        if type(f) in (str, list, dict):
            raise ValueError('f parameter must be float or np.ndarray type')

        return  (self.impedance(f) - z0)\
               /(self.impedance(f) + z0)



    def _parse_number(self, number, precision, inverse = False):

        power_ten = int(np.log10(abs(number)))/3*3

        if power_ten >= -24 and power_ten <= 18 :

            prefix = {-24 : 'y',
                      -21 : 'z',
                      -18 : 'a',
                      -15 : 'p',
                      -12 : 'p',
                       -9 : 'n',
                       -6 : 'µ',
                       -3 : 'm',
                        0 : '',
                        3 : 'k',
                        6 : 'M',
                        9 : 'G',
                       12 : 'T',
                       15 : 'p',
                       18 : 'E'}

            if inverse:
                return str(round(number*10.**-power_ten, precision)),\
                       prefix[-power_ten]
            else:
                return str(round(number*10.**-power_ten, precision)),\
                       prefix[power_ten]
        else:
            return str(round(number, precision)), ''



    def __str__(self):

        Ic_p, Ic_t = self._parse_number(self.I_c, 3)
        delta_theta_p, delta_theta_t = self._parse_number(self.delta_theta()/np.pi, 3)
        phi_s_p, phi_s_t = self._parse_number(self.phi_s, 3)
        phi_dc_p, phi_dc_t = self._parse_number(self.phi_dc, 3)
        phi_ac_p, phi_ac_t = self._parse_number(self.phi_ac, 3)

        C_p, C_t = self._parse_number(self.C, 3)
        Ls_p, Ls_t = self._parse_number(self.L_s, 3)



        return '------------------------------------------------------------\n'\
               'LJPA instanced with following parameters:\n'\
               '\n'\
               '    Pumpistor parameters:\n'\
               '        SQUID critical current:       '+Ic_p+' '+Ic_t+'A\n'\
               '        Signal-pump phase difference: '+delta_theta_p+' '+delta_theta_t+'π\n'\
               '        Signal amplitude:             '+phi_s_p+' '+phi_s_t+'rad\n'\
               '        DC pump amplitude:            '+phi_dc_p+' '+phi_dc_t+'Φ0\n'\
               '        AC pump amplitude:            '+phi_ac_p+' '+phi_ac_t+'Φ0\n'\
               '\n'\
               '    Resonator parameters:\n'\
               '        Capacitance:      '+C_p+' '+C_t+'F\n'\
               '        Stray inductance: '+Ls_p+' '+Ls_t+'H\n'\
               '------------------------------------------------------------\n'


    def print_result(self):

        f0_p, f0_t = self._parse_number(self.find_resonance_frequency(), 3)
        fwhm_p, fwhm_t = self._parse_number(self.find_reflection_fwhm(), 3)
        max_gain_p, max_gain_t = self._parse_number(self.find_max_gain(scale='log'), 3)
        Qi_p, Qi_t = self._parse_number(self.internal_quality_factor(), 3)
        Qc_p, Qc_t = self._parse_number(self.coupling_quality_factor(), 3)

        return '------------------------------------------------------------\n'\
               '                       Results:\n'\
               '\n'\
               '    Resonator:\n'\
               '        Resonance frequency:     '+f0_p+' '+f0_t+'Hz\n'\
               '        FWHM:                    '+fwhm_p+' '+fwhm_t+'Hz\n'\
               '        Max gain:                '+max_gain_p+' '+max_gain_t+'dB\n'\
               '        Internal quality factor: '+Qi_p+' '+Qi_t+'\n'\
               '        Coupling quality factor: '+Qc_p+' '+Qc_t+'\n'\
               '------------------------------------------------------------\n'
