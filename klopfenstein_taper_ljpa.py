# This Python file uses the following encoding: utf-8

# Implementation of the Klopfenstein taper from R. W. Klopfenstein:
# "A Transmission Line Taper of Improved Design"
# Proceedings of the IRE, 44 (1956)
# , the book "Microwave engineering" from D. M. Pozar
# and the paper from M. A. Grossberg:
# "Extremely rapid computation of the Klopfenstein impedance taper"
# Proceedings of the IEEE, 56 (1968)

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
from scipy.optimize import minimize, fsolve
from multiprocessing import Pool
import itertools

from JPA import JPA
from find import Find
from klopfenstein_discretization import (reflection_discretization,
                                         external_discretization,
                                         ljpa_external_discretization)


class KlopfensteinTaperLJPA(JPA, Find):



    def __init__(self, C, L_s, I_c, phi_s, phi_dc, phi_ac, theta_p,
                       Z_l, l, g_m, L_l, C_l,
                       theta_s=0., f_p=None):
        """
        Implementation of the Klopfenstein taper from R. W. Klopfenstein:
        "A Transmission Line Taper of Improved Design"
        Proceedings of the IRE, 44 (1956)
        , the book "Microwave engineering" from D. M. Pozar
        and the paper from M. A. Grossberg:
        "Extremely rapid computation of the Klopfenstein impedance taper"
        Proceedings of the IEEE, 56 (1968)

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
        Z_l : float,
            Impedance of the loaded element in ohm.
        l : float
            Length of the taper line in meter.
        g_m : float
            Magnitude of the ripple in the bandpass of the taper line.
        L_l : float
            Inductance per unit length of the taper line in henry per meter.
        C_l : float
            capacitance per unit length of the taper line in farad per meter.
        theta_s : {0.} float, optional
            Phase of the pump in rad, default is zero which implies that
            the signal phase is the reference.
        f_p : float, optional
            Pump frequency. If None we assume  f_p = 2*f_s.

        Raises
        ------
        ValueError
            If the parameters are not in the good type.
        """

        if not isinstance(Z_l, float):
            raise ValueError('Z_l parameter must be float type.')
        if not isinstance(l, float):
            raise ValueError('l parameter must be float type.')
        if not isinstance(g_m, float):
            raise ValueError('g_m parameter must be float type.')
        if not isinstance(L_l, float):
            raise ValueError('L_l parameter must be float type.')
        if not isinstance(C_l, float):
            raise ValueError('C_l parameter must be float type.')

        JPA.__init__(self, I_c, phi_s, phi_dc, phi_ac, theta_p, theta_s)

        self.zl  = Z_l
        self.l   = l
        self.gm  = g_m
        self.cl  = C_l
        self.ll  = L_l

        self.C = C
        self.L_s = L_s

        self.f_p = f_p






    def _phi(self, x, A, convergence=1e-7):
        """
        Return the integral of the modified Bessel function of the first kind
        by using the method explain by M. A. Grossberg in
        "Extremely rapid computation of the Klopfenstein impedance taper"
        Proceedings of the IEEE, 56 (1968).

        Parameters
        ----------
        convergence : float
            Value at which the method stop to refine the result of phi.
        """

        a   = 1.
        b   = 0.5*x
        phi = b
        k   = 1.

        def condition(phi, a, b):

            if type(b) is float:
                if b == 0:
                    return convergence/10.
                else:
                    return abs(1. - (phi + a*b)/phi)
            else:
                return abs(1. - (phi + a*b)/phi).all()

        while condition(phi, a, b) > convergence:

            a = A**2./(4.*k*(k + 1.))*a
            b = (x*(1 - x**2.)**k/2. + 2.*k*b)/(2*k + 1.)
            phi += a*b
            k   += 1.

        return phi



    def A(self):
        """
        Return maximum magnitude of reflection coefficient in the pass band.
        """

        y = self.g0()/self.gm
        if y < 1.:
            raise ValueError('g0/gm must be higher than 1.')

        return np.arccosh(self.g0()/self.gm)



    def g0(self, z0=50.):
        """
        Return the reflection coeficient at zero frequency.

        Parameters
        ----------
        z0 : {50.} float optional
            Impedance of the incoming transmission line.
        """

        return abs(self.zl - z0)/(self.zl + z0)



    def beta(self, f):
        """
        Return the wave vector of the taper line.

        Parameters
        ----------
        f : float, np.ndarray
            Frequency in GHz
        """

        return 2.*np.pi*f*np.sqrt(self.cl*self.ll)



    def characteristic_impedance(self, z, z0=50.):
        """
        Return the characteristic impedance of the tapper along its length.

        Parameters
        ----------
        z : float, np.ndarray
            Length position along the taper in meter.
            z must be higher than 0 and lower than the taper length.
        z0 : {50.} float optional
            Impedance of the incoming transmission line.

        Raises
        ------
        ValueError
            If z is not in the good range.
        """

        if (z < 0.).any():
            raise ValueError('z must be greater than 0')
        if (z > self.l).any():
            raise ValueError('z must be smaller than the tapper length: '+self.l)

        y = np.exp(np.log(z0*self.zl)/2.\
                      + self.g0()*self.A()**2.*self._phi(2.*z/self.l - 1., self.A())\
                        /np.cosh(self.A()))

        if z0 > self.zl:
            return y[::-1]
        else:
            return y



    def reflection_theory(self, f):
        """
        Return the theoretical reflection of the taper.
        As a theoreticalk result this reflection doesn't take into account the
        impedance of the LJPA.

        Parameters
        ----------
        f : float, np.ndarray
            Frequency in GHz.
        """

        v = (self.beta(f)*self.l)**2. - self.A()**2.

        if type(f) is float:
            if v < 0:
                c = np.cosh(np.sqrt(self.A()**2. - (self.beta(f)*self.l)**2.))
            else:
                c = np.cos(np.sqrt((self.beta(f)*self.l)**2. - self.A()**2.))
        else:
            c = np.ones_like(v)
            c[v<0] = np.cosh(np.sqrt(self.A()**2. - (self.beta(f[v<0])*self.l)**2.))
            c[v>0] = np.cos(np.sqrt((self.beta(f[v>0])*self.l)**2. - self.A()**2.))

        return self.gm*np.exp(-1j*self.beta(f)*self.l)*c



    def find_ll_cl(self, z):
        """
        Quick and dirty method to find correct inductance and capacitance per
        unit length which give a target z impedance.
        This is used by the reflection method because of the wave vector
        calculation.

        Parameters
        ----------
        z : np.ndarray
            Target impedance in ohm.
        """

        def func(x0, z):

            ll, cl = abs(x0)

            return ((np.sqrt(ll/cl) - z)/z)**2

        temp = []
        for i in z:

            temp.append(abs(minimize(func,
                                 [self.ll, self.cl],
                                 args=(i,),
                                 method='Nelder-Mead').x))

        temp = np.array(temp)

        return temp[:,0], temp[:,1]



    def external_impedance(self, f, n=1e2, R0=50., as_theory=False, simple_ext=False):
        """
        Return the impedance of the electrical environment seen by the SQUID.
        We assume the circuit to be 50 ohm matched.

        Use multithreading for faster calculation.

        Parameters
        ----------
        f : float, np.ndarray
            Signal frequency in hertz.
        n : float, optional
            Number of discret elements used to model the taper line.
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

        if self.f_p is None:

            return None

        # Calculate the different characteristic impedance of the different
        # part of the transnmission line
        # We inverse it because of the SQUID point of view
        z = self.characteristic_impedance(np.linspace(0., self.l, n))[::-1]

        # Calculate the corresponding L_l and C_l for the elements
        if as_theory:
            ll = np.ones_like(z)*self.ll
            cl = np.ones_like(z)*self.cl
        else:
            ll, cl = self.find_ll_cl(z)

        prod = self.l*np.sqrt(ll*cl)/(n - 1.)

        # We need iterable frequency for the parallelization
        if isinstance(f, float):

            return external_discretization((self.f_p - f, z, prod, self.zl,
                                            self.C, self.L_s, self.I_c, self.phi_s,
                                            self.phi_dc, self.phi_ac, self.theta_p,
                                            self.theta_s, as_theory, simple_ext,
                                            self.f_p))
        else:

            # Create a pool a thread for fast computation
            # We look at the impedance at the pump frequency
            pool = Pool()
            result = pool.map(external_discretization,
                              itertools.izip(self.f_p - f,
                                             itertools.repeat(z),
                                             itertools.repeat(prod),
                                             itertools.repeat(self.zl),
                                             itertools.repeat(self.C),
                                             itertools.repeat(self.L_s),
                                             itertools.repeat(self.I_c),
                                             itertools.repeat(self.phi_s),
                                             itertools.repeat(self.phi_dc),
                                             itertools.repeat(self.phi_ac),
                                             itertools.repeat(self.theta_p),
                                             itertools.repeat(self.theta_s),
                                             itertools.repeat(as_theory),
                                             itertools.repeat(simple_ext),
                                             itertools.repeat(self.f_p)))

            pool.close()
            pool.join()

            return np.array(result)




    def coupling_impedance(self, n=1e2, as_theory=False):
        """
        Return the impedance of the environment seen by the LJPA

        Parameters
        ----------
        n : float, optional
            Number of discret elements used to model the taper line.
        as_theory : bool, optional
            If true use the load impedance of the characteristic impeance
            calculation to try to mimic the theoritical reflection.
            Use this parameter to test if this method can correctly mimic
            the theoretical expectation.
        """

        # Calculate the different characteristic impedance of the different
        # part of the transnmission line
        z = self.characteristic_impedance(np.linspace(0., self.l, n))

        # Calculate the corresponding L_l and C_l for the elements
        if as_theory:
            ll = np.ones_like(z)*self.ll
            cl = np.ones_like(z)*self.cl
        else:
            ll, cl = self.find_ll_cl(z)

        prod = self.l*np.sqrt(ll*cl)/(n - 1.)
        f = self.find_resonance_frequency()
        param = (f, z, prod, self.zl, self.C, self.L_s,
                                     self.I_c, self.phi_s, self.phi_dc,
                                     self.phi_ac, self.theta_p, self.theta_s,
                                     as_theory, self.f_p)
        result = ljpa_external_discretization(param)

        return result



    def equivalent_resistance(self, f=None, R0=50.):
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

        z_ext = self.external_impedance(f, R0)

        a = self.squid_inductance(f, z_ext).real
        b = self.squid_inductance(f, z_ext).imag

        o0 = self.find_angular_resonance_frequency(R0)

        return -o0*(self.L_s + a)**2./b



    def equivalent_capacitance(self, f=None, R0=50.):
        """
        Return the capacitance in farad of the equivalent resonator formed by
        the SQUID, the stray inductance and the capacitance.

        Parameters
        ----------
        f : float, np.ndarray, optional
            Signal frequency in hertz.
            Is required in the non-degenerate case but optional for the
            degenerate one.
        """

        z_ext = self.external_impedance(f, R0)

        a = self.squid_inductance(f, z_ext).real
        b = self.squid_inductance(f, z_ext).imag

        return self.C/2.*(3. - (self.L_s + a)**2./(b**2. + (self.L_s + a)**2.))



    def equivalent_inductance(self, f=None, R0=50.):
        """
        Return the inductance in henry of the equivalent resonator formed by
        the SQUID, the stray inductance and the capacitance.

        Parameters
        ----------
        f : float, np.ndarray, optional
            Signal frequency in hertz.
            Is required in the non-degenerate case but optional for the
            degenerate one.
        """

        z_ext = self.external_impedance(f, R0)

        a = self.squid_inductance(f, z_ext).real
        b = self.squid_inductance(f, z_ext).imag

        return 2.*(self.L_s + a)/(3. - 1./(1. + (b/(self.L_s + a))**2.))



    def equivalent_rlc(self, f=None, R0=50.):

        z_ext = self.external_impedance(f, R0)

        o0 = self.find_angular_resonance_frequency(R0)

        a = self.squid_inductance(f, z_ext).real
        b = self.squid_inductance(f, z_ext).imag

        l_eq =  2.*(self.L_s + a)/(3. - 1./(1. + (b/(self.L_s + a))**2.))
        c_eq = self.C/2.*(3. - (self.L_s + a)**2./(b**2. + (self.L_s + a)**2.))
        r_eq = -o0*(self.L_s + a)**2./b

        return r_eq, l_eq, c_eq



    def total_quality_factor(self, f=None, R0=50., n=1e2, as_theory=False):
        """
        Return the total quality factor (Qc) of the equivalent resonator
        formed by the SQUID, the stray inductance, the capacitance and the
        Klopfenstein taper.

        Parameters
        ----------
        f : float, np.ndarray, optional
            Signal frequency in hertz.
            Is required in the non-degenerate case but optional for the
            degenerate one.
        n : float, optional
            Number of discrete elements used to model the taper line.
        as_theory : bool, optional
            If true use the load impedance of the characteristic impedance
            calculation to try to mimic the theoretical reflection.
            Use this parameter to test if this method can correctly mimic
            the theoretical expectation.
        """

        # find the equivalent parallel coupling resistance and inductance at the
        # resonance frequency
        z_coupling = self.coupling_impedance(n, as_theory)
        r_eq, l_eq, c_eq = self.equivalent_rlc()

        rcoup = 1./np.real(1./z_coupling)
        lcoup = -1./self.find_angular_resonance_frequency(R0)/np.imag(1./z_coupling)

        # find the total resistance and inductance
        rtot = 1./(1./r_eq + 1./rcoup)
        ltot = 1./(1./l_eq + 1./lcoup)

        # find the total quality factor
        return rtot*np.sqrt(c_eq/ltot)



    def coupling_quality_factor(self, f=None, R0=50., n=1e2, as_theory=False):
        """
        Return the coupling quality factor (Qc) of the equivalent resonator
        formed by the SQUID, the stray inductance, the capacitance and the
        Klopfenstein taper.

        Parameters
        ----------
        f : float, np.ndarray, optional
            Signal frequency in hertz.
            Is required in the non-degenerate case but optional for the
            degenerate one.
        n : float, optional
            Number of discrete elements used to model the taper line.
        as_theory : bool, optional
            If true use the load impedance of the characteristic impedance
            calculation to try to mimic the theoretical reflection.
            Use this parameter to test if this method can correctly mimic
            the theoretical expectation.
        """

        return 1./(1./self.total_quality_factor(f, R0, n, as_theory)\
                - 1./self.internal_quality_factor(f, R0))



    def internal_quality_factor(self, f=None, R0=50.):
        """
        Return the internal quality factor (Qi) of the equivalent resonator
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

        z_ext = self.external_impedance(f, R0)

        a = self.squid_inductance(f, z_ext).real
        b = self.squid_inductance(f, z_ext).imag

        return -(self.L_s+a)/2./b*(3 - (self.L_s+a)**2./(b**2.+(self.L_s+a)**2.))



    def reflection(self, f, n=1e2, as_theory=False, simple_ext=False):
        """
        Return the reflection of the taper.
        To do so, the taper impedance is discretised in n sections.
        Each of these sections is modeled as a lossless transmission line.
        At the end the LJPA is modeled as a lumped element to ground and very
        high impedance (1e99 ohm) to the line.

        This method use multithreading calculation to be faster.
        as a consequence asking only one frequency is slower than without
        multithreading.
        Since this case of use is rare it shouln't be a problem.

        Parameters
        ----------
        f : float, np.ndarray
            Frequency in GHz.
        n : float, optional
            Number of discret elements used to model the taper line.
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

        # Calculate the different characteristic impedance of the different
        # part of the transnmission line
        z = self.characteristic_impedance(np.linspace(0., self.l, n))

        # Calculate the corresponding L_l and C_l for the elements
        if as_theory:
            ll = np.ones_like(z)*self.ll
            cl = np.ones_like(z)*self.cl
        else:
            ll, cl = self.find_ll_cl(z)

        prod = self.l*np.sqrt(ll*cl)/(n - 1.)

        # We need iterable frequency for the parallelization
        if isinstance(f, float):

            # If the pump frequency is None, we don't have to calculate the impedance
            # seen by the pumpistor
            if self.f_p is not None:
                z_ext = self.external_impedance(f, n, 50., as_theory, simple_ext)
            else:
                z_ext = None

            return reflection_discretization((f, z_ext, z, prod, self.zl, self.C,
                                              self.L_s, self.I_c, self.phi_s,
                                              self.phi_dc, self.phi_ac, self.theta_p,
                                              self.theta_s, as_theory, self.f_p))
        else:

            # If the pump frequency is None, we don't have to calculate the impedance
            # seen by the pumpistor
            if self.f_p is not None:
                z_ext = self.external_impedance(f, n, 50., as_theory, simple_ext)
            else:
                z_ext = itertools.repeat(None)

            # Create a pool a thread for fast computation
            pool = Pool()
            result = pool.map(reflection_discretization,
                              itertools.izip(f,
                                             z_ext,
                                             itertools.repeat(z),
                                             itertools.repeat(prod),
                                             itertools.repeat(self.zl),
                                             itertools.repeat(self.C),
                                             itertools.repeat(self.L_s),
                                             itertools.repeat(self.I_c),
                                             itertools.repeat(self.phi_s),
                                             itertools.repeat(self.phi_dc),
                                             itertools.repeat(self.phi_ac),
                                             itertools.repeat(self.theta_p),
                                             itertools.repeat(self.theta_s),
                                             itertools.repeat(as_theory),
                                             itertools.repeat(self.f_p)))

            pool.close()
            pool.join()

            # If the user a np.ndarray frequency, we return a np.ndarray
            return np.array(result)



    def impedance(self, f, R0=50, n=1e2, as_theory=False):
            """
            Return the impedance of the taper.

            Parameters
            ----------
            f : float, np.ndarray
                Frequency in GHz.
            R0 : float, optional
                The characteristic impedance of the incoming line. Assumed to be
                50 ohm.
            n : float, optional
                Number of discret elements used to model the taper line.
            as_theory : bool, optional
                If true use the load impedance of the characteristic impedance
                calculation to try to mimic the theoretical reflection.
                Use this parameter to test if this method can correctly mimic
                the theoretical expectation.
            """

            r = self.reflection(f, n, as_theory)

            return R0*(1+r)/(1-r)



    def optimized_KLJPA(self, f0, BW=None,
                             R0=50.,
                             fixed=[None],
                             weight={'f0':10., 'R0' : 1., 'BW':1.},
                             update_parameters=False,
                             verbose=False,
                             method='Nelder-Mead',
                             bounds=None):
        """
        Optimized the different parameters of the LJPA to reached a target
        frequency.
        This is done by minimizing the relative error of three values:
            1 - the resonance frequency,
            2 - the absolute difference between the coupling and internal
                quality factor.
            3 (optional) - the bandwidth.

        Parameters
        ----------
        f0 : float
            Target resonance frequency in GHz.
        BW : float, optional
            Target bandwidth.
            If None the bandwidth is free during the optimization.
        update_parameters : bool, optional
            If the differents parameters found after the optimization are set
            to be the parameters of the LPJA instance.
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
        bounds : list of tuples, optional
            Bounds (min, max) pairs for each parameter.
            Only for L-BFGS-B, TNC and SLSQP methods.
            Use None for one of min or max when there is no bound in that direction.

        Return
        ----------
        x : np.ndarray
            The solution of the optimization ['phi_ac', 'phi_dc', 'I_c', 'L_s', 'C'].
        """

        def func(x, f0, R0, names):

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
            current_real_impedance = self.impedance(current_f0, R0=R0).real

            if BW is not None:
                current_BW = self.find_reflection_fwhm()
                relative_error_BW = ((current_BW - BW)/BW*weight['BW'])**2.
            else:
                relative_error_BW = 0.

            y =  np.sum(((current_f0 - f0)/f0*weight['f0'])**2.\
                      + ((current_real_impedance + R0)/R0*weight['R0'])**2.\
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
                print '    real impedance  = '+str(round(current_real_impedance, 3))+' ohm, weight: '+str(weight['R0'])
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

        minimize(func,
                 values,
                 args=(f0, R0, names),
                 method=method,
                 bounds=bounds)

        # Create a list containing the result of the optimization
        result = [self.phi_ac, self.phi_dc, self.I_c, self.L_s, self.C]

        # In case the user don't want to update the instance attributes
        if not update_parameters:

            self.phi_ac = backups[0]
            self.phi_dc = backups[1]
            self.I_c    = backups[2]
            self.L_s    = backups[3]
            self.C      = backups[4]

        return result
