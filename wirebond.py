# This Python file uses the following encoding: utf-8

import numpy as np
from multiprocessing import Pool
import itertools

from KLJPA import KLJPA


from klopfenstein_discretization import (wb_reflection_discretization,
                                         wb_external_discretization,
                                         wb_ljpa_external_discretization)

class Wirebond(KLJPA):



    def __init__(self, C, L_s, I_c, phi_s, phi_dc, phi_ac, theta_p,
                       Z_l, l, g_m, L_l, C_l, L_b,
                       theta_s=0., f_p=None):
        """
        A wirebond placed in front of the KLJPA

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
        L_b : float
            Wirebond inductance in henry
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

        if not isinstance(L_b, float):
            raise ValueError('L_b parameter must be float type.')

        KLJPA.__init__(self, C, L_s, I_c, phi_s, phi_dc, phi_ac, theta_p,
                           Z_l, l, g_m, L_l, C_l, theta_s, f_p)

        self.L_b = L_b



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
                                     as_theory, self.f_p, self.L_b)
        result = wb_ljpa_external_discretization(param)

        return result



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

            return wb_external_discretization((self.f_p - f, z, prod, self.zl,
                                            self.C, self.L_s, self.I_c, self.phi_s,
                                            self.phi_dc, self.phi_ac, self.theta_p,
                                            self.theta_s, as_theory, simple_ext,
                                            self.f_p, self.L_b))
        else:

            # Create a pool a thread for fast computation
            # We look at the impedance at the pump frequency
            pool = Pool()
            result = pool.map(wb_external_discretization,
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
                                             itertools.repeat(self.f_p),
                                             itertools.repeat(self.L_b)))

            pool.close()
            pool.join()

            return np.array(result)




    def reflection(self, f, n=1e2, as_theory=False, simple_ext=False):
        """
        Return the reflection of the circuit.

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

            return wb_reflection_discretization((f, z_ext, z, prod, self.zl, self.C,
                                              self.L_s, self.I_c, self.phi_s,
                                              self.phi_dc, self.phi_ac, self.theta_p,
                                              self.theta_s, as_theory, self.f_p,
                                              self.L_b))
        else:

            # If the pump frequency is None, we don't have to calculate the impedance
            # seen by the pumpistor
            if self.f_p is not None:
                z_ext = self.external_impedance(f, n, 50., as_theory, simple_ext)
            else:
                z_ext = itertools.repeat(None)

            # Create a pool a thread for fast computation
            pool = Pool()
            result = pool.map(wb_reflection_discretization,
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
                                             itertools.repeat(self.f_p),
                                             itertools.repeat(self.L_b)))

            pool.close()
            pool.join()

            # If the user a np.ndarray frequency, we return a np.ndarray
            return np.array(result)
