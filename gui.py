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

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
from matplotlib.backends.backend_gtk3 import NavigationToolbar2GTK3 as NavigationToolbar
from gi.repository import Gtk
from scipy.optimize import minimize

from LJPA import LJPA
from KLJPA import KLJPA

class Window(object):

    def __init__(self):


        self.builder = Gtk.Builder()
        self.builder.add_from_file('gui.glade')
        self.window = self.builder.get_object('window1')
        self.builder.connect_signals(self)
        figure_box = self.builder.get_object('scrolledwindow1')

        self.builder.get_object('spinbuttonphi_s').set_increments(0.005, 0.1)
        self.builder.get_object('spinbuttonphi_p').set_increments(0.005, 0.1)
        self.builder.get_object('spinbuttonL_b').set_increments(0.001, 0.1)
        self.simple_ext = False
        self.freeze_plot = False

        self.paramp = LJPA(I_c=self.builder.get_object('spinbuttonI_c').get_value()*1e-6,
                           phi_dc=self.builder.get_object('spinbuttonphi_dc').get_value(),
                           phi_ac=self.builder.get_object('spinbuttonphi_p').get_value(),
                           phi_s=self.builder.get_object('spinbuttonphi_s').get_value(),
                           theta_p=self.builder.get_object('spinbuttontheta_p').get_value(),
                           C=self.builder.get_object('spinbuttonC').get_value()*1e-12,
                           L_s=self.builder.get_object('spinbuttonL_s').get_value()*1e-12,
                           f_p=self.builder.get_object('spinbuttonf_p').get_value()*1e9)

        # Start of Matplotlib specific code
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(4, 1, sharex=True)
        # figure = Figure()#figsize=(8, 6), dpi=71)
        # self.ax = figure.add_subplot(111)

        f = self.get_frequency()
        z = self.get_impedance(f)
        re = np.real(z)
        im = np.imag(z)
        r = 20.*np.log10(abs((z - 50.)/(z + 50.)))

        self.line_r,  = self.ax1.plot(f/1e9, r)
        self.line_20, = self.ax1.plot([f[0]/1e9, f[-1.]/1e9], [20., 20.], 'r--')

        self.line_re, = self.ax2.plot(f/1e9, re)
        self.line_50, = self.ax2.plot([f[0]/1e9, f[-1.]/1e9], [-50., -50.], 'r--')

        self.line_im, = self.ax3.plot(f/1e9, im)
        self.line_0, = self.ax3.plot([f[0]/1e9, f[-1.]/1e9], [0., 0.], 'r--')

        self.line_abs, = self.ax4.plot(f/1e9, abs(z))
        self.line_00, = self.ax4.plot([f[0]/1e9, f[-1.]/1e9], [0., 0.], 'r--')

        self.ax1.set_ylabel('Gain [dB]')
        self.ax2.set_ylabel('Re(Z)')
        self.ax3.set_ylabel('Im(Z)')
        self.ax4.set_ylabel('|Z|')

        self.ax4.set_xlabel('Frequency [GHz]')

        self.canvas = FigureCanvas(self.fig)

        self.canvas.set_size_request(800, 600)
        figure_box.add_with_viewport(self.canvas)

        toolbar_box = self.builder.get_object('scrolledwindow2')
        toolbar = NavigationToolbar(self.canvas, self.window)
        toolbar_box.add_with_viewport(toolbar)

        self.window.show_all()

        Gtk.main()



    def get_frequency(self):

        return np.linspace(self.builder.get_object('spinbuttonf_s_start').get_value(),
                           self.builder.get_object('spinbuttonf_s_stop').get_value(),
                           self.builder.get_object('spinbuttonf_s_nb_point').get_value())*1e9


    def get_impedance(self, f):

        if self.builder.get_object('checkbutton3').get_active():

            return self.paramp.impedance(f, simple_ext=self.simple_ext)
        else:

            return self.paramp.impedance(f)


    def checkbutton1_toggled(self, checkbutton):

        if checkbutton.get_active():
            self.builder.get_object('box11').set_sensitive(True)
            self.builder.get_object('box20').set_sensitive(False)
            self.paramp.f_p = self.builder.get_object('spinbuttonf_p').get_value()*1e9
        else:
            self.builder.get_object('box11').set_sensitive(False)
            self.builder.get_object('box20').set_sensitive(True)
            self.paramp.f_p = None

        self.update_plot()



    def checkbutton3_toggled(self, checkbutton):

        if checkbutton.get_active():
            self.simple_ext = True
        else:
            self.simple_ext = False

        self.update_plot()



    def checkbutton2_toggled(self, checkbutton):

        if checkbutton.get_active():

            self.builder.get_object('box21').set_sensitive(True)
            self.builder.get_object('box25').set_sensitive(True)
            self.builder.get_object('box26').set_sensitive(True)
            self.builder.get_object('box27').set_sensitive(True)
            self.builder.get_object('checkbutton3').set_sensitive(True)
            self.builder.get_object('box31').set_sensitive(True)
            self.builder.get_object('box30').set_sensitive(True)
            self.builder.get_object('box29').set_sensitive(True)
            self.builder.get_object('box28').set_sensitive(True)
            self.builder.get_object('box13').set_sensitive(True)
            self.builder.get_object('box8').set_sensitive(True)
            self.builder.get_object('label35').set_sensitive(True)
            self.builder.get_object('label31').set_sensitive(True)
            self.builder.get_object('button1').set_sensitive(True)

            self.paramp = KLJPA(I_c=self.builder.get_object('spinbuttonI_c').get_value()*1e-6,
                               phi_dc=self.builder.get_object('spinbuttonphi_dc').get_value(),
                               phi_ac=self.builder.get_object('spinbuttonphi_p').get_value(),
                               phi_s=self.builder.get_object('spinbuttonphi_s').get_value(),
                               theta_p=self.builder.get_object('spinbuttontheta_p').get_value(),
                               C=self.builder.get_object('spinbuttonC').get_value()*1e-12,
                               L_s=self.builder.get_object('spinbuttonL_s').get_value()*1e-12,
                               f_p=self.builder.get_object('spinbuttonf_p').get_value()*1e9,
                                Z_l=self.builder.get_object('spinbuttonz_l').get_value(),
                                l=self.builder.get_object('spinbuttonl').get_value()*1e-2,
                                g_m=self.builder.get_object('spinbuttong_m').get_value(),
                                C_l=1.66e-10,
                                L_l=4.21e-7,
                                L_b=self.builder.get_object('spinbuttonL_b').get_value()*1e-9)

        else:

            self.builder.get_object('box21').set_sensitive(False)
            self.builder.get_object('box25').set_sensitive(False)
            self.builder.get_object('box26').set_sensitive(False)
            self.builder.get_object('box27').set_sensitive(False)
            self.builder.get_object('checkbutton3').set_sensitive(False)
            self.builder.get_object('box31').set_sensitive(False)
            self.builder.get_object('box30').set_sensitive(False)
            self.builder.get_object('box29').set_sensitive(False)
            self.builder.get_object('box28').set_sensitive(False)
            self.builder.get_object('box13').set_sensitive(False)
            self.builder.get_object('box8').set_sensitive(False)
            self.builder.get_object('label35').set_sensitive(False)
            self.builder.get_object('label31').set_sensitive(False)
            self.builder.get_object('button1').set_sensitive(False)

            self.paramp = LJPA(I_c=self.builder.get_object('spinbuttonI_c').get_value()*1e-6,
                               phi_dc=self.builder.get_object('spinbuttonphi_dc').get_value(),
                               phi_ac=self.builder.get_object('spinbuttonphi_p').get_value(),
                               phi_s=self.builder.get_object('spinbuttonphi_s').get_value(),
                               theta_p=self.builder.get_object('spinbuttontheta_p').get_value(),
                               C=self.builder.get_object('spinbuttonC').get_value()*1e-12,
                               L_s=self.builder.get_object('spinbuttonL_s').get_value()*1e-12,
                               f_p=self.builder.get_object('spinbuttonf_p').get_value()*1e9)

        self.update_plot()



    def spinbuttonC_changed(self, spinbutton):

        self.paramp.C = spinbutton.get_value()*1e-12

        self.update_plot()



    def spinbuttonI_c_changed(self, spinbutton):

        self.paramp.I_c = spinbutton.get_value()*1e-6

        self.update_plot()



    def spinbuttonL_s_changed(self, spinbutton):

        self.paramp.L_s = spinbutton.get_value()*1e-12

        self.update_plot()



    def spinbuttonf_p_changed(self, spinbutton):

        self.paramp.f_p = spinbutton.get_value()*1e9

        self.update_plot()



    def spinbuttonphi_p_changed(self, spinbutton):

        self.paramp.phi_ac = spinbutton.get_value()

        self.update_plot()



    def spinbuttontheta_p_changed(self, spinbutton):

        self.paramp.theta_p = spinbutton.get_value()

        self.update_plot()



    def spinbuttonphi_s_changed(self, spinbutton):

        self.paramp.phi_s = spinbutton.get_value()*1e-3

        self.update_plot()



    def spinbuttonphi_dc_changed(self, spinbutton):

        self.paramp.phi_dc = spinbutton.get_value()

        self.update_plot()



    def spinbuttonl_changed(self, spinbutton):

        self.paramp.l = spinbutton.get_value()*1e-2

        self.update_plot()



    def spinbuttonz_l_changed(self, spinbutton):

        self.paramp.zl = spinbutton.get_value()

        self.update_plot()



    def spinbuttong_m_changed(self, spinbutton):

        self.paramp.gm = spinbutton.get_value()

        self.update_plot()



    def spinbuttonL_b_changed(self, spinbutton):

        self.paramp.L_b = spinbutton.get_value()*1e-9

        self.update_plot()



    def spinbuttonf_s_nb_point_changed(self, spinbutton):

        self.update_plot()



    def spinbuttonf_s_start_changed(self, spinbutton):

        self.update_plot()



    def spinbuttonf_s_stop_changed(self, spinbutton):

        self.update_plot()



    def update_plot(self, x=None, y=None):

        if not self.freeze_plot:

            if x is not None and y is not None:
                f = x
                z = y
            else:
                f = self.get_frequency()
                z = self.get_impedance(f)

            re = np.real(z)
            im = np.imag(z)
            r = 20.*np.log10(abs((z - 50.)/(z + 50.)))

            self.line_r.set_data(f/1e9, r)
            self.line_20.set_data(f/1e9, r)

            self.line_re.set_data(f/1e9, re)
            self.line_50.set_data([f[0]/1e9, f[-1.]/1e9], [-50., -50.])

            self.line_im.set_data(f/1e9, im)
            self.line_0.set_data([f[0]/1e9, f[-1.]/1e9], [0., 0.])

            self.line_abs.set_data(f/1e9, abs(z))
            self.line_00.set_data([f[0]/1e9, f[-1.]/1e9], [0., 0.])

            for ax in (self.ax1, self.ax2, self.ax3, self.ax4):

                ax.relim()
                ax.autoscale_view()

            self.line_20.set_data([f[0]/1e9, f[-1.]/1e9], [20., 20.])
            self.canvas.draw()


    def optimization(self, button):

        self.iteration = 0
        def func(x, names, f):

            x = abs(x)

            for value, name in zip(x, names):

                if name == 'c':
                    self.paramp.C = value*1e-12
                if name == 'Ic':
                    self.paramp.I_c = value*1e-6
                if name == 'Ls':
                    self.paramp.L_s = value*1e-12
                if name == 'l':
                    self.paramp.l = value*1e-2
                if name == 'zl':
                    self.paramp.zl = value
                if name == 'gm':
                    self.paramp.gm = value*1e-1
                if name == 'lb':
                    self.paramp.L_b = value*1e-1
                if name == 'fp':
                    self.paramp.f_p = value*1e9
                if name == 'phi_ac':
                    self.paramp.phi_ac = value*1e-1
                if name == 'phi_dc':
                    self.paramp.phi_dc = value*1e-1

            z = self.get_impedance(f)

            re = np.real(z)
            im = np.imag(z)

            re_condition = np.sum(((re + 50.)/re)**2.)
            im_condition = np.sum((im/100.)**2.)

            y =  np.sum(((re + 50.)/re)**2. + (im/100.)**2.)
            self.iteration += 1

            print ''
            print 'Iteration: ', self.iteration
            for value, name in zip(x, names):
                if name == 'c':
                    print 'C: '+str(round(self.paramp.C*1e12, 3))
                if name == 'Ic':
                    print 'Ic: '+str(round(self.paramp.I_c*1e6, 3))
                if name == 'Ls':
                    print 'L_s: '+str(round(self.paramp.L_s*1e12, 3))
                if name == 'l':
                    print 'l: '+str(round(self.paramp.l*1e2, 3))
                if name == 'zl':
                    print 'zl: '+str(round(self.paramp.zl, 3))
                if name == 'gm':
                    print 'gm: '+str(round(self.paramp.gm, 3))
                if name == 'lb':
                    print 'L_b: '+str(round(self.paramp.L_b*1e9, 3))
                if name == 'fp':
                    print 'f_p: '+str(round(self.paramp.f_p*1e-9, 3))
                if name == 'phi_ac':
                    print 'phi_ac: '+str(round(self.paramp.phi_ac, 3))
                if name == 'phi_dc':
                    print 'phi_dc: '+str(round(self.paramp.phi_dc, 3))
            print 'Real part: '+str(round(re_condition, 3))+', imaginary part: '+str(round(im_condition, 3))+', least square: '+str(round(y, 3))
            print ''

            return y


        names  = ['c', 'Ic', 'Ls', 'l', 'zl', 'gm', 'lb', 'fp', 'phi_ac', 'phi_dc']
        values = [self.builder.get_object('spinbuttonC').get_value(),
                  self.builder.get_object('spinbuttonI_c').get_value(),
                  self.builder.get_object('spinbuttonL_s').get_value(),
                  self.builder.get_object('spinbuttonl').get_value(),
                  self.builder.get_object('spinbuttonz_l').get_value(),
                  self.builder.get_object('spinbuttong_m').get_value()*10.,
                  self.builder.get_object('spinbuttonL_b').get_value()*10.,
                  self.builder.get_object('spinbuttonf_p').get_value(),
                  self.builder.get_object('spinbuttonphi_p').get_value()*10.,
                  self.builder.get_object('spinbuttonphi_dc').get_value()*10.]
        bounds = ((2., 8.),
                  (2., 6.),
                  (20., 30.),
                  (0.5, 4.),
                  (3., 20.),
                  (0.01, 5),
                  (0.01, 20.),
                  (5., 20.),
                  (0.1, 9.),
                  (1., 4.))


        variables = []
        if self.builder.get_object('checkbutton_optimized_c').get_active():
            variables.append('c')
        if self.builder.get_object('checkbutton_optimized_Ic').get_active():
            variables.append('Ic')
        if self.builder.get_object('checkbutton_optimized_ls').get_active():
            variables.append('Ls')
        if self.builder.get_object('checkbutton_optimized_l').get_active():
            variables.append('l')
        if self.builder.get_object('checkbutton_optimized_zl').get_active():
            variables.append('zl')
        if self.builder.get_object('checkbutton_optimized_gm').get_active():
            variables.append('gm')
        if self.builder.get_object('checkbutton_optimized_lb').get_active():
            variables.append('lb')
        if self.builder.get_object('checkbutton_optimized_fp').get_active():
            variables.append('fp')
        if self.builder.get_object('checkbutton_optimized_phi_ac').get_active():
            variables.append('phi_ac')
        if self.builder.get_object('checkbutton_optimized_phi_dc').get_active():
            variables.append('phi_dc')

        names_to_optimized  = []
        values_to_optimized = []
        bounds_to_optimized = []

        for name, value, bound in zip(names, values, bounds):
            if name in variables:
                names_to_optimized.append(name)
                values_to_optimized.append(value)
                bounds_to_optimized.append(bound)

        f = np.linspace(self.builder.get_object('spinbutton1').get_value(),
                        self.builder.get_object('spinbutton2').get_value(),
                        self.builder.get_object('spinbutton3').get_value())*1e9

        self.freeze_plot = True
        results = minimize(func,
                           values_to_optimized,
                           args=(names_to_optimized, f),
                        #    method='TNC',
                           method='SLSQP',
                        #    method='L-BFGS-B',
                           bounds=bounds_to_optimized)
        values_optimized = results.x

        self.freeze_plot = False

        for value, name in zip(values_optimized, names_to_optimized):
            if name == 'c':
                self.builder.get_object('spinbuttonC').set_value(value)
            if name == 'Ic':
                self.builder.get_object('spinbuttonI_c').set_value(value)
            if name == 'Ls':
                self.builder.get_object('spinbuttonL_s').set_value(value)
            if name == 'l':
                self.builder.get_object('spinbuttonl').set_value(value)
            if name == 'zl':
                self.builder.get_object('spinbuttonz_l').set_value(value)
            if name == 'gm':
                self.builder.get_object('spinbuttong_m').set_value(value/10.)
            if name == 'lb':
                self.builder.get_object('spinbuttonL_b').set_value(value/10.)
            if name == 'fp':
                self.builder.get_object('spinbuttonf_p').set_value(value)
            if name == 'phi_ac':
                self.builder.get_object('spinbuttonphi_p').set_value(value/10.)
            if name == 'phi_dc':
                self.builder.get_object('spinbuttonphi_dc').set_value(value/10.)

        self.update_plot()
        print 'Done'
        return True




    def quit(self, event):

        Gtk.main_quit()

if __name__ == "__main__":
    a = Window()
