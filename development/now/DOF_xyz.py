def get_DOF(self, z, num_processors, verbose=False, has_draw=False):
    """
    u_xyz = Scalar_field_XYZ(self.x, self.y, z, self.wavelength)

    Get depth of focus of the lens in a certain interval z measuring the beam width.
        It also determines the field at positions z

        Parameters:
            z (numpy.array): positions z where focus is placed
            num_procesors (int): number of processors for computation.
            verbose (bool): if True, prints interesting information about the computation.
            has_draw (bool): if True, draws the depth of focus.

        Returns:
            width (numpy.array): beam width at positions given at z.
            u_xyz (Scalar_field_XYZ): fields at positions z.
        """

    u_xyz.incident_field(self)
     u_xyz.RS(verbose, num_processors)

      width, _, _ = u_xyz.beam_widths(False)

       if has_draw is True:
            plt.figure()
            plt.plot(z / mm, width, 'k--', label='lens')
            plt.ylim(bottom=0)

            plt.xlabel("z (mm)")
            plt.ylabel("beam width $\omega$ ($\mu$m)")
            plt.xlim(z[0] / mm, z[-1] / mm)

        return width



    def draw_comparison(self,
                        z0,
                        widths_lens,
                        widths_star,
                        focals,
                        coverture=2,
                        ylims=(-100, 100),
                        filename=''):
        """Draws comparison of daisy_lens with standard lens

        Parameters:
            z0 (numpy.array): positions of z0
            widths_lens (numpy.array): beam width for standard lens
            widths_star (numpy.array): beam width for star lens
            focals (float, float): f0 and f1 focals for definiton of star lens
            coverture (float): coverture in f_incr for drawing
        """
        f0, f1 = focals
        f_mean = (f0 + f1) / 2
        f_incr = (f1 - f0) / 2

        plt.figure(figsize=(12, 6))
        plt.plot(z0 / mm, widths_lens / 2, 'r--', lw=4, label='standard')
        plt.plot(z0 / mm, widths_star / 2, 'k', lw=4, label='star lens')
        plt.plot(z0 / mm, -widths_lens / 2, 'r--', lw=4)
        plt.plot(z0 / mm, -widths_star / 2, 'k', lw=4)
        plt.xlabel("$z (mm)$", fontsize=20)
        plt.ylabel("$\omega (\mu m)$", fontsize=20)
        plt.legend()
        if coverture > 0:
            plt.xlim((f_mean - coverture * f_incr) / mm,
                     (f_mean + coverture * f_incr) / mm)
        plt.ylim(ylims)
        if not filename == '':
            plt.savefig(
                filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
