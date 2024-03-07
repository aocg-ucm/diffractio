   def draw_XYZ_deprecated(self,
                 kind='intensity',
                 logarithm=False,
                 normalize='',
                 pixel_size=(128, 128, 128)):
        """Draws  XZ field.

        Parameters:
            kind (str): type of drawing: 'intensity', 'phase', 'real_field'
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            pixel_size (float, float, float): pixels for drawing
            """
        try:
            from .utils_slicer_deprecated import slicerLM
            is_slicer = True
        except ImportError:
            print("slicerLM is not loaded.")
            is_slicer = False

        if is_slicer:
            u_xyz_r = self.cut_resample(num_points=(128, 128, 128),
                                        new_field=True)

            if kind == 'intensity' or kind == '':
                drawing = np.abs(u_xyz_r.u)**2
            if kind == 'phase':
                drawing = np.angle(u_xyz_r.u)
            if kind == 'real_field':
                drawing = np.real(u_xyz_r.u)

            if logarithm == 1:
                drawing = np.log(drawing**0.5 + 1)

            if normalize == 'maximum':
                factor = max(0, drawing.max())
                drawing = drawing / factor

            slicerLM(drawing)
        else:
            return
        
        
    def draw_volume_deprecated(self, logarithm=0, normalize='', maxintensity=None):
        """Draws  XYZ field with mlab

        Parameters:
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            maxintensity (float): maximum value of intensity

        TODO: Simplify, drawing
            include kind and other parameters of draw
        """

        try:
            from mayavi import mlab
            is_mayavi = True
        except ImportError:
            print("mayavi.mlab is not imported.")
            is_mayavi = False

        if is_mayavi:

            intensity = np.abs(self.u)**2

            if logarithm == 1:
                intensity = np.log(intensity + 1)

            if normalize == 'maximum':
                intensity = intensity / intensity.max()
            if normalize == 'area':
                area = (self.y[-1] - self.y[0]) * (self.z[-1] - self.z[0])
                intensity = intensity / area
            if normalize == 'intensity':
                intensity = intensity / (intensity.sum() / len(intensity))

            if maxintensity is None:
                intMin = intensity.min()
                intMax = intensity.max()
            else:
                intMin = maxintensity[0]
                intMax = maxintensity[1]

            mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
            mlab.clf()
            source = mlab.pipeline.scalar_field(intensity)
            mlab.pipeline.volume(source,
                                 vmin=intMin + 0.1 * (intMax - intMin),
                                 vmax=intMin + 0.9 * (intMax - intMin))
            # mlab.view(azimuth=185, elevation=0, distance='auto')
            print("Close the window to continue.")
            mlab.show()
        else:
            return

    def draw_refractive_index_deprecated(self, kind='real'):
        """Draws XYZ refraction index with slicer

        Parameters:
            kind (str): 'real', 'imag', 'abs'
        """
        try:
            from .utils_slicer_deprecated import slicerLM
            is_slicer = True
        except ImportError:
            print("slicerLM is not loaded.")
            is_slicer = False

        try:
            from mayavi import mlab
            is_mayavi = True
        except ImportError:
            print("mayavi.mlab is not imported.")
            is_mayavi = False

        if is_slicer:

            print("close the window to continue")
            if kind == 'real':
                slicerLM(np.real(self.n))
            elif kind == 'imag':
                slicerLM(np.imag(self.n))
            elif kind == 'abs':
                slicerLM(np.abs(self.n))
        else:
            return