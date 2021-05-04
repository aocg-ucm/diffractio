.. highlight:: shell

============
Installation
============


Stable release
--------------

To install Python diffraction and interference, run this command in your terminal:

.. code-block:: console

	# Linux:
	$ pip3 install diffractio

	# Windows:
	$ pip install diffractio


This is the preferred method to install Python diffraction and interference, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


Additional packages
------------------------

Diffractio uses also the following non-standard modules:

* py-pol

In some schemes, the following modules are also required:

* screeninfo
* PIL
* mayavi, traits, tvtk
* python-opencv

They should previously be installed before Diffractio module.


From sources
------------

The sources for Python diffraction and interference can be downloaded from the `Bitbucket repo`_.

You can either clone the public repository:

.. code-block:: console

	$ git clone git@bitbucket.org:optbrea/diffractio.git
	$ git clone https://optbrea@bitbucket.org/optbrea/diffractio.git



Once you have a copy of the source, you can install it with:

.. code-block:: console

	# Linux:
	$ python3 setup.py install

	# Windows:
	$ python setup.py install



.. _Bitbucket repo: https://bitbucket.org/optbrea/diffractio/src/master/
