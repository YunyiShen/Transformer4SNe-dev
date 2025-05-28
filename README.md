# transceiver
Perceiver for data from transients like supernovae.

### What is this?
This is a boilerplate package for transformers specialized for data from transients like supernovae: photometry, spectra, and host images (coming soon). Hence the name `transceiver` (transient perceiver). We’ve tried to keep it minimal — currently, it depends only on torch and Python’s built-in math module.

### How do I play with it?
You can install it using: `pip install ./package` after cloning this repo and `cd` into it.  
The main modules are `transceiver.PhotometricLayers` for photometry and `transceiver.SpectraLayers` for spectra and `transceiver.ImageLayers` for image (e.g. host galaxy). We plan to add some data modules in the near future. 
