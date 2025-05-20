# Transformer4SNe
Transformers for data from supernova science

### What is this?
This is a boilerplate package for transformers specialized for data from supernova science: photometry, spectra, and host images (coming soon). Hence the name transformer4sne. We’ve tried to keep it minimal — currently, it depends only on torch and Python’s built-in math module.

### How do I play with it?
You can install it using: `pip install ./package` after cloning this repo and `cd` into it.  
The main modules are `transformer4sne.PhotometricLayers` for photometry and `transformer4sne.SpectraLayers` for spectra. We plan to add `transformer4sne.ImageLayers` for image (e.g. host galaxy) in the near future.
