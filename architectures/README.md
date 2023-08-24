# Architectures
This folder contains all the architectures we have implemented and tested.
We have adapted these to work with the `timm` library and follow our [interface](../resizing_interface.py) for resizing of inputs and outputs and for additional loss terms.
Most architectures build on our implementation of the [ViT](vit.py) and only change the attention mechanism.

Whenever we have adapted other peoples code, the original is linked in the respective file; licenses can be found [here](../licenses).

