# Molecular Dynamics with Unity ECS

This is a project that implements simple molecular dynamics of soft spheres using Unity's new job system. Currently only harmonic spheres are implemented. Gradient descent, NVE, and NVT ensembles can be simulated. NVT is implemented using a Nose-Hoover thermostat. The number of particles, the size of the simulation region, and the interaction scale can all be played with. Currently, the code only works with the version of Unity 2018.12b that contains the beta of the job system (see ). A lot of stuff could be added. A particularly notable absense is the lack of variable masses / particle sizes. I can comfortably simulate ~100k particles as well as all of their interactions on my desktop.

See a video [here.](https://www.youtube.com/watch?v=366SbR28ejM&feature=youtu.be)


## Architecture

The basic architecture involves a number of steps and essentially revolves around the hashing of positions into a dense grid as a means of spatially partitioning. This allows us to split the force computation (which is by far the most expensive part of the computation) so that each job operates on a single cell of the grid.

The main steps that are common to all methods of simulation are:

1) Hash Particle Positions: This job computes the hash (just the cell index in the grid) and puts copies data into a number of NativeMultiHashMaps.

2) Compute Forces and Energies: This job is split over the different cells and computes the forces on each particle in the cell due to its neighbors. Here each thread can read from any cell but can only write to the cell being operatored on.

3) Unhash Forces: This job copies the forces from the NativeMultiHashMap to a NativeArray that shares the correct indices with the position and velocity data.

Once the forces are computed it is straight forward to use standard molecular dynamics techniques to do the simulation itself.
