"""EarthShineGen: dark Earthshine dimuon event generation.

DarkCapPy supplies the absolute rate (capture, annihilation, decay length),
EarthShine supplies the kinematics (decay points in the rock, the two-body
decay, propagation through the overburden, the detector geometry).  This
package merges the two and writes Les Houches events.

The detector is described entirely by the parameter card -- an outer and an
inner cylinder plus the hand-off surface, all with configurable radius and half
length -- so nothing here is tied to a particular experiment.
"""

__version__ = '1.0.0'
