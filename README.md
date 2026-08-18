# EarthShineGen

A single-executable event generator for the dark Earthshine signal: dark matter
captured by the Earth annihilates near the core, the dark photons travel
outward, decay in the rock under the detector, and a pair of muons walks up
into it.

It merges two existing packages:

| from | what it contributes |
| --- | --- |
| [DarkCapPy](https://github.com/agree019/DarkCapPy) | the "cross section": capture rate, annihilation rate, Sommerfeld enhancement, dark photon decay length, and hence the absolute rate of muon pairs |
| [EarthShine](https://github.com/mattbellis/EarthShine) | the kinematics: decay points in the rock, the two-body decay, propagation through the overburden, the detector geometry |

The detector is described entirely by the parameter card, so nothing here is
tied to a particular experiment. The output is a Les Houches event file, so it
drops into the standard gridpack path the same way BlackMax and Charybdis do.

## Quick start

```bash
# any environment with numpy and scipy will do
./EarthShineGen --write-card parameter.txt
./EarthShineGen                              # uses ./parameter.txt
./EarthShineGen --m_X 10000 --n_events 50000 # command line overrides the card
./EarthShineGen --report-only                # rate only, no events
```

Two files come out: `events.lhe` and `earthshinegen_report.txt`.

## What it computes

```
kappa_0(m_X)                                    capture, stripped of m_A, epsilon
C_cap    = eps^2 alpha_X kappa_0 / m_A^4        captures per second
C_ann                                           annihilation, with Sommerfeld
Gamma_ann = C_cap tanh^2(t_earth/tau) / 2       annihilations per second today
L                                               dark photon decay length
f_shell                                         fraction decaying in the volume
BR(A' -> mu mu)                                 visible fraction
R = 2 Gamma_ann (A / 4 pi R_E^2) f_shell BR     muon pairs per second in the volume
R x eps_MC                                      observable rate at the detector
```

`eps_MC` is measured by the generator itself: the fraction of thrown decays
whose muons reach the detector and clear the momentum thresholds. The report
prints the whole chain and the cut flow.

## Differences from the packages it merges

These are deliberate. Each one is either a dependency the gridpack could not
carry, or a physics point worth raising.

**No TensorFlow.** EarthShine draws the decay with `phasespace`, which needs
TensorFlow. `A' -> mu+ mu-` is a two-body decay of a spinless resonance, so it
is done analytically here. The self-tests check four-momentum conservation, the
muon mass shell and the reconstructed `m_A`.

**No pandas, no parquet.** The gridpack runtime ships numpy, scipy and pandas
but not pyarrow, so parquet is unreadable inside a job. The Earth model and the
branching-ratio tables are read from csv with hand-rolled readers, and the only
runtime dependencies are numpy and scipy.

**kappa_0 is ~1000x faster.** DarkCapPy evaluates roughly 5000 two-dimensional
quadratures per mass point, which takes minutes. The recoil-energy integral of
`exp(-E_R/E_N)` is elementary, so it is done in closed form and the remaining
integral over the incident speed is vectorised across all 500 shells at once.
`test/validate_against_darkcappy.py` compares the two: they agree to 1 part in
10^7, in about 1 second instead of 2 minutes. `kappa_method dblquad` restores
the original if you want to check.

**The decay depth is sampled, not reweighted.** A dark photon surviving to
radius `r` has probability `exp(-r/L)`, so the density of decay points inside
the generation volume goes as `exp(+d/L)` in depth `d` below the detector: the
deep end is favoured, because a decay near the surface had to survive further.
EarthShineGen samples that directly (`depth_sampling exponential`, the
default), which is what makes this a merge of the two packages rather than a
concatenation. Sampling the profile also removes the need to throw decays on a
grid of depths and reweight between them, and with it the closure uncertainty
that interpolation costs. `depth_sampling uniform` reproduces the older
behaviour for comparison.

At small `epsilon`, `L` is ~10^5 km against a 4 km volume, so the exponential is
uniform to a part in 10^4. The difference only appears at large `epsilon`, where
`L` shrinks as `1/epsilon^2`.

**The branching ratio enters twice, and DarkCapPy only used it once.**
DarkCapPy's `decayLength(m_X, m_A, epsilon, BR)` is `gamma c BR / Gamma_ee`,
i.e. the *total* decay length written in terms of the electron partial width,
so its `BR` argument should be `BR(A' -> e e)`. The rate notebook
(`generate_rates.ipynb`) passes `BR(A' -> mu mu)` there instead, and then does
not multiply the yield by the dimuon branching ratio at all. Below the dimuon
threshold `BR(ee) = 1` and neither choice mattered; at `m_A = 0.23 GeV`,
`BR(ee) = 0.64` and `BR(mumu) = 0.24`, so it does.

The default `decay_length_convention total` uses `BR(ee)` for the width and
multiplies the yield by `BR(mumu)`. `decay_length_convention darkcappy`
reproduces the notebook exactly, for comparison against the existing numbers.
**This is worth a look before the next round of limits**; it is flagged here
rather than silently changed.

**The rate normalisation is core-model only.** `Gamma_ann` is the annihilation
rate of dark matter that has thermalised into the centre of the Earth, and the
`A / 4 pi R_E^2` factor is the solid angle the generation volume subtends from
there. `dm_model floating` and `dm_model monoenergetic` change the kinematics
but not the normalisation, which DarkCapPy does not provide for a crust
distribution. The generator says so on stderr when you use them; treat the
quoted rate as core-model.

**Muons are on shell.** EarthShine's `momentum_constrained` model rescales the
whole four-vector, which puts the muon off its mass shell by
`O(m_mu^2 / p^2)`. Here the energy is recomputed from the rescaled momentum,
because an LHE record with an inconsistent mass is rejected downstream. The
difference is parts in 10^8.

**Energy loss.** `eloss_model running` (the default) uses the PDG 2004
standard-rock coefficients with `a` and `b` running with `log10 E`. That is
bit-for-bit the parameterisation the standard cosmic muon generators use, so
samples compare like for like against existing single-muon productions.
`constant` uses fixed coefficients. EarthShine's GEANT4-derived splines are not
carried: they need a large pickle, and the published acceptance numbers use the
average loss.

**Multiple scattering is available, and it is not a small correction.**
EarthShine does not simulate it at all. `ms_model highland` does; see the next
section for why it matters and why it is nevertheless off by default.

## Multiple scattering

`ms_model` is `none` by default, so existing samples reproduce bit for bit.
`highland` turns on multiple Coulomb scattering in the overburden.

The RMS projected angle is integrated along the path rather than evaluated at a
single momentum, because the muon softens as it goes and the scattering piles up
at the far end:

```
sigma^2 = (1 + 0.038 ln(L/X0))^2  int_0^L (13.6 MeV / (beta p(x)))^2 dx / X0
```

`p(x)` comes from whichever `eloss_model` is in force, so the two cannot drift
apart. `rock_radiation_length` sets `X0` and defaults to standard rock,
26.54 g/cm^2.

The trajectory gets one kink rather than a stepped transport, placed at
`1 - 1/sqrt(3)` of the way to the hand-off surface. A single deflection at that
fraction reproduces both the RMS arrival angle and the RMS lateral offset of
scattering spread continuously along the path, and it keeps the whole batch
vectorised. Finding the kink needs the path length, and the kink then changes
the path length, so the geometry is done in two passes.

### How big it is

On the default card, `m_X = 7000 GeV`, for the muons that get accepted:

| | median | 90th pct |
| --- | --- | --- |
| path in rock | 722 m | 1335 m |
| `sigma_theta` per muon | 0.65 mrad | 2.8 mrad |
| lateral offset at arrival | 0.21 m | 0.95 m |

Two scales to compare against. The detector subtends about 10 mrad from a 722 m
path, so the acceptance moves by a few percent: 0.0338 to 0.0310, an 8% relative
drop, at that parameter point. But the `A'` opening angle is only
`m_A/E` ~ 0.2 mrad, so the scattering is several times larger than the decay
itself, and the pair invariant mass at the hand-off surface goes from

```
ms_model none        median 0.225 GeV     99.8% below 1 GeV
ms_model highland    median 1.381 GeV     32.5% below 1 GeV
```

The resonance does not survive the rock. If you are fitting or cutting on the
dimuon mass of what arrives, this is the difference between a narrow peak and a
broad continuum, and it is the reason the option exists.

The same thing shows up as the separation between the two muons where they
cross the hand-off surface, after a median flight of ~700 m:

| separation | 10% | median | 90% | 99% |
| --- | --- | --- | --- | --- |
| `none` | 0.3 cm | 1.7 cm | 3.9 cm | 9.5 cm |
| `highland` | 2.6 cm | 54.5 cm | 3.76 m | 10.7 m |

A factor of 31 on the median. Without scattering every pair lands inside 10 cm;
with it, only 22% do and a third are more than a metre apart. As an angle that
is 0.025 mrad against 0.835 mrad, the latter being the expected `sqrt(2) *
sigma_theta` for two independently scattered muons.

This is what breaks the single-vertex approximation: `#vertex` at the midpoint
of the two crossings is a good stand-in for both muons at 2 cm and a poor one
at half a metre. A GEN-SIM job running with `ms_model highland` should read
`#vertex_mu1` and `#vertex_mu2` rather than `#vertex`.

The table comes from running the generator twice at the same seed and reading
the two crossing points back out of the `#vertex_mu1` and `#vertex_mu2` comment
lines.

### What it does not do

Highland is the Gaussian core. At the thousands of radiation lengths involved
the single-hard-scatter tail is not a small correction to it, so treat the
result as a floor on the deflection rather than a full description. That is why
the default is `none`: the numbers above should be checked against a GEANT4
reference before anything downstream depends on them.

Turning it on also makes the reported acceptance depend on the scattering model
rather than on geometry alone. The report says so when `ms_model` is not `none`.

`stage vertex` momenta are untouched either way, since the scattering happens
after the decay; but it still changes which decays are accepted.

## The two stages

`stage vertex` writes the decay itself: the vertex is the true `A'` decay point
in the rock and the muon momenta are undegraded. Exact, and the right input for
an acceptance study that does its own geometry.

`stage detector` writes what arrives: the muon momenta after the energy loss
over their path through the rock, positioned where they cross the hand-off
surface. That is the same kind of target surface a cosmic muon generator uses,
which makes this a drop-in replacement for fixed-momentum single-muon samples,
with the correct pair kinematics and the correct absolute rate.

Two caveats for the `detector` stage:

* LHE has one vertex per event, and the two muons cross the surface at two
  different points. The event vertex is written at their midpoint, and all
  three positions plus the decay point go into comment lines that
  `LHEEventProduct::comments()` preserves. See `gridpack/run3_fragment.py` for
  what a vertex producer needs to do with them. **A GEN-SIM job that does not
  read them will put the muons at the interaction point flying outward, which
  is not the signal.**
* The `A'` line in the record is the reconstructed pair four-vector. After the
  two muons lose different amounts of energy in the rock, that is no longer an
  on-shell 0.23 GeV dark photon. The record stays momentum-conserving; the mass
  field simply reports the invariant mass of what arrived. Use `stage vertex`
  if you want the resonance.

## Describing the detector

The detector is three coaxial cylinders centred on the origin with their axis
along `z`, in the frame where `+y` is up. Every size is a card parameter, and
nothing about the geometry is hard-coded:

| parameter | default [m] | what it is |
| --- | --- | --- |
| `detector_radius` | 7.5 | outer cylinder radius |
| `detector_half_length` | 15.0 | outer cylinder half length |
| `inner_detector_radius` | 1.0 | inner cylinder radius |
| `inner_detector_half_length` | 2.5 | inner cylinder half length |
| `handoff_radius` | `auto` | radius of the surface the muons are delivered on |
| `handoff_half_length` | `auto` | half length of that surface |

`require_hit` picks which cylinder a muon has to cross to be accepted:
`detector`, `inner_detector` or `none`.

The hand-off surface defaults to `auto`, meaning the outer cylinder itself, so
resizing the detector moves the delivery surface with it and there is only one
geometry to keep consistent:

```bash
./EarthShineGen --detector_radius 30 --detector_half_length 60 --depth_max -40
```

Set `handoff_radius` and `handoff_half_length` explicitly if you want the muons
delivered on a surface standing off from the detector. The card checks that the
three surfaces nest (`inner <= detector <= hand-off`) and that the generation
volume starts below the hand-off surface, and refuses the run if they do not.

The defaults describe a general-purpose collider detector. They are only
defaults.

## Parameter card

`./EarthShineGen --write-card parameter.txt` writes every parameter at its
default with a one-line explanation. It is a `key  value` file, not a
positional one, so the gridpack scripts edit it by name:

```bash
sed -i "s|^m_X .*|m_X  10000|" parameter.txt
```

Every key is also a command-line option, and the command line wins over the
card. `./EarthShineGen --help` lists them.

## Gridpacks

Same shape as the BlackMax and Charybdis gridpacks in
`~/BlackHole/test_exo_bh_gridpacks`. Nothing is compiled, so the build is quick.

```bash
cd gridpack

# generic: the parameter point is passed at run time through the fragment
./earthshinegen_gridpack.sh run3

# specific: the point is baked into parameter.txt
./earthshinegen_gridpack.sh run3 7000 0.23 1e-8 max core
```

That produces `earthshinegen_gridpack_run3.tar.xz` containing `runcmsgrid.sh`
and the `EarthShineGen/` tree. `gridpack/earthshinegen.py` is the
`ExternalLHEProducer` snippet for the generic case and
`gridpack/run3_fragment.py` is the GEN fragment.

The build runs a ten-event smoke test before packing.

## Tests

```bash
./test/run_tests.py                     # numpy + scipy only
./test/validate_against_darkcappy.py    # needs pandas and a DarkCapPy checkout
```

The comparison script looks for DarkCapPy in `$DARKCAPPY_DIR`, falling back to
`../DarkCapPy`; `--darkcappy PATH` overrides both.

`run_tests.py` covers the decay kinematics, the depth sampling against its
analytic mean, the energy loss against the EarthShine reference, the geometry,
the detector sizing and its validation, the rate scaling, the LHE record, and
an end-to-end run in every model and stage. `validate_against_darkcappy.py` compares the Earth model shell by shell
and every rate quantity against the DarkCapPy package itself.

## Layout

```
EarthShineGen              the executable
parameter.txt              the default card
earthshinegen/
  card.py                  parameter card and command line
  constants.py             physical constants and conversions
  planet.py                PREM Earth model, velocity distribution, branching ratios
  darkphoton.py            capture, annihilation, decay  (ports DarkPhoton.py)
  rate.py                  the merged rate chain and the report
  kinematics.py            dark photon production and the two-body decay
  geometry.py              decay points and the detector cylinders
  eloss.py                 muon energy loss in the overburden
  lhe.py                   Les Houches output
  generator.py             the event loop
data/
  PREM500_Mod.csv          Earth structure and composition
  EarthDMVelDist.csv       dark matter velocity distribution
  br_mumu.csv              BR(A' -> mu mu), digitised from Feng et al.
  br_ee.csv                BR(A' -> e e)
gridpack/                  gridpack build, runcmsgrid drivers, CMSSW fragments
test/                      self-tests and the DarkCapPy comparison
```

## References

### Upstream packages

* DarkCapPy — https://github.com/agree019/DarkCapPy
* EarthShine — https://github.com/mattbellis/EarthShine

### Papers

* Feng, Smolinsky, Tanedo, *Dark photons from the centre of the Earth*,
  [arXiv:1509.07525](https://arxiv.org/abs/1509.07525)
* Green, Rentala et al., DarkCapPy, [arXiv:1812.07573](https://arxiv.org/abs/1812.07573)
* Leane et al., [arXiv:2209.09834](https://arxiv.org/abs/2209.09834) (the crust,
  or "floating", distribution)
