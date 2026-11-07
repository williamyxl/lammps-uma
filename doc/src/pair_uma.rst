.. index:: pair_style uma

pair_style uma command
======================

Syntax
""""""

.. code-block:: LAMMPS

   pair_style uma keyword value ...

* zero or more keyword/value pairs may be appended

  .. parsed-literal::

     keyword = *precision* or *devices* or *virial* or *ac* or *decomp* or *neigh*
       *precision* value = *double*
         *double* = FP64 positions, energy, and forces (required)
       *devices* value = N
         N = number of GPU devices for same-node graph-parallel workers (default 1)
       *virial* value = *auto* or *yes* or *no*
         *auto* = enable strain-gradient virial when a barostat is present (default)
         *yes* = always request the virial
         *no* = never compute the virial (NVE/NVT only)
       *ac* value = *off* or *chunk* or *block* or *full*
         *off* = no activation checkpointing (default; required for NPT)
         *chunk*, *block*, *full* = increasing recompute / lower HBM
       *decomp* value = *graph* or *spatial*
         *graph* = graph-parallel over MPI, one device per rank (default)
         *spatial* = LAMMPS spatial domain decomposition with a halo
       *neigh* value = *lammps* or *engine*
         *lammps* = consume the LAMMPS neighbor list (default)
         *engine* = rebuild the graph inside the engine (debug / triclinic)

Examples
""""""""

.. code-block:: LAMMPS

   pair_style uma precision double
   pair_coeff * * /path/to/uma-s-1p2-omat-f64 Na Cl
   newton off

   pair_style uma precision double devices 1 virial auto ac off decomp graph
   pair_coeff * * ${uma_art} Na Cl
   newton off

Example input scripts (NVT and NPT NaCl) are in ``examples/PACKAGES/uma``.
They require a user-exported TorchScript artifact; UMA weights are not
shipped with LAMMPS.

Description
"""""""""""

Style *uma* evaluates Meta FAIR's Universal Models for Atoms (UMA)
:ref:`(Wood25) <Wood25>` as a native LAMMPS pair style.  Energies and
forces come from a GPU-persistent C++ LibTorch engine
(``src/ML-UMA/uma-engine``).  Forces are obtained by autograd through a
TorchScript energy module.  This is not FairChem's Python ``fix
external`` / ASE calculator path, and it is not a Kokkos accelerator
variant.  There is no ``uma/kk`` style.

Only a single :doc:`pair_coeff <pair_coeff>` command is used.  It names
the TorchScript **artifact directory** (not a Hydra ``.pt`` checkpoint)
followed by N element names, where N is the number of LAMMPS atom types:

* artifact directory
* N element names = mapping of UMA elements to atom types

The artifact directory is produced once, offline, from a user-supplied
FairChem UMA checkpoint with
``src/ML-UMA/uma-engine/python/export_artifact.py`` (``--dtype float64``,
typically ``--task omat``).  LAMMPS does not import FairChem at run time.
The only extra-repo files are that checkpoint at **export** time and the
artifact directory at **run** time.

The pair style and uma-engine C++ are GNU GPL-2, the same as LAMMPS.
Vendored third-party headers: vesin (BSD-3-Clause).  nlohmann JSON (MIT)
is the LAMMPS-bundled copy in ``src/nlohmann/``.  LibTorch / PyTorch and
fairchem-core are user-installed and are not shipped here.

This package does **not** redistribute UMA model weights.  Checkpoints
such as ``uma-s-1p2.pt`` (Hugging Face ``facebook/UMA``) and exported
TorchScript artifacts remain under the FAIR Chemistry License, which is
separate from LAMMPS GPL-2 and from FairChem's MIT software license.
Users must obtain and accept those terms themselves.  See
``src/ML-UMA/LICENSE``.

As an example, a two-type NaCl system uses:

.. code-block:: LAMMPS

   pair_style uma precision double
   pair_coeff * * /path/to/uma-s-1p2-omat-f64 Na Cl
   newton off

----------

See the :doc:`pair_coeff <pair_coeff>` page for alternate ways to specify
the path for the artifact directory.

Precision
"""""""""

Only ``precision double`` is supported.  The pair style and the
TorchScript artifact must both be FP64.  ``precision mixed`` is refused.
Do not enable TF32, mixed precision, or ``torch.compile`` for this style.

The *devices* keyword selects same-node graph-parallel **fork** workers
(one MPI rank, N devices).  Multi-node / multi-rank runs use **one MPI
rank per GPU** with ``devices 1`` (the default).  Combining ``devices`` >
1 with more than one MPI rank is an error.

The *ac*, *virial*, *decomp*, and *neigh* keywords are the only runtime
API.  Process environment variables are not read for these knobs.

Mixing, shift, table, tail correction, restart, rRESPA info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This pair style does not support the :doc:`pair_modify <pair_modify>`
mix, shift, table, and tail options.

This pair style does not write its information to :doc:`binary restart
files <restart>`, since it is stored in the artifact directory.  Thus,
you need to re-specify the pair_style and pair_coeff commands in an
input script that reads a restart file.

This pair style can only be used via the *pair* keyword of the
:doc:`run_style respa <run_style>` command.  It does not support the
*inner*, *middle*, *outer* keywords.

Restrictions
""""""""""""

This pair style is part of the ML-UMA package.  It is only enabled if
LAMMPS was built with that package.  See the :doc:`Build package
<Build_package>` page for more info.  Building ML-UMA requires CMake and
LibTorch; see the :ref:`ML-UMA extras <ml-uma>` notes.

This pair style requires :doc:`newton <newton>` pair to be *off*, and
requires atom IDs.

The pair style does not implement ``single()`` and does not yet support
per-atom energy or per-atom virial.

A global virial is not computed on the default energy/force path.  With
*virial auto* (the default), a barostat (for example :doc:`fix npt
<fix_nh>`) auto-enables a strain-gradient virial on a single tile and on
the graph-parallel path, provided *ac off*.  *decomp spatial* does not
compute a virial; a barostat is refused there.  *ac chunk/block/full* is
incompatible with the virial; use NVE/NVT or *ac off*.

Related commands
""""""""""""""""

:doc:`pair_style pace <pair_pace>`, :doc:`pair_style uf3 <pair_uf3>`,
:doc:`pair_style quip <pair_quip>`

Default
"""""""

precision = double, devices = 1, virial = auto, ac = off, decomp = graph, neigh = lammps

----------

.. _Wood25:

**(Wood25)** Wood, Dzamba, Fu, Gao, Shuaibi, et al., arXiv:2506.23971 (2025).
