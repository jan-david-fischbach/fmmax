---
id: sources.amplitudes_for_fields
---

    
### `sources.amplitudes_for_fields`
Computes the amplitudes for fields.

The fields are expected to be given for a supercell containing one or more unit
cells, i.e. for a calculation with in-plane wavevector generated by
`basis.brillouin_zone_in_plane_wavevector`. The shape of the supercell is
inferred from the batch shape of the `layer_solve_result` and the specified
`brillouin_grid_axes`.

#### Args:
- **ex** (None): The x-oriented electric field on the real-space grid, with a trailing
batch dimension.
- **ey** (None): The y-oriented electric field on the real-space grid.
- **hx** (None): The x-oriented magnetic field on the real-space grid.
- **hy** (None): The y-oriented magnetic field on the real-space grid.
- **layer_solve_result** (None): The eigensolve result for the layer containing the fields.
- **brillouin_grid_axes** (None): The batch axes corresponding to the Brillouin zone grid.

#### Returns:
- **None**: The forward and backward wave amplitudes.