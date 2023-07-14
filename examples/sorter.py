"""An example which sets up a polarization-sorting metasurface component."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax

from fmmax import basis, fields, fmm, layer, scattering, utils

Aux = Dict[str, Any]
Initializer = Callable[[jax.random.PRNGKeyArray, Tuple[int, int]], jnp.ndarray]
Params = Dict[str, Any]

_DEFAULT_WAVELENGTH: jnp.ndarray = jnp.asarray([0.55])


class PolarizationSorterComponent:
    """Defines a polarization sorter component.

    The polarization sorter consists of a substrate subdivided into four quadrants,
    with a patterned sorter layer. The substrate and sorter are separated by a
    spacer, and the sorter is covered with a cap. Above the cap is the ambient.
    This is illustrated below.
                               __________________________
                              /                         /|
                             /                         //|
                            /                         ///|
                           /________________________ ////|
              ambient --> |_________________________|/// |
                  cap --> |_________________________|// /|
               sorter --> |_________________________|/ / |
               spacer --> |                         | /| |
                          |_________________________|/ |/
                          |            |            |  /
            substrate --> |            |            | /
                          |____________|____________|/

    The polarization sorter is illuminated with plane waves having four different
    linear polarizations. The response of the sorter (for each linear polarization)
    consists of the reflected power and the power transmitted into each quadrant.

    The polarization sorter is parameterized by the thicknesses of all its layers
    and the density of the sorter layer.
    """

    def __init__(
        self,
        ambient_permittivity: complex = (1.0 + 0.0j) ** 2,
        ambient_thickness: float = 1.0,
        cap_permittivity: complex = (1.5 + 0.0j) ** 2,
        cap_thickness: float = 0.0,
        sorter_permittivity_solid: complex = (4.0 + 0.0j) ** 2,
        sorter_permittivity_void: complex = (1.5 + 0.0j) ** 2,
        sorter_thickness: float = 0.15,
        spacer_permittivity: complex = (1.5 + 0.0j) ** 2,
        spacer_thickness: float = 1.0,
        substrate_permittivity: complex = (4.0730 + 0.028038j) ** 2,
        substrate_thickness: float = 1.0,
        wavelength: jnp.ndarray = _DEFAULT_WAVELENGTH,
        pitch: float = 2.0,
        approximate_num_terms: int = 200,
        truncation: basis.Truncation = basis.Truncation.CIRCULAR,
        fmm_configuration: fmm.FmmConfiguration = fmm.BASIC_CONFIGURATION,
        density_grid_shape: Tuple[int, int] = (128, 128),
        field_grid_shape: Tuple[int, int] = (128, 128),
        field_z_offset: float = 0.0,
        density_initializer: Initializer = lambda key, shape: jnp.ones(shape) * 0.5,
    ) -> None:
        """Initializes the `PolarizationSorterComponent`.

        Args:
            ambient_permittivity: Permittivity of the ambient.
            ambient_thickness: Thickness of the ambient layer.
            cap_permittivity: Permittivity of the cap.
            cap_thickness: Thickness of the cap layer.
            sorter_permittivity_solid: Permittivity of solid regions in the sorter.
            sorter_permittivity_void: Permittivity of void regions in the sorter.
            sorter_thickness: Thickness of the sorter layer.
            spacer_permittivity: Permittivity of the spacer.
            spacer_thickness: Thickness of the spacer layer.
            substrate_permittivity: Permittivity of the substrate.
            substrate_thickness: Thickness of the susbtrate layer.
            wavelength: Wavelength, with units consistent with thicknesses.
            pitch: Pitch for the pixel quad, with units consistent with thicknesses.
            approximate_num_terms: The approximate number of terms in the expansion.
            truncation: Specifies the truncation of terms.
            fmm_configuration: The configuration for the calculation.
            density_grid_shape: The shape of the density grid defining the patterning
                of the sorter layer.
            field_grid_shape: The shape of the grid for the physical grid computation.
            field_z_offset: The offset into the substrate at which the power into each
                quadrant is computed.
            density_initializer: Callable used to initialize the sorter density.
        """
        self._ambient_permittivity: jnp.ndarray = jnp.asarray(ambient_permittivity)
        self._ambient_thickness: jnp.ndarray = jnp.asarray(ambient_thickness)
        self._cap_permittivity: jnp.ndarray = jnp.asarray(cap_permittivity)
        self._cap_thickness: jnp.ndarray = jnp.asarray(cap_thickness)
        self._sorter_permittivity_solid: jnp.ndarray = jnp.asarray(
            sorter_permittivity_solid
        )
        self._sorter_permittivity_void: jnp.ndarray = jnp.asarray(
            sorter_permittivity_void
        )
        self._sorter_thickness: jnp.ndarray = jnp.asarray(sorter_thickness)
        self._spacer_permittivity: jnp.ndarray = jnp.asarray(spacer_permittivity)
        self._spacer_thickness: jnp.ndarray = jnp.asarray(spacer_thickness)
        self._substrate_permittivity: jnp.ndarray = jnp.asarray(substrate_permittivity)
        self._substrate_thickness: jnp.ndarray = jnp.asarray(substrate_thickness)
        self._wavelength: jnp.ndarray = jnp.asarray(wavelength)
        self._primitive_lattice_vectors = basis.LatticeVectors(
            u=pitch * basis.X,
            v=pitch * basis.Y,
        )
        self._expansion: basis.Expansion = basis.generate_expansion(
            primitive_lattice_vectors=self._primitive_lattice_vectors,
            approximate_num_terms=approximate_num_terms,
            truncation=truncation,
        )
        self._fmm_configuration: fmm.FmmConfiguration = fmm_configuration
        self._density_grid_shape: Tuple[int, int] = density_grid_shape
        self._field_grid_shape: Tuple[int, int] = field_grid_shape
        self._field_z_offset: jnp.ndarray = jnp.asarray(field_z_offset)
        self._density_initializer: Initializer = density_initializer

    def init(self, key: jax.random.PRNGKeyArray) -> Params:
        """Returns initial parameters for the polarization sorter."""
        return {
            "primitive_lattice_vectors": self._primitive_lattice_vectors,
            "layers": {
                "ambient": {"thickness": self._ambient_thickness},
                "cap": {"thickness": self._cap_thickness},
                "sorter": {
                    "thickness": self._sorter_thickness,
                    "density": self._density_initializer(key, self._density_grid_shape),
                },
                "spacer": {"thickness": self._spacer_thickness},
                "substrate": {"thickness": self._substrate_thickness},
            },
        }

    def response(
        self,
        params: Params,
        wavelength: Optional[jnp.ndarray] = None,
        expansion: Optional[basis.Expansion] = None,
        fmm_configuration: Optional[fmm.FmmConfiguration] = None,
        field_grid_shape: Optional[Tuple[int, int]] = None,
        field_z_offset: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, Aux]:
        """Computes the response of the sorter defined by `params`.

        Simulation parameters may optionally be provided, to override the default
        values of the `PolarizationSorterComponent`.

        Args:
            params: Parameters for the sorter, e.g. layer thicknesses.
            wavelength: Wavelength, with units consistent with thicknesses.
            expansion: The field expansion used for the calculation.
            fmm_configuration: The configuration for the calculation.
            field_grid_shape: The shape of the grid for the physical grid computation.
            field_z_offset: The offset into the substrate at which the power into each
                quadrant is computed.

        Returns:
            A `(response, aux)` tuple, where the response is an array giving the
            reflected power and power transmitted into each of the quadrants, for
            each incident linear polarization. `aux` is a dictionary containing the
            grid and fields at the surface of the substrate.
        """

        if wavelength is None:
            wavelength = self._wavelength
        if expansion is None:
            expansion = self._expansion
        if fmm_configuration is None:
            fmm_configuration = self._fmm_configuration
        if field_grid_shape is None:
            field_grid_shape = self._field_grid_shape
        if field_z_offset is None:
            field_z_offset = self._field_z_offset

        return _simulate_polarization_sorter(
            params=params,
            ambient_permittivity=self._ambient_permittivity,
            cap_permittivity=self._cap_permittivity,
            sorter_permittivity_solid=self._sorter_permittivity_solid,
            sorter_permittivity_void=self._sorter_permittivity_void,
            spacer_permittivity=self._spacer_permittivity,
            substrate_permittivity=self._substrate_permittivity,
            wavelength=jnp.asarray(wavelength),
            expansion=expansion,
            fmm_configuration=fmm_configuration,
            field_grid_shape=field_grid_shape,
            field_z_offset=jnp.asarray(field_z_offset),
        )


def _simulate_polarization_sorter(
    params: Params,
    ambient_permittivity: jnp.ndarray,
    cap_permittivity: jnp.ndarray,
    sorter_permittivity_solid: jnp.ndarray,
    sorter_permittivity_void: jnp.ndarray,
    spacer_permittivity: jnp.ndarray,
    substrate_permittivity: jnp.ndarray,
    wavelength: jnp.ndarray,
    expansion: basis.Expansion,
    fmm_configuration: fmm.FmmConfiguration,
    field_grid_shape: Tuple[int, int],
    field_z_offset: jnp.ndarray,
) -> Tuple[jnp.ndarray, Aux]:
    """Simulates the polarization sorter.

    Args:
        params: Parameters for the sorter, e.g. layer thicknesses.
        ambient_permittivity: Permittivity of the ambient.
        cap_permittivity: Permittivity of the cap.
        sorter_permittivity_solid: Permittivity of solid regions in the sorter.
        sorter_permittivity_void: Permittivity of void regions in the sorter.
        spacer_permittivity: Permittivity of the spacer.
        substrate_permittivity: Permittivity of the substrate.
        wavelength: Wavelength, with units consistent with thicknesses.
        expansion: The field expansion used for the calculation.
        fmm_configuration: The configuration for the calculation.
        field_grid_shape: The shape of the grid for the physical grid computation.
        field_z_offset: The offset into the substrate at which the power into each
            quadrant is computed.

    Returns:
        A `(response, aux)` tuple, where the response is an array giving the
        reflected power and power transmitted into each of the quadrants, for
        each incident linear polarization. `aux` is a dictionary containing the
        grid and fields at the surface of the substrate.
    """

    # Assemble the lists of layer permittivities and thicknesses.
    permittivities = [
        ambient_permittivity[jnp.newaxis, jnp.newaxis],
        cap_permittivity[jnp.newaxis, jnp.newaxis],
        utils.interpolate_permittivity(
            permittivity_solid=sorter_permittivity_solid,
            permittivity_void=sorter_permittivity_void,
            density=params["layers"]["sorter"]["density"],
        ),
        spacer_permittivity[jnp.newaxis, jnp.newaxis],
        substrate_permittivity[jnp.newaxis, jnp.newaxis],
    ]

    thicknesses = jnp.asarray(
        [
            params["layers"]["ambient"]["thickness"],
            params["layers"]["cap"]["thickness"],
            params["layers"]["sorter"]["thickness"],
            params["layers"]["spacer"]["thickness"],
            params["layers"]["substrate"]["thickness"],
        ]
    )

    # Perform the eigensolve for each layer in the stack.
    layer_solve_results = [
        layer.eigensolve_isotropic_media(
            wavelength=wavelength,
            in_plane_wavevector=jnp.zeros((2,)),
            primitive_lattice_vectors=params["primitive_lattice_vectors"],
            permittivity=p,
            expansion=expansion,
            fmm_configuration=fmm_configuration,
        )
        for p in permittivities
    ]

    s_matrix = scattering.stack_s_matrix(layer_solve_results, thicknesses)

    # We consider incident plane waves with four different linear polarizations:
    # TE, TM, (TE + TM) / sqrt(2), and (TE - TM) / sqrt(2). These defined by the
    # forward going wave amplitudes in layer `0`, i.e. the ambeint.
    n = expansion.num_terms
    fwd_amplitude_0_start = jnp.zeros((1, 2 * n, 4), dtype=complex)
    fwd_amplitude_0_start = fwd_amplitude_0_start.at[0, 0, 0].set(1)
    fwd_amplitude_0_start = fwd_amplitude_0_start.at[0, 0, 1].set(1 / jnp.sqrt(2))
    fwd_amplitude_0_start = fwd_amplitude_0_start.at[0, n, 1].set(1 / jnp.sqrt(2))
    fwd_amplitude_0_start = fwd_amplitude_0_start.at[0, 0, 2].set(1 / jnp.sqrt(2))
    fwd_amplitude_0_start = fwd_amplitude_0_start.at[0, n, 2].set(-1 / jnp.sqrt(2))
    fwd_amplitude_0_start = fwd_amplitude_0_start.at[0, n, 3].set(1)

    # The backward going wae amplitudes in layer `N`, i.e. the substrate.
    bwd_amplitude_N_end = jnp.zeros_like(fwd_amplitude_0_start)

    # Compute wave amplitudes colocated at the start of the ambient.
    bwd_amplitude_0_end = s_matrix.s21 @ fwd_amplitude_0_start
    fwd_amplitude_0_start, bwd_amplitude_0_start = fields.colocate_amplitudes(
        fwd_amplitude_0_start,
        bwd_amplitude_0_end,
        z_offset=jnp.zeros(()),  # At the start of the ambient.
        layer_solve_result=layer_solve_results[0],
        layer_thickness=params["layers"]["ambient"]["thickness"],
    )
    sz_fwd_0, sz_bwd_0 = fields.amplitude_poynting_flux(
        fwd_amplitude_0_start, bwd_amplitude_0_start, layer_solve_results[0]
    )
    sz_fwd_ambient_sum = jnp.sum(jnp.abs(sz_fwd_0), axis=-2)
    sz_bwd_ambient_sum = jnp.sum(jnp.abs(sz_bwd_0), axis=-2)
    reflection = jnp.abs(sz_bwd_ambient_sum) / jnp.abs(sz_fwd_ambient_sum)

    # Compute the forward-going and backward-going wave amplitudes in the substrate.
    fwd_amplitude_N_start = s_matrix.s11 @ fwd_amplitude_0_start
    fwd_amplitude_N_offset, bwd_amplitude_N_offset = fields.colocate_amplitudes(
        fwd_amplitude_N_start,
        bwd_amplitude_N_end,
        z_offset=field_z_offset,
        layer_solve_result=layer_solve_results[-1],
        layer_thickness=params["layers"]["substrate"]["thickness"],
    )

    # Compute the Fourier field coefficients from the wave amplitudes in the substrate.
    ef, hf = fields.fields_from_wave_amplitudes(
        fwd_amplitude_N_offset,
        bwd_amplitude_N_offset,
        layer_solve_result=layer_solve_results[-1],
    )

    # Compute the physical fields on the real-space grid in the substrate.
    (ex, ey, ez), (hx, hy, hz), (x, y) = fields.fields_on_grid(
        electric_field=ef,
        magnetic_field=hf,
        layer_solve_result=layer_solve_results[-1],
        shape=field_grid_shape,
        num_unit_cells=(1, 1),
    )
    assert ex.shape == wavelength.shape + field_grid_shape + (4,)

    # Compute the Poynting flux on the real-space grid in the substrate.
    sz = _time_average_z_poynting_flux((ex, ey, ez), (hx, hy, hz))

    # Create masks for the four quadrants.
    mask = jnp.zeros(field_grid_shape + (1, 4))
    xdim = field_grid_shape[0] // 2
    ydim = field_grid_shape[1] // 2
    mask = mask.at[:xdim, :ydim, 0, 0].set(1)
    mask = mask.at[:xdim, ydim:, 0, 1].set(1)
    mask = mask.at[xdim:, :ydim, 0, 2].set(1)
    mask = mask.at[xdim:, ydim:, 0, 3].set(1)

    # Use the mask to compute the time average Poynting flux into each quadrant. The axes for
    # `quadrant_sz` are `(wavelength, excitation, quadrant)`. Normalize this using incident power.
    quadrant_sz = jnp.mean(mask * sz[..., jnp.newaxis], axis=(-4, -3))
    quadrant_sz /= sz_fwd_ambient_sum[..., jnp.newaxis]

    # The response consists of the power into each quadrant, and the reflected power. The
    # axes for `response` are `(wavelength, excitation, port)`, where the ports correspond tos
    # reflection or transmission into one of the quadrants.
    response = jnp.concatenate([reflection[..., jnp.newaxis], quadrant_sz], axis=-1)

    aux = {"x": x, "y": y, "ex": ex, "ey": ey, "ez": ez, "hx": hx, "hy": hy, "hz": hz}

    return response, aux


def _time_average_z_poynting_flux(
    electric_fields: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    magnetic_fields: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Computes the time-average z-directed Poynting flux, given the physical fields."""
    ex, ey, _ = electric_fields
    hx, hy, _ = magnetic_fields
    return jnp.real(ex * jnp.conj(hy) - ey * jnp.conj(hx))


def optimize(steps: int = 1000, approximate_num_terms: int = 400) -> List[jnp.ndarray]:
    """Optimizes the density of the polarization sorter component."""

    psc: PolarizationSorterComponent = PolarizationSorterComponent(
        approximate_num_terms=approximate_num_terms
    )

    # Define the target response, which is the response for an ideal polarization
    # sorter. The target response specifies values for reflected power and transmitted
    # power into each of the four subpixels ("quadrants"), for each of four specific
    # linear polarizations.
    target_response: jnp.ndarray = jnp.asarray(
        [
            # R,  Q1,  Q2,   Q3,   Q4
            [0.0, 0.5, 0.25, 0.25, 0.0],  # TE
            [0.0, 0.25, 0.5, 0.0, 0.25],  # (TE + TM) / sqrt(2)
            [0.0, 0.25, 0.0, 0.5, 0.25],  # (TE - TM) / sqrt(2)
            [0.0, 0.0, 0.25, 0.25, 0.5],  # TM
        ]
    )

    params: Params = psc.init(jax.random.PRNGKey(0))
    # TODO(mfschubert): Write some utilities for filtering pytrees and applying
    # leaf-specific optimizers.
    density = params["layers"]["sorter"]["density"]

    opt: optax.GradientTransformation = optax.adam(0.002, b1=0.67, b2=0.9)
    opt_state = opt.init(density)

    def train_step(  # pyre-ignore[3]
        density: jnp.ndarray,
        opt_state: Any,  # pyre-ignore[2]
    ) -> Tuple[jnp.ndarray, Any, jnp.ndarray, jnp.ndarray, Aux]:
        def loss_fn(
            density: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, Aux]]:
            params["layers"]["sorter"]["density"] = density
            response, aux = psc.response(params)
            loss = jnp.sum((response - target_response) ** 2) / response.shape[0]
            return loss, (response, aux)

        (value, (response, aux)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            density
        )
        updates, opt_state = opt.update(grads, opt_state)
        density = jnp.asarray(optax.apply_updates(density, updates))
        density = jnp.clip(density, 0, 1)
        return density, opt_state, value, response, aux

    values = []
    for i in range(steps):
        density, opt_state, value, response, aux = train_step(density, opt_state)
        values.append(value)
        print(i, value)

    # Check if the result is converged.
    params["layers"]["sorter"]["density"] = density
    for factor in (0.5, 1.0, 1.5, 2.0):
        n = int(factor * approximate_num_terms)
        response, _ = psc.response(
            params,
            expansion=basis.generate_expansion(
                primitive_lattice_vectors=params["primitive_lattice_vectors"],
                approximate_num_terms=n,
                truncation=basis.Truncation.CIRCULAR,
            ),
        )
        print(f"Response with approximate_num_terms={n}:")
        print(response)

    return values


if __name__ == "__main__":
    optimize()
