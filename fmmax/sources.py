"""Functions related to sources in the FMM algorithm."""

from typing import Tuple

import jax.numpy as jnp

from lightcycle.fmmax.fmmax import basis, layer, scattering, utils


def dirac_delta_source(
    location: jnp.ndarray,
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: basis.LatticeVectors,
    expansion=basis.Expansion,
) -> jnp.ndarray:
    """Returns the coefficients for a delta source at the specified location.

    This function is appropriate for creating sources to be used with
    `amplitudes_for_source`.

    Args:
        location: The location of the source, with shape `(num_sources, 2)` and
            the trailing axis giving the x and y location. By convention, the
            center of the unit cell is at `(0, 0)`.
        in_plane_wavevector: The in-plane wavevevector for the calculation, which
            gives the offset of the plane wave decomposition. Has shape `(..., 2)`
            with possible batch dimensions.
        primitive_lattice_vectors: The primitive lattice vectors of the unit cell.
        expansion: The Fourier expansion used for the calculation.

    Returns:
        The coefficients, with the shape `(..., expansion.num_terms, num_sources)`.
    """
    if location.ndim != 2 or location.shape[-1] != 2:
        raise ValueError(
            f"`location` must be rank-2 with a trailing axis size of 2, but got "
            f"shape {location.shape}"
        )
    if in_plane_wavevector.shape[-1] != 2:
        raise ValueError(
            f"`in_plane_wavevector` must have a trailing axis size of 2, but got "
            f"shape {in_plane_wavevector.shape}"
        )

    transverse_wavevectors = basis.transverse_wavevectors(
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
    )
    kx = transverse_wavevectors[..., 0][..., jnp.newaxis]
    ky = transverse_wavevectors[..., 1][..., jnp.newaxis]
    x = location[:, 0]
    y = location[:, 1]
    return jnp.exp(-1j * (kx * x + ky * y))


def gaussian_source(
    fwhm: jnp.ndarray,
    location: jnp.ndarray,
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: basis.LatticeVectors,
    expansion=basis.Expansion,
) -> jnp.ndarray:
    """Returns the coefficients for a Gaussian source at the specified location.

    This function is appropriate for creating sources to be used with
    `amplitudes_for_source`.

    Args:
        fwhm: The full-width at half-maximum for the Gaussian source.
        location: The location of the source, with shape `(num_sources, 2)` and
            the trailing axis giving the x and y location. By convention, the
            center of the unit cell is at `(0, 0)`.
        in_plane_wavevector: The in-plane wavevevector for the calculation, which
            gives the offset of the plane wave decomposition. Has shape `(..., 2)`
            with possible batch dimensions.
        primitive_lattice_vectors: The primitive lattice vectors of the unit cell.
        expansion: The Fourier expansion used for the calculation.

    Returns:
        The coefficients, with the shape `(..., expansion.num_terms, num_sources)`.
    """
    if location.ndim != 2 or location.shape[-1] != 2:
        raise ValueError(
            f"`location` must be rank-2 with a trailing axis size of 2, but got "
            f"shape {location.shape}"
        )
    if in_plane_wavevector.shape[-1] != 2:
        raise ValueError(
            f"`in_plane_wavevector` must have a trailing axis size of 2, but got "
            f"shape {in_plane_wavevector.shape}"
        )

    transverse_wavevectors = basis.transverse_wavevectors(
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
    )
    kx = transverse_wavevectors[..., 0][..., jnp.newaxis]
    ky = transverse_wavevectors[..., 1][..., jnp.newaxis]
    x = location[:, 0]
    y = location[:, 1]
    coeffs = jnp.exp(-1j * (kx * x + ky * y))

    sigma = fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))
    return coeffs * jnp.exp(-(sigma**2) * (kx**2 + ky**2) / 2)


# -----------------------------------------------------------------------------
# Functions related to internal sources.
# -----------------------------------------------------------------------------


def amplitudes_for_source(
    jx: jnp.ndarray,
    jy: jnp.ndarray,
    jz: jnp.ndarray,
    s_matrix_before_source: scattering.ScatteringMatrix,
    s_matrix_after_source: scattering.ScatteringMatrix,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Computes wave amplitudes resulting from an internal source.

    The configuration of the calculation is depicted below. A source is located at
    the interface of two layers. The layer stacks before and after the source are
    named as such. The function computes the amplitudes of forward-going and
    backward-going waves at various locations within the stack, also depicted below.

                                source _____
                                            |
             __before_______________________V___after_____________________

            |             |     |           |           |     |           |
            |   layer 0   | ... |  layer i  |   layer   | ... |  layer N  |
            | start layer |     |           |   i + 1   |     | end layer |
            |             |     |           |           |     |           |

                                 -> a_i      -> a_i+1          -> a_N
                    b_0 <-            b_i <-    b_i+1 <-

    Args:
        jx: The x-oriented dipole amplitude; must be at least rank-3 with a
            trailing batch dimension.
        jy: The y-oriented dipole amplitude, with shape matching `jx`.
        jz: The z-oriented dipole amplitude, with shape matching `jx`.
        s_matrix_before_source: The scattering matrix for the layer substack
            before the source, having no overlap with the after-source substack.
            Scattering matrix pairs returned by `scattering.stack_s_matrices_interior`
            may not be directly used.
        s_matrix_after_source: The scattering matrix for the layer substack after
            the source.

    Returns:
        The wave amplitudes:
            backward_amplitude_0_end: The backward-going wave amplitude at the end
                of the first layer.
            forward_amplitude_before_start: The forward-going wave amplitude at the
                start of the layer preceding the source.
            backward_amplitude_before_end: The backward-going wave amplitude at the
                end of the layer preceding the source, i.e. just before the source.
            forward_amplitude_after_start: The forward-going wave amplitude at the
                start of the layer following the source, i.e. just after the source.
            backward_amplitude_after_end: The backward-going wave amplitude at the
                end of the layer following the source.
            forward_amplitude_N_start: The forward-going wave amplitude at the start
                of the final layer.
    """
    pol = polarization_terms(
        jx, jy, jz, layer_solve_result=s_matrix_before_source.end_layer_solve_result
    )
    matrix = emission_matrix(
        s_matrix_before_source=s_matrix_before_source,
        s_matrix_after_source=s_matrix_after_source,
    )

    # The matrix solve directly yields the forward-going wave amplitudes after the
    # source, and the backward-going wave amplitudes before the source.
    ab = jnp.linalg.solve(matrix, pol)
    fwd_amplitude_after_start, bwd_amplitude_before_end = jnp.split(ab, 2, axis=-2)

    # Other wave amplitudes are computed from the scattering matrices.
    bwd_amplitude_0_end = s_matrix_before_source.s22 @ bwd_amplitude_before_end
    fwd_amplitude_before_start = s_matrix_before_source.s12 @ bwd_amplitude_before_end
    bwd_amplitude_after_end = s_matrix_after_source.s21 @ fwd_amplitude_after_start
    fwd_amplitude_N_start = s_matrix_after_source.s11 @ fwd_amplitude_after_start

    return (
        bwd_amplitude_0_end,
        fwd_amplitude_before_start,
        bwd_amplitude_before_end,
        fwd_amplitude_after_start,
        bwd_amplitude_after_end,
        fwd_amplitude_N_start,
    )


def polarization_terms(
    jx: jnp.ndarray,
    jy: jnp.ndarray,
    jz: jnp.ndarray,
    layer_solve_result: layer.LayerSolveResult,
) -> jnp.ndarray:
    """Computes the polarization terms for currents on the real-space grid.

    The polarization terms are discussed in section 7 of [1999 Whittaker].

    Args:
        jx: The Fourier-transformed x-oriented dipole amplitude, with a
            trailing batch dimension.
        jy: The Fourier-transformed y-oriented dipole amplitude.
        jz: The Fourier-transformed z-oriented dipole amplitude.
        layer_solve_result: The results of the layer eigensolve.

    Returns:
        The polarization vector containing parallel and z-oriented terms.
    """
    if not (jx.shape == jy.shape == jz.shape) or jx.ndim < 2:
        raise ValueError(
            f"`jx`, `jy`, and `jz` must have matching shape and be at least "
            f"rank-2, but got shapes {jx.shape, jy.shape, jz.shape}."
        )

    # Add dimensions as needed to ensures shapes are compatible.
    leading_ndim = max(jx.ndim - 3, layer_solve_result.eta_matrix.ndim - 2)
    jx = utils.atleast_nd(jx, n=(leading_ndim + 2))
    jy = utils.atleast_nd(jy, n=(leading_ndim + 2))
    jz = utils.atleast_nd(jz, n=(leading_ndim + 2))
    eta_matrix = utils.atleast_nd(layer_solve_result.eta_matrix, n=(leading_ndim + 2))

    p_parallel = jnp.concatenate([jy, -jx], axis=-2)

    transverse_wavevectors = basis.transverse_wavevectors(
        in_plane_wavevector=layer_solve_result.in_plane_wavevector,
        primitive_lattice_vectors=layer_solve_result.primitive_lattice_vectors,
        expansion=layer_solve_result.expansion,
    )
    kx = transverse_wavevectors[..., 0, jnp.newaxis]
    ky = transverse_wavevectors[..., 1, jnp.newaxis]

    # Use the `eta_matrix`, the Fourier convolution matrix for the inverse of
    # the permittivity rather than inverting the Fourier convolution matrix for
    # permittivity itself. The improved convergence due to this choice is
    # discussed in [1999 Whittaker].
    pzy = ky * eta_matrix @ jz
    pzx = kx * eta_matrix @ jz
    pz = jnp.concatenate([-pzy, pzx], axis=-2)

    p_parallel = jnp.broadcast_to(p_parallel, pz.shape)
    return jnp.concatenate([pz, p_parallel], axis=-2)


def emission_matrix(
    s_matrix_before_source: scattering.ScatteringMatrix,
    s_matrix_after_source: scattering.ScatteringMatrix,
) -> jnp.ndarray:
    """Returns the emission matrix for a source between two layers.

    Args:
        s_matrix_before_source: The scattering matrix for the layer substack
            before the source, having no overlap with the after-source substack.
            Scattering matrix pairs returned by `scattering.stack_s_matrices_interior`
            may not be directly used.
        s_matrix_after_source: The scattering matrix for the layer substack after
            the source.

    Returns:
        The emission matrix.
    """
    q = s_matrix_before_source.end_layer_solve_result.eigenvalues
    phi = s_matrix_before_source.end_layer_solve_result.eigenvectors
    omega_k = s_matrix_before_source.end_layer_solve_result.omega_script_k_matrix
    layer_thickness = s_matrix_before_source.end_layer_thickness

    next_q = s_matrix_after_source.start_layer_solve_result.eigenvalues
    next_phi = s_matrix_after_source.start_layer_solve_result.eigenvectors
    next_omega_k = s_matrix_after_source.start_layer_solve_result.omega_script_k_matrix
    next_layer_thickness = s_matrix_after_source.start_layer_thickness

    # Compute terms used in the matrix and assemble the matrix.
    fd = jnp.exp(1j * q * layer_thickness)
    next_fd = jnp.exp(1j * next_q * next_layer_thickness)

    fd_s12_before = utils.diag(fd) @ s_matrix_before_source.s12
    fd_s21_after = utils.diag(next_fd) @ s_matrix_after_source.s21

    shape = jnp.broadcast_shapes(q.shape, next_q.shape)
    eye = utils.diag(jnp.ones(shape))

    # Equation 7.9 from [1999 Whittaker].
    matrix = jnp.block(
        [
            [
                next_omega_k @ next_phi @ utils.diag(1 / next_q) @ (eye - fd_s21_after),
                omega_k @ phi @ utils.diag(1 / q) @ (eye - fd_s12_before),
            ],
            [
                next_phi @ (eye + fd_s21_after),
                -phi @ (eye + fd_s12_before),
            ],
        ]
    )

    return matrix
