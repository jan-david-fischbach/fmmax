"""Tests for the shared vector field source and the batched multi-layer eigensolve."""

import unittest

import jax
import jax.numpy as jnp
import numpy as onp
from parameterized import parameterized

from fmmax import basis, fmm

jax.config.update("jax_enable_x64", True)

LATTICE = basis.LatticeVectors(u=basis.X, v=basis.Y)
EXPANSION = basis.generate_expansion(
    primitive_lattice_vectors=LATTICE,
    approximate_num_terms=40,
    truncation=basis.Truncation.CIRCULAR,
)
VECTOR_FORMULATIONS = [
    (fmm.Formulation.JONES_DIRECT,),
    (fmm.Formulation.JONES_DIRECT_FOURIER,),
    (fmm.Formulation.POL,),
    (fmm.Formulation.NORMAL,),
]


def disc(radius, n=48):
    x = (jnp.arange(n) + 0.5) / n - 0.5
    return (x[:, None] ** 2 + x[None, :] ** 2 <= radius**2).astype(float)


def assert_results_close(test, a, b, rtol=1e-10, atol=1e-10):
    for name in (
        "eigenvalues",
        "z_permittivity_matrix",
        "inverse_z_permittivity_matrix",
        "transverse_permittivity_matrix",
    ):
        onp.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=rtol, atol=atol, err_msg=name)
    # Eigenvectors are unique up to a phase per mode: compare via the solved matrix.
    onp.testing.assert_allclose(
        a.eigenvectors @ (a.eigenvalues[..., None] * jnp.swapaxes(jnp.linalg.inv(a.eigenvectors), -1, -2)).swapaxes(-1, -2),
        b.eigenvectors @ (b.eigenvalues[..., None] * jnp.swapaxes(jnp.linalg.inv(b.eigenvectors), -1, -2)).swapaxes(-1, -2),
        rtol=1e-8,
        atol=1e-8,
    )
    test.assertEqual(a.batch_shape, b.batch_shape)


class VectorFieldSourceTest(unittest.TestCase):
    @parameterized.expand(VECTOR_FORMULATIONS)
    def test_source_equal_to_permittivity_matches_default(self, formulation):
        permittivity = 1 + 10 * disc(0.3)
        kwargs = dict(
            wavelength=jnp.asarray(0.6),
            in_plane_wavevector=jnp.asarray([0.3, 0.1]),
            primitive_lattice_vectors=LATTICE,
            permittivity=permittivity,
            expansion=EXPANSION,
            formulation=formulation,
        )
        expected = fmm.eigensolve_isotropic_media(**kwargs)
        result = fmm.eigensolve_isotropic_media(vector_field_source=permittivity, **kwargs)
        assert_results_close(self, result, expected)

    @parameterized.expand(VECTOR_FORMULATIONS)
    def test_unbatched_source_matches_batched_dispersive_permittivity(self, formulation):
        # A dispersive material on a fixed geometry: the permittivity varies with the
        # batch (wavelength), the geometry, and therefore the vector field, does not.
        wavelength = jnp.linspace(0.5, 0.7, 5)
        eps = 11 + 0.5j + 0.3 * jnp.arange(5)
        pattern = disc(0.3)
        permittivity = 1 + pattern * (eps[:, None, None] - 1)
        kwargs = dict(
            wavelength=wavelength,
            in_plane_wavevector=jnp.asarray([0.3, 0.1]),
            primitive_lattice_vectors=LATTICE,
            permittivity=permittivity,
            expansion=EXPANSION,
            formulation=formulation,
        )
        expected = fmm.eigensolve_isotropic_media(**kwargs)
        result = fmm.eigensolve_isotropic_media(vector_field_source=permittivity[0], **kwargs)
        assert_results_close(self, result, expected, rtol=1e-8, atol=1e-8)
        self.assertEqual(result.tangent_vector_field[0].shape, permittivity.shape)

    def test_source_ignored_for_fft_formulation(self):
        permittivity = 1 + 10 * disc(0.3)
        kwargs = dict(
            wavelength=jnp.asarray(0.6),
            in_plane_wavevector=jnp.asarray([0.3, 0.1]),
            primitive_lattice_vectors=LATTICE,
            permittivity=permittivity,
            expansion=EXPANSION,
            formulation=fmm.Formulation.FFT,
        )
        expected = fmm.eigensolve_isotropic_media(**kwargs)
        result = fmm.eigensolve_isotropic_media(vector_field_source=disc(0.1), **kwargs)
        assert_results_close(self, result, expected)

    def test_source_with_wrong_shape_raises(self):
        with self.assertRaisesRegex(ValueError, "vector_field_source"):
            fmm.eigensolve_isotropic_media(
                wavelength=jnp.asarray(0.6),
                in_plane_wavevector=jnp.asarray([0.0, 0.0]),
                primitive_lattice_vectors=LATTICE,
                permittivity=1 + 10 * disc(0.3),
                expansion=EXPANSION,
                formulation=fmm.Formulation.JONES_DIRECT,
                vector_field_source=disc(0.3, n=32),
            )


class LayersEigensolveTest(unittest.TestCase):
    @parameterized.expand([(fmm.Formulation.FFT,), (fmm.Formulation.JONES_DIRECT,), (fmm.Formulation.POL,)])
    def test_matches_individual_solves(self, formulation):
        wavelength = jnp.linspace(0.5, 0.7, 4)
        eps = 11 + 0.5j + 0.3 * jnp.arange(4)
        permittivities = [1 + disc(r)[None] * (eps[:, None, None] - 1) for r in (0.2, 0.25, 0.3)]
        kwargs = dict(
            wavelength=wavelength,
            in_plane_wavevector=jnp.asarray([0.3, 0.1]),
            primitive_lattice_vectors=LATTICE,
            expansion=EXPANSION,
            formulation=formulation,
        )
        results = fmm.eigensolve_isotropic_media_layers(permittivities=permittivities, **kwargs)
        self.assertEqual(len(results), 3)
        for result, permittivity in zip(results, permittivities):
            assert_results_close(self, result, fmm.eigensolve_isotropic_media(permittivity=permittivity, **kwargs))
            self.assertEqual(result.expansion, EXPANSION)

    def test_matches_individual_solves_with_sources(self):
        wavelength = jnp.linspace(0.5, 0.7, 4)
        eps = 11 + 0.5j + 0.3 * jnp.arange(4)
        radii = (0.2, 0.25, 0.3)
        permittivities = [1 + disc(r)[None] * (eps[:, None, None] - 1) for r in radii]
        sources = [disc(r) for r in radii]
        kwargs = dict(
            wavelength=wavelength,
            in_plane_wavevector=jnp.asarray([0.3, 0.1]),
            primitive_lattice_vectors=LATTICE,
            expansion=EXPANSION,
            formulation=fmm.Formulation.JONES_DIRECT,
        )
        results = fmm.eigensolve_isotropic_media_layers(
            permittivities=permittivities, vector_field_sources=sources, **kwargs
        )
        for result, permittivity, source in zip(results, permittivities, sources):
            expected = fmm.eigensolve_isotropic_media(permittivity=permittivity, vector_field_source=source, **kwargs)
            assert_results_close(self, result, expected)

    def test_results_usable_in_s_matrix(self):
        from fmmax import scattering

        wavelength = jnp.asarray([0.55, 0.6])
        kwargs = dict(
            wavelength=wavelength,
            in_plane_wavevector=jnp.asarray([0.2, 0.0]),
            primitive_lattice_vectors=LATTICE,
            expansion=EXPANSION,
            formulation=fmm.Formulation.JONES_DIRECT,
        )
        permittivities = [1 + 10 * disc(r)[None] * jnp.ones((2, 1, 1)) for r in (0.3, 0.2)]
        layers = fmm.eigensolve_isotropic_media_layers(permittivities=permittivities, **kwargs)
        single = [fmm.eigensolve_isotropic_media(permittivity=p, **kwargs) for p in permittivities]
        ambient = fmm.eigensolve_isotropic_media(permittivity=jnp.ones((1, 1)), **kwargs)
        thicknesses = [jnp.asarray(0.0), jnp.asarray(0.1), jnp.asarray(0.1), jnp.asarray(0.0)]
        s_a = scattering.stack_s_matrix([ambient, *layers, ambient], thicknesses)
        s_b = scattering.stack_s_matrix([ambient, *single, ambient], thicknesses)
        onp.testing.assert_allclose(jnp.abs(s_a.s11), jnp.abs(s_b.s11), atol=1e-9)
        onp.testing.assert_allclose(jnp.abs(s_a.s21), jnp.abs(s_b.s21), atol=1e-9)

    def test_empty_and_mismatched_inputs(self):
        kwargs = dict(
            wavelength=jnp.asarray(0.6),
            in_plane_wavevector=jnp.asarray([0.0, 0.0]),
            primitive_lattice_vectors=LATTICE,
            expansion=EXPANSION,
        )
        self.assertEqual(fmm.eigensolve_isotropic_media_layers(permittivities=[], **kwargs), [])
        with self.assertRaisesRegex(ValueError, "same shape"):
            fmm.eigensolve_isotropic_media_layers(permittivities=[disc(0.2), disc(0.2, n=32)], **kwargs)
        with self.assertRaisesRegex(ValueError, "one vector field source per layer"):
            fmm.eigensolve_isotropic_media_layers(
                permittivities=[disc(0.2), disc(0.3)], vector_field_sources=[disc(0.2)], **kwargs
            )
