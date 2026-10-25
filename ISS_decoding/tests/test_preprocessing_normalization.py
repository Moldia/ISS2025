import numpy as np
from starfish import ImageStack
from starfish.image import Filter
from starfish.types import Axes

from ISS_decoding import decoding


def test_compatible_histogram_matching_matches_starfish_on_small_stack():
    rng = np.random.default_rng(42)
    image = rng.random((3, 2, 1, 12, 10), dtype=np.float32)
    stack = ImageStack.from_numpy(image)

    expected = Filter.MatchHistograms({Axes.CH, Axes.ROUND}).run(
        stack,
        n_processes=1,
        in_place=False,
    )
    actual = decoding.match_histograms_compatible(stack, n_processes=1)

    np.testing.assert_allclose(
        actual.xarray.values,
        expected.xarray.values,
        rtol=1e-6,
        atol=1e-7,
    )


def test_preprocessing_uses_compatible_histogram_matching_for_mh(monkeypatch):
    class FakeStack:
        def reduce(self, *_args, **_kwargs):
            return self

    stack = FakeStack()

    class FakeTile:
        def get_image(self, _name):
            return stack

    scaled = object()
    calls = []

    def fake_match_histograms(image, **kwargs):
        calls.append((image, kwargs))
        return scaled

    monkeypatch.setattr(
        decoding,
        "match_histograms_compatible",
        fake_match_histograms,
    )
    monkeypatch.setattr(
        decoding.Filter,
        "MatchHistograms",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Starfish's xarray reference calculation must not be used")
        ),
    )

    primary, result = decoding.preprocess_iss_tile(
        FakeTile(),
        register=False,
        filter_images=False,
        channel_normalization="MH",
    )

    assert primary is stack
    assert result is scaled
    assert calls == [
        (
            stack,
            {
                "group_by": {Axes.CH, Axes.ROUND},
                "n_processes": 1,
            },
        )
    ]
