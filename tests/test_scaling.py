import pytest

torch = pytest.importorskip("torch")

from shape_flow.scaling import Standardizer


def test_standardizer_round_trip_and_log_det():
    values = torch.tensor(
        [[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]],
        dtype=torch.float32,
    )
    scaler = Standardizer().fit(values)

    transformed = scaler.transform(values)
    recovered = scaler.inverse_transform(transformed)

    assert torch.allclose(recovered, values, atol=1.0e-6)
    assert torch.allclose(transformed.mean(dim=0), torch.zeros(2), atol=1.0e-6)
    assert torch.allclose(
        scaler.log_abs_det_jacobian(),
        torch.log(scaler.std).sum(),
    )


def test_constant_feature_is_finite():
    values = torch.ones(4, 2)
    scaler = Standardizer().fit(values)
    transformed = scaler.transform(values)

    assert torch.isfinite(transformed).all()
    assert torch.isfinite(scaler.log_abs_det_jacobian())
