"""Stationarity and availability of derived uncertainty are separate contracts."""

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from scipy.sparse import csc_matrix

from graphld.heritability import FLAGS, GraphREML, MethodOptions, ModelOptions
from graphld.precision import PrecisionOperator


@pytest.mark.parametrize("squared_observations,available", [
    ((1., 1.002), True),
    ((.25, 1.752), False),
])
def test_stationary_fit_checks_derived_delete_uncertainty(
    monkeypatch, squared_observations, available,
):
    # In both cases the Gaussian full optimum is h2=mean(y**2)-1=.001.
    # Heterogeneous blocks can still give extreme one-step delete predictions.
    model = ModelOptions(
        annotation_columns=["base"], params=np.array([[np.log(.001)]]),
        sample_size=1, link_fn_denominator=1., link_function="exponential",
    )
    method = MethodOptions(num_iterations=10, num_jackknife_blocks=2)
    blocks, operators = [], []
    for idx, squared_y in enumerate(squared_observations):
        nodes = np.arange(3)
        operators.append(PrecisionOperator(
            csc_matrix(np.eye(3)),
            pl.DataFrame(dict(index=nodes, annot_indices=nodes, base=np.ones(3))),
        ))
        blocks.append(dict(
            sumstats=pl.DataFrame(dict(
                SNP=[f"{idx}:{i}" for i in nodes], CHR=[1]*3, POS=nodes,
                base=np.ones(3),
            )),
            Pz=np.full(3, np.sqrt(squared_y)), block_index=idx,
            variant_offset=idx*3,
        ))
    shared = GraphREML.create_shared_memory(
        pl.DataFrame({"block": [0, 1]}), blocks, num_params=1, method=method,
    )

    class Manager:
        def start_workers(self, flag):
            # Blocks are already loaded in effect-size coordinates.
            if flag == FLAGS["INITIALIZE"]:
                flag = FLAGS["COMPUTE_ALL"]
            for operator, block in zip(operators, blocks):
                GraphREML.process_block(
                    operator, SimpleNamespace(value=flag), shared, 0, block,
                    (model, method),
                )

        def await_workers(self):
            pass

    raw_deletes = {}
    predict = GraphREML._compute_jackknife_heritability

    def capture_deletes(*args):
        h2, sums = predict(*args)
        raw_deletes["h2"] = h2.copy()
        return h2, sums

    monkeypatch.setattr(GraphREML, "_compute_jackknife_heritability", capture_deletes)
    result = GraphREML.supervise(
        Manager(), shared, blocks, num_iterations=10, num_params=1,
        verbose=False, method=method, model=model,
    )
    assert result["log"]["converged"]
    assert result["log"]["termination_reason"] == "stationary"
    assert np.isfinite(raw_deletes["h2"]).all()
    np.testing.assert_allclose(result["variant_h2"], .001, rtol=1e-10)
    if available:
        assert result["log"]["uncertainty_status"] == "available"
        assert np.isfinite(result["heritability_se"]).all()
        assert np.isfinite(result["enrichment_se"]).all()
    else:
        assert raw_deletes["h2"].max() > 1e180
        assert raw_deletes["h2"].min() == 0
        assert result["log"]["uncertainty_status"] == "nonfinite_delete_uncertainty"
        for key in ["jackknife_params", "jackknife_h2", "jackknife_enrichment",
                    "parameters_se", "heritability_se", "enrichment_se",
                    "parameters_log10pval", "heritability_log10pval",
                    "enrichment_log10pval"]:
            assert np.isnan(result[key]).all(), key
