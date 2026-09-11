"""REML worker callbacks for the shared AI/BFGS optimization framework."""

import time
import numpy as np
from ._optimizer import maximize, limit_linear_predictor_step
from .information_penalty import information_logdet


def optimize_reml(processor, manager, shared, block_data, model, method, flags):
    started = time.perf_counter()
    n = len(model.annotation_columns)
    counts = dict(
        full=0,
        precise=0,
        likelihood_only=0,
        information=0,
        penalty_gradient=0,
        screened_out=0,
        exact_screen_fallbacks=0,
        screen_false_rejections_detected=0,
        counterfactual_linear_screen_rejections=0,
        counterfactual_linear_screen_false_rejections=0,
    )
    seconds = {
        k: 0.0
        for k in [
            "full",
            "precise",
            "likelihood_only",
            "information",
            "penalty_gradient",
        ]
    }
    initialized = False
    weight = method.information_penalty
    accepted = []
    base = {}
    last_likelihood = np.nan
    designs = [
        np.asarray(
            b["sumstats"].select(model.annotation_columns).to_numpy(), dtype=float
        )
        for b in block_data
        if b["sumstats"] is not None and len(b["sumstats"])
    ]

    def evaluate(flag):
        precise = flag == flags["COMPUTE_PRECISE"]
        key = (
            "full"
            if flag
            in [flags["INITIALIZE"], flags["COMPUTE_ALL"], flags["COMPUTE_PRECISE"]]
            else "information"
            if flag == flags["COMPUTE_INFORMATION"]
            else "penalty_gradient"
            if flag == flags["COMPUTE_PENALTY_GRADIENT"]
            else "likelihood_only"
        )
        begin = time.perf_counter()
        manager.start_workers(flag)
        manager.await_workers()
        duration = time.perf_counter() - begin
        counts[key] += 1
        seconds[key] += duration
        if precise:
            counts["precise"] += 1
            seconds["precise"] += duration

    def information():
        return -processor._sum_blocks(shared["hessian"], (n, n))

    def penalty(with_gradient=False):
        if not weight:
            return 0.0, np.zeros(n)
        value, inverse = information_logdet(information())
        if not with_gradient:
            return weight * value, None
        shared["penalty_inverse"] = inverse.ravel()
        evaluate(flags["COMPUTE_PENALTY_GRADIENT"])
        gradient = weight * processor._sum_blocks(shared["penalty_gradient"], (n,))
        if not np.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite information-penalty gradient")
        return weight * value, gradient

    def derivatives(theta, precise=False):
        nonlocal initialized, last_likelihood
        shared["params"] = theta.copy()
        evaluate(
            flags["INITIALIZE"]
            if not initialized
            else flags["COMPUTE_PRECISE"]
            if precise
            else flags["COMPUTE_ALL"]
        )
        initialized = True
        last_likelihood = float(shared["likelihood"].sum())
        g = processor._sum_blocks(shared["gradient"], (n,)).copy()
        I = information().copy()
        try:
            value, gp = penalty(with_gradient=True)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "Information penalty requires positive-definite global information at an accepted point; use finite nonsaturated starting coefficients"
            ) from error
        total = last_likelihood + value
        if (
            not np.isfinite(total)
            or not np.isfinite(g).all()
            or not np.isfinite(I).all()
            or not np.isfinite(shared["variant_data"]).all()
        ):
            raise FloatingPointError(
                "Nonfinite accepted likelihood, score or information"
            )
        base.update(
            parameters=theta.copy(),
            likelihood=last_likelihood,
            penalty=value,
            penalty_gradient=gp.copy(),
        )
        row = dict(
            parameters=theta.copy(),
            likelihood=last_likelihood,
            penalty=value,
            objective=total,
        )
        if accepted and np.array_equal(accepted[-1]["parameters"], theta):
            accepted[-1] = row
        else:
            accepted.append(row)
        if method.verbose:
            print(
                f"REML {method.optimizer}: objective {total:.9g}; precise score {precise}",
                flush=True,
            )
        return total, g + gp, I

    def objective(theta):
        nonlocal last_likelihood
        shared["params"] = theta.copy()
        evaluate(
            flags["COMPUTE_INFORMATION"] if weight else flags["COMPUTE_LIKELIHOOD_ONLY"]
        )
        last_likelihood = float(shared["likelihood"].sum())
        if not np.isfinite(shared["variant_data"]).all():
            raise FloatingPointError(
                "Nonfinite trial heritability cannot be used for optimization"
            )
        if not np.isfinite(last_likelihood):
            return -np.inf
        try:
            value, _ = penalty()
        except (np.linalg.LinAlgError, ValueError):
            return -np.inf
        total = last_likelihood + value
        return float(total) if np.isfinite(total) else -np.inf

    def threshold(f, g, step):
        return max(4 * np.spacing(abs(float(f))), 1e-4 * float(g @ step))

    def screen(theta, step, f, g):
        nonlocal last_likelihood
        assert np.array_equal(theta, base["parameters"])
        shared["params"] = theta + step
        evaluate(flags["COMPUTE_LIKELIHOOD_ONLY"])
        last_likelihood = float(shared["likelihood"].sum())
        estimate = last_likelihood - base["likelihood"]
        if method.penalty_trial_strategy == "linear_screen":
            estimate += float(base["penalty_gradient"] @ step)
        passed = bool(np.isfinite(estimate) and estimate > threshold(f, g, step))
        if not passed:
            counts["screened_out"] += 1
        return passed

    def observe_trial(theta, step, f, g, actual, previously_screened):
        if not weight:
            return
        assert np.array_equal(theta, base["parameters"])
        cutoff = threshold(f, g, step)
        actual_pass = bool(np.isfinite(actual) and actual - f > cutoff)
        estimate = (
            last_likelihood
            - base["likelihood"]
            + float(base["penalty_gradient"] @ step)
        )
        rejected = not (np.isfinite(estimate) and estimate > cutoff)
        counts["counterfactual_linear_screen_rejections"] += int(rejected)
        counts["counterfactual_linear_screen_false_rejections"] += int(
            rejected and actual_pass
        )
        counts["screen_false_rejections_detected"] += int(
            previously_screened and actual_pass
        )

    result = maximize(
        np.asarray(model.params).ravel(),
        objective,
        derivatives,
        exact_derivatives=lambda theta: derivatives(theta, precise=True),
        limit_step=lambda theta, step: limit_linear_predictor_step(designs, step, 2.0),
        strategy="ai_line" if method.optimizer == "ai" else "bfgs",
        max_iterations=method.num_iterations,
        objective_tolerance=method.convergence_tol,
        audit_correction=True,
        score_oracle_updates_on_audit=True,
        screen_proposal=screen
        if weight and method.penalty_trial_strategy != "exact"
        else None,
        trial_observer=observe_trial,
    )
    # Mandatory final precise grouped likelihood derivatives supply the existing
    # pseudo-jackknife; the penalty Hessian is deliberately not substituted.
    final_objective, final_score, final_information = derivatives(
        result.parameters, precise=True
    )
    counts["exact_screen_fallbacks"] = sum(
        h["exact_screen_fallback"] for h in result.history
    )
    return dict(
        result=result,
        accepted=accepted,
        final_likelihood=base["likelihood"],
        final_penalty=base["penalty"],
        final_objective=final_objective,
        final_score=final_score,
        final_penalty_gradient=base["penalty_gradient"].copy(),
        final_information=final_information,
        evaluation_counts=counts,
        evaluation_seconds=seconds,
        optimization_seconds=time.perf_counter() - started,
    )
