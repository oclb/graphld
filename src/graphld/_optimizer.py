"""Objective-monotone optimization with explicit stationarity auditing.

The score callback can use fixed stochastic probes. A precise callback and
signed objective checks determine stationary status. Average information
supplies proposal scaling and the score criterion.
"""

from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import time


def limit_linear_predictor_step(designs, step, cap):
    """Shorten a linear-predictor step without overflowing its norm calculation."""
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("Step cap must be finite and positive")
    step = np.asarray(step, dtype=float)
    size = float(np.max(np.abs(step)))
    if size == 0 or not np.isfinite(size):
        return step
    unit = step / size
    movement = max(
        float(np.max(np.abs(np.asarray(a, dtype=float) @ unit))) for a in designs
    )
    if movement == 0:
        return step
    fraction = min(1.0, (cap / size) / (movement * (1 + 1e-5)))
    return step * fraction


def finite_difference_block_score(
    parameters, information, block_objective, *, tolerance=1e-3
):
    """Audit a low-dimensional score using deterministic block likelihoods.

    The two step sizes provide a measured numerical-error diagnostic. A
    floating-point resolution guard also catches differences that both round
    to zero in weak coordinates; it is not a rigorous oracle-error bound. The
    information metric is used only to scale perturbations and that diagnostic.
    The oracle is restored to the center even when the audit fails.
    """
    t = np.asarray(parameters, dtype=float)
    steps = np.minimum(0.001, 0.01 / np.sqrt(np.maximum(np.diag(information), 1e-12)))

    def difference(multiplier):
        columns = []
        resolution = []
        for k in range(len(t)):
            plus = t.copy()
            minus = t.copy()
            plus[k] += steps[k] * multiplier
            minus[k] -= steps[k] * multiplier
            fp = np.array(block_objective(plus), copy=True)
            fm = np.array(block_objective(minus), copy=True)
            width = plus[k] - minus[k]
            columns.append((fp - fm) / width)
            resolution.append(
                32.0
                * np.sum(np.spacing(np.abs(fp)) + np.spacing(np.abs(fm)))
                / abs(width)
            )
        return np.column_stack(columns), np.asarray(resolution)

    try:
        coarse, _ = difference(1.0)
        refined, resolution = difference(0.5)
        error = refined.sum(0) - coarse.sum(0)
        error_norm = float(
            np.sqrt(max(error @ np.linalg.solve(information, error), 0.0))
        )
        inverse = np.linalg.solve(information, np.eye(len(t)))
        resolution_norm = float(
            np.sqrt(max(resolution @ np.abs(inverse) @ resolution, 0.0))
        )
        threshold = 0.1 * np.sqrt(2 * tolerance)
        if (
            not np.isfinite(refined).all()
            or not np.isfinite(error_norm)
            or not np.isfinite(resolution_norm)
            or error_norm + resolution_norm > threshold
        ):
            raise RuntimeError("Finite-difference score audit unresolved")
    finally:
        block_objective(t)
    return refined, dict(
        steps=steps.tolist(),
        information_scaled_step_difference=error_norm,
        roundoff_resolution_norm=resolution_norm,
        threshold=threshold,
    )


@dataclass
class OptimizationResult:
    parameters: np.ndarray
    likelihood: float
    status: str
    history: list = field(default_factory=list)
    objective_evaluations: int = 0
    derivative_evaluations: int = 0
    exact_derivative_evaluations: int = 0
    iterations: int = 0
    stationarity: dict = field(default_factory=dict)
    metric_resets: list = field(default_factory=list)
    proposal_selection_seconds: float = 0.0
    initial_parameter_scale: np.ndarray | None = None


def select_capped_proposal(score, information, candidates, shorten, *,
                          objective_resolution=0., minimum_relative_gain=.1):
    """Retain a primary proposal satisfying sufficient predicted progress.

    The primary direction must resolve an objective improvement and attain a
    fixed fraction of the best reference candidate's capped quadratic gain.
    Otherwise use the best productive candidate. The shared line search still
    determines whether the resulting objective improves.
    """
    if not 0 < minimum_relative_gain <= 1:
        raise ValueError('Minimum relative gain must lie in (0, 1]')
    details = {}
    steps = {}
    best_step = np.zeros_like(score)
    best_gain = -np.inf
    selected = None
    for name, raw_step in candidates:
        step = shorten(raw_step)
        steps[name] = step
        raw_size = float(np.max(np.abs(raw_step)))
        valid = np.isfinite(step).all() and np.isfinite(raw_step).all()
        with np.errstate(over='ignore', invalid='ignore'):
            linear = float(score @ step) if valid else -np.inf
            gain = float(linear - .5 * step @ information @ step) if valid else -np.inf
        valid = bool(valid and np.isfinite(linear) and np.isfinite(gain)
                     and linear > 0 and gain > 0)
        details[name] = dict(
            cap_multiplier=float(np.max(np.abs(step))/raw_size) if raw_size > 0 and np.isfinite(raw_size) and np.isfinite(step).all() else None,
            directional_derivative=linear if np.isfinite(linear) else None,
            predicted_gain=gain if np.isfinite(gain) else None,
            productive=valid,
        )
        if valid and gain > best_gain:
            best_step, best_gain, selected = step, gain, name
    primary = candidates[0][0]
    primary_gain = details[primary]['predicted_gain']
    resolvable = (details[primary]['productive'] and primary_gain > objective_resolution)
    sufficient = resolvable and primary_gain >= minimum_relative_gain*best_gain
    reason = ('primary_sufficient_progress' if sufficient else
              'insufficient_relative_predicted_gain' if resolvable else
              'primary_below_objective_resolution')
    if sufficient:
        best_step, selected = steps[primary], primary
    return best_step, dict(direction=selected, candidates=details,
                          criterion='sufficient_capped_quadratic_gain', reason=reason,
                          minimum_relative_gain=minimum_relative_gain,
                          primary_relative_gain=primary_gain/best_gain if primary_gain is not None and best_gain > 0 else None,
                          objective_resolution=objective_resolution)


def maximize(
    initial: np.ndarray,
    objective: Callable,
    derivatives: Callable,
    *,
    exact_derivatives: Callable | None = None,
    limit_step: Callable | None = None,
    strategy: str = "bfgs",
    max_iterations: int = 60,
    objective_tolerance: float = 1e-3,
    initial_damping: float = 0.1,
    audit_correction: bool = False,
    score_oracle_updates_on_audit: bool = False,
    screen_proposal: Callable | None = None,
    trial_observer: Callable | None = None,
):
    """Maximize an objective using AI or BFGS proposals and exact trial values.

    ``derivatives(theta)`` returns likelihood, score, positive information.
    ``limit_step(theta, step)`` can shorten a proposal in model coordinates.
    An exhausted line search triggers an exact-score audit, when supplied.
    Only exact-score stationarity plus signed objective checks can certify the
    result. Without exact derivatives, stagnation is explicitly inconclusive.
    All inference remains the caller's responsibility.

    With ``audit_correction``, a precise audit anchors a control variate for
    subsequent fixed-probe scores. Every stationary classification still
    requires a fresh exact audit at the final parameters.

    If the precise callback updates the approximate score oracle itself, set
    ``score_oracle_updates_on_audit`` with ``audit_correction``. The optimizer
    then relies on that correction and adds no second parameter-score offset.
    """
    if strategy not in {"bfgs", "ai_line", "gradient_line", "trust"}:
        raise ValueError("Unknown optimization strategy")
    if score_oracle_updates_on_audit and not audit_correction:
        raise ValueError("An updating score oracle requires audit_correction")
    x = np.asarray(initial, dtype=float).copy()
    f, g, information = derivatives(x)
    result = OptimizationResult(
        x.copy(), float(f), "max_iterations", derivative_evaluations=1
    )
    scale = np.sqrt(
        np.maximum(np.diag(information), max(np.max(np.diag(information)), 1.0) * 1e-12)
    )
    result.initial_parameter_scale = scale.copy()
    B = information / scale[:, None] / scale[None, :]
    damping = initial_damping
    exact = False
    score_is_exact = False
    correction = np.zeros_like(g)
    raw_point = x.copy()
    raw_score = g.copy()
    small_changes = 0

    def reset_metric(reason):
        nonlocal B
        B = information / scale[:, None] / scale[None, :]
        result.metric_resets.append(dict(iteration=iteration, reason=reason))

    def value(t):
        result.objective_evaluations += 1
        return float(objective(t))

    def refresh(t, force_audit=False):
        nonlocal correction, score_is_exact, raw_point, raw_score
        result.derivative_evaluations += 1
        if exact and (not audit_correction or force_audit):
            if audit_correction and not np.array_equal(t, raw_point):
                result.derivative_evaluations += 1
                _, raw_score, _ = derivatives(t)
                raw_point = t.copy()
            result.exact_derivative_evaluations += 1
            audited = exact_derivatives(t)
            if audit_correction:
                correction = (
                    np.zeros_like(raw_score)
                    if score_oracle_updates_on_audit
                    else audited[1] - raw_score
                )
            score_is_exact = True
            return audited
        f, raw_score, information = derivatives(t)
        raw_point = t.copy()
        score_is_exact = False
        return f, raw_score + correction, information

    def normalized_metric(matrix):
        diagonal = np.diag(matrix)
        d = np.sqrt(np.where(diagonal > 0, diagonal, 1.0))
        return d, (
            matrix / d[:, None] / d[None, :] + matrix.T / d[:, None] / d[None, :]
        ) / 2

    def direction(matrix, score, ridge=0.0, audit=False):
        # A numerical ridge stabilizes proposals only. No inverse-information
        # uncertainty or rank certification is inferred from this solve.
        d, normalized = normalized_metric(matrix)
        e, U = np.linalg.eigh(normalized)
        floor = max(float(e.max()), 1.0) * 1e-10
        denominator = (
            np.where(e > 0, e, floor) if audit else np.maximum(e + ridge, floor)
        )
        return (U @ ((U.T @ (score / d)) / denominator)) / d

    def shorten(step):
        return step if limit_step is None else limit_step(x, step)

    for iteration in range(max_iterations):
        result.iterations = iteration + 1
        if not (
            np.isfinite(f) and np.isfinite(g).all() and np.isfinite(information).all()
        ):
            raise FloatingPointError("Nonfinite accepted objective or derivatives")
        I = information / scale[:, None] / scale[None, :]
        gs = g / scale
        audit_eigenvalues = np.linalg.eigvalsh(normalized_metric(I)[1])
        audit_rank_deficient = bool(
            audit_eigenvalues.min() <= max(float(audit_eigenvalues.max()), 1.0) * 1e-10
        )
        raw_scoring = direction(I, gs, audit=True) / scale
        scoring = shorten(raw_scoring)
        predicted_gain = float(
            g @ raw_scoring - 0.5 * raw_scoring @ information @ raw_scoring
        )
        if (
            not exact
            and exact_derivatives is not None
            and (predicted_gain <= objective_tolerance or small_changes >= 2)
        ):
            exact = True
            f, g, information = refresh(x, force_audit=True)
            reset_metric('precise_refresh')
            small_changes = 1
            continue
        if (
            exact
            and audit_correction
            and not score_is_exact
            and (predicted_gain <= objective_tolerance or small_changes >= 2)
        ):
            f, g, information = refresh(x, force_audit=True)
            reset_metric('precise_refresh')
            small_changes = 0
            continue
        # Audit undamped directions; tiny damped steps are never stationarity.
        if exact and predicted_gain <= objective_tolerance:
            gradient_direction = shorten(gs / scale)
            norm = float(
                np.sqrt(max(gradient_direction @ information @ gradient_direction, 0.0))
            )
            if norm > 0:
                gradient_direction *= min(1.0, np.sqrt(2 * objective_tolerance) / norm)
            gains = []
            candidates = []
            for d in (scoring, gradient_direction):
                for sign in (1.0, -1.0):
                    trial = x + sign * d
                    trial_f = value(trial)
                    gains.append(trial_f - f)
                    candidates.append((trial_f, trial))
            result.stationarity = dict(
                exact_score=True,
                undamped_predicted_gain=predicted_gain,
                signed_objective_gains=gains,
                information_rank_deficient=audit_rank_deficient,
            )
            if max(gains) <= objective_tolerance:
                result.status = (
                    "stationary_singular_information"
                    if audit_rank_deficient
                    else "stationary"
                )
                value(x)  # Restore a stateful oracle to the accepted endpoint.
                break
            best_f, best_x = max(candidates, key=lambda t: t[0])
            x = best_x
            f, g, information = refresh(x)
            reset_metric('signed_check_recovery')
            small_changes = 0
            continue

        metric = B if strategy == "bfgs" else I
        proposal = {}
        if strategy in {'bfgs', 'ai_line'}:
            proposal_started = time.perf_counter()
            candidates = [('bfgs' if strategy == 'bfgs' else 'ai', direction(metric, gs)/scale)]
            if strategy == 'bfgs':
                candidates.append(('ai', direction(I, gs)/scale))
            gradient_step = gs / scale
            curvature = float(gradient_step @ information @ gradient_step)
            if curvature > 0:
                gradient_step *= float(g @ gradient_step) / curvature
            candidates.append(('scaled_gradient', gradient_step))
            selected_step, proposal = select_capped_proposal(
                g, information, candidates, shorten,
                objective_resolution=4*np.spacing(abs(float(f))))
            primary_gain = proposal['candidates'][candidates[0][0]]['predicted_gain']
            proposal['primary_below_objective_resolution'] = (
                primary_gain is None or primary_gain <= 4*np.spacing(abs(float(f))))
            result.proposal_selection_seconds += time.perf_counter() - proposal_started
        accepted = False
        tried = []
        attempt = 0
        screening_enabled = screen_proposal is not None
        screened_attempts = []
        screen_streak = 0
        used_exact_fallback = False
        while attempt < 8 or (screening_enabled and screened_attempts):
            if attempt >= 8:
                screening_enabled = False
                used_exact_fallback = True
                attempt = 0
            if strategy == "trust":
                metric_diag = np.where(np.diag(I) > 0, np.diag(I), 1.0)
                step = shorten(
                    direction(I + damping * np.diag(metric_diag), gs) / scale
                )
            elif strategy == "gradient_line":
                step = gs / scale
                curvature = float(step @ information @ step)
                if curvature > 0:
                    step *= float(g @ step) / curvature
                step = shorten(step) * (0.5**attempt)
            else:
                step = selected_step * (0.5**attempt)
            pred = float(g @ step - 0.5 * step @ information @ step)
            trial = x + step
            valid_proposal = (
                np.isfinite(trial).all() and np.isfinite(pred) and np.isfinite(g @ step)
            )
            if (
                valid_proposal
                and screening_enabled
                and not screen_proposal(x, step, f, g)
            ):
                screened_attempts.append(attempt)
                screen_streak += 1
                if screen_streak >= 2:
                    # Retry the complete original search exactly, including
                    # previously screened proposals, before declaring failure.
                    screening_enabled = False
                    used_exact_fallback = True
                    attempt = 0
                else:
                    attempt += 1
                continue
            screen_streak = 0
            trial_f = value(trial) if valid_proposal else -np.inf
            gain = trial_f - f
            numerical_floor = 4 * np.spacing(abs(float(f)))
            armijo = max(numerical_floor, 1e-4 * float(g @ step))
            rho = gain / pred if pred > 0 else -np.inf
            if trial_observer is not None:
                trial_observer(x, step, f, g, trial_f, attempt in screened_attempts)
            tried.append(
                dict(gain=float(gain), predicted_gain=pred, damping=float(damping),
                     direction=proposal.get('direction', strategy), step_fraction=0.5**attempt,
                     directional_derivative=float(g @ step), armijo_threshold=float(armijo))
            )
            accepted = bool(
                np.isfinite(trial_f)
                and gain > armijo
                and (strategy != "trust" or rho >= 0.1)
            )
            tried[-1]['accepted'] = accepted
            if accepted:
                if strategy == "trust":
                    damping = max(1e-8, damping / (5 if rho > 0.75 else 1))
                break
            if strategy == "trust":
                damping = max(1e-8, damping * 5)
            # Do not spend repeated exact likelihood evaluations on steps too
            # small to resolve numerically; proceed to the derivative audit.
            if np.max(np.abs(step) * scale) < 1e-10:
                attempt = 8
            else:
                attempt += 1

        result.history.append(
            dict(
                iteration=iteration,
                likelihood=float(f),
                exact_score=score_is_exact,
                score_is_exact=score_is_exact,
                predicted_gain=predicted_gain,
                trials=tried,
                accepted=accepted,
                screened_attempts=screened_attempts,
                exact_screen_fallback=used_exact_fallback,
                proposal=proposal,
            )
        )
        if not accepted:
            value(x)
            if not exact and exact_derivatives is not None:
                exact = True
                f, g, information = refresh(x, force_audit=True)
                reset_metric('precise_refresh')
                damping = initial_damping
                small_changes = 1
                continue
            if exact and audit_correction and not score_is_exact:
                f, g, information = refresh(x, force_audit=True)
                reset_metric('precise_refresh')
                damping = initial_damping
                small_changes = 0
                continue
            if not accepted:
                result.status = (
                    "stalled_exact_score" if exact else "stalled_stochastic_score"
                )
                break
        old_x = x.copy()
        old_g = g.copy()
        old_f = f
        x = trial
        f, g, information = refresh(x)
        if abs(float(f) - trial_f) > max(1e-6, 32 * np.spacing(abs(trial_f))):
            raise RuntimeError("Objective changed during derivative refresh")
        small_changes = small_changes + 1 if f - old_f < objective_tolerance else 0
        if strategy == "bfgs":
            s = (x - old_x) * scale
            y = (old_g - g) / scale
            Bs = B @ s
            sy = float(s @ y)
            sBs = float(s @ Bs)
            if (
                sy > 1e-10 * max(np.linalg.norm(s) * np.linalg.norm(y), 1e-30)
                and sBs > 0
            ):
                B = B + np.outer(y, y) / sy - np.outer(Bs, Bs) / sBs
                result.history[-1]['bfgs_update'] = 'accepted'
            else:
                reset_metric('curvature_condition')
                result.history[-1]['bfgs_update'] = 'reset_curvature_condition'
    result.parameters = x.copy()
    result.likelihood = float(f)
    value(x)
    return result
