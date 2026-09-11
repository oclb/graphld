"""Penalty algebra and real worker/supervisor integration on dense small blocks."""
from types import SimpleNamespace
import numpy as np
import polars as pl
import pytest

from graphld.heritability import FLAGS, GraphREML, ModelOptions, MethodOptions, _get_link_functions
from graphld.information_penalty import information_terms, information_logdet, information_penalty_gradient


class DenseOperator:
    def __init__(self, matrix, annotations, nodes):
        self.matrix = matrix.copy()
        self.variant_indices = nodes
        self.shape = matrix.shape
        self.variant_info = pl.DataFrame(dict(annot_indices=np.arange(len(nodes)),
                                             **{f'a{k}': annotations[:, k] for k in range(annotations.shape[1])}))

    def solve(self, x):
        return np.linalg.solve(self.matrix, x)

    def logdet(self):
        return np.linalg.slogdet(self.matrix)[1]

    def inverse_diagonal(self, **kwargs):
        return np.diag(np.linalg.inv(self.matrix))

    def update_matrix(self, delta):
        self.matrix += np.diag(delta)

    def get_diagonal(self):
        return np.diag(self.matrix).copy()

    def set_diagonal(self, values):
        np.fill_diagonal(self.matrix, values)

    def del_factor(self):
        pass


def blocks(seed=0):
    rng = np.random.default_rng(seed)
    result = []
    for n in [8, 11]:
        a = np.column_stack([np.ones(n + 3), rng.normal(size=n + 3)])
        nodes = np.r_[np.arange(n), [0, 2, 4]]
        l = rng.normal(size=(n, n))
        matrix = l @ l.T / n + np.eye(n)
        result.append((a, nodes, matrix, rng.normal(size=n)))
    return result


@pytest.mark.parametrize('link', ['softplus', 'exponential'])
def test_global_penalty_gradient_with_aggregated_variants(link):
    theta = np.array([[-.6], [.3]])
    fn, grad, second = _get_link_functions(link, 2.3)

    def compute(t, with_gradient=False):
        information = np.zeros((2, 2))
        records = []
        for a, nodes, matrix, y in blocks():
            d = np.zeros(len(y)); np.add.at(d, nodes, fn(a, t).ravel())
            op = DenseOperator(matrix + np.diag(d), a, nodes)
            j = np.zeros((len(y), 2)); np.add.at(j, nodes, grad(a, t))
            info, b, v = information_terms(op, y, j)
            information += info
            records.append((op, a, nodes, second(np.array(1), a @ t), j, b, v))
        value, inverse = information_logdet(information)
        if not with_gradient:
            return value
        return sum(information_penalty_gradient(*r, inverse) for r in records)

    eps = 1e-5
    numeric = np.array([(compute(theta + eps*e[:, None]) - compute(theta - eps*e[:, None]))/(2*eps)
                        for e in np.eye(2)])
    np.testing.assert_allclose(compute(theta, True), numeric, atol=1e-8)


class Manager:
    def __init__(self, shared, data, operators, model, method):
        self.shared, self.data, self.operators = shared, data, operators
        self.model, self.method = model, method
        self.calls = []

    def start_workers(self, flag):
        self.calls.append(flag)
        # These blocks are already initialized in effect-size coordinates.
        if flag == FLAGS['INITIALIZE']:
            flag = FLAGS['COMPUTE_ALL']
        for op, data in zip(self.operators, self.data):
            GraphREML.process_block(op, SimpleNamespace(value=flag), self.shared, 0, data,
                                    (self.model, self.method))

    def await_workers(self):
        pass


def fit(strategy, weight=1., optimizer='ai'):
    model = ModelOptions(annotation_columns=['a0', 'a1'], params=np.array([[-.6], [.3]]),
                         sample_size=1, link_fn_denominator=2.3)
    penalty_options = {} if weight is None else dict(information_penalty=weight)
    method = MethodOptions(**penalty_options, optimizer=optimizer, penalty_trial_strategy=strategy,
                           num_iterations=12, num_jackknife_blocks=2, reset_trust_region=True)
    data, operators = [], []
    offset = 0
    for idx, (a, nodes, matrix, y) in enumerate(blocks()):
        operators.append(DenseOperator(matrix, a, nodes))
        df = pl.DataFrame(dict(SNP=[f'{idx}:{i}' for i in range(len(nodes))],
                               CHR=[1]*len(nodes), POS=list(range(len(nodes))), a0=a[:,0], a1=a[:,1]))
        data.append(dict(sumstats=df, Pz=y, block_index=idx, variant_offset=offset))
        offset += len(nodes)
    shared = GraphREML.create_shared_memory(pl.DataFrame({'x':[0,1]}), data, num_params=2, method=method)
    manager = Manager(shared, data, operators, model, method)
    result = GraphREML.supervise(manager, shared, data, num_iterations=12, num_params=2,
                                 verbose=False, method=method, model=model)
    return result, manager


@pytest.mark.parametrize('optimizer', ['ai'])
@pytest.mark.parametrize('strategy', ['exact', 'linear_screen', 'likelihood_screen'])
def test_worker_and_supervisor_penalty_objective(strategy, optimizer):
    result, manager = fit(strategy, optimizer=optimizer)
    assert np.isfinite(result['parameters']).all()
    assert np.min(np.diff(result['objective_history'])) >= -1e-10
    assert result['log']['final_objective'] == pytest.approx(
        result['log']['final_likelihood'] + result['log']['final_penalty'])
    assert result['log']['evaluation_counts']['penalty_gradient'] > 0
    assert FLAGS['COMPUTE_PENALTY_GRADIENT'] in manager.calls
    assert all('penalty_cache' not in d for d in manager.data)


def test_disabled_penalty_never_requests_extra_passes():
    result, manager = fit('exact', weight=0.)
    assert result['penalty_history'] == [0.] * len(result['likelihood_history'])
    assert FLAGS['COMPUTE_INFORMATION'] not in manager.calls
    assert FLAGS['COMPUTE_PENALTY_GRADIENT'] not in manager.calls


def test_singular_information_is_not_regularized_silently():
    with pytest.raises(np.linalg.LinAlgError):
        information_logdet(np.diag([1., 0.]))


def test_options_and_default_link_derivatives():
    assert MethodOptions().information_penalty == 0
    for value in [-1., np.nan, np.inf]:
        with pytest.raises(ValueError):
            MethodOptions(information_penalty=value)
    with pytest.raises(ValueError):
        MethodOptions(penalty_trial_strategy='unknown')
    with pytest.raises(ValueError):
        ModelOptions(link_function=lambda d: (lambda a,t: a, lambda a,t: a))
    for link in ['softplus', 'exponential']:
        fn, grad, second = _get_link_functions(link, 3.)
        x = np.array([[-10.], [0.], [2.]])
        eps = 1e-5
        np.testing.assert_allclose(second(np.array(1), x),
                                   (grad(np.array(1), x+eps)-grad(np.array(1), x-eps))/(2*eps), rtol=1e-6)


def test_cli_penalty_options_and_diagnostics(tmp_path):
    import csv
    from graphld._cli_parser import build_parser
    from graphld.cli import write_convergence_results
    base = ['reml', 'trait.sumstats', 'out', '--annot-dir', 'annotations']
    parser = build_parser()
    assert parser.parse_args(base).information_penalty == 0.
    args = parser.parse_args(base + ['--information-penalty', '--link-function', 'exponential',
                                     '--penalty-trial-strategy', 'linear_screen'])
    assert args.information_penalty == 1.
    assert args.link_function == 'exponential'
    assert args.penalty_trial_strategy == 'linear_screen'
    assert parser.parse_args(base + ['--information-penalty', '0.5']).information_penalty == .5
    result, _ = fit('exact')
    output = tmp_path / 'convergence.csv'
    write_convergence_results(str(output), result)
    with output.open() as f:
        rows = list(csv.reader(f))
    summary = dict(zip(rows[0], rows[1]))
    assert float(summary['final_objective']) == pytest.approx(result['log']['final_objective'])
    assert summary['uncertainty_method'] == 'unpenalized_information_pseudojackknife'
    assert ['iteration', 'likelihood', 'penalty', 'objective', 'trust_region_lambda'] in rows


def test_extreme_penalty_trial_restores_identical_covariance_and_objective():
    a, nodes, matrix, y = blocks()[0]
    op = DenseOperator(matrix, a, nodes)
    theta = np.array([[-.6], [.3]])
    baseline = op.get_diagonal()
    old_h2 = np.zeros(len(nodes))
    def evaluate(t):
        nonlocal old_h2
        likelihood, _, hessian, old_h2 = GraphREML._compute_block_likelihood(
            op, y, a, t, 2.3, old_h2, 10, False, information_only=True,
            residual_diagonal=baseline)
        penalty, _ = information_logdet(-hessian)
        return likelihood, penalty
    initial = evaluate(theta)
    initial_matrix = op.matrix.copy()
    # This trial loses the small residual diagonal in floating point arithmetic.
    evaluate(np.array([[1e22], [0.]]))
    restored = evaluate(theta)
    np.testing.assert_array_equal(op.matrix, initial_matrix)
    np.testing.assert_array_equal(restored, initial)


def test_default_off_matches_explicit_zero_without_penalty_work():
    implicit, manager1 = fit('exact', weight=None)
    explicit, manager2 = fit('exact', weight=0.)
    for key in ['parameters','heritability','enrichment','variant_h2','final_gradient_groups','final_hessian_groups']:
        np.testing.assert_array_equal(implicit[key],explicit[key])
    assert manager1.calls == manager2.calls
    assert FLAGS['COMPUTE_INFORMATION'] not in manager1.calls
    assert FLAGS['COMPUTE_PENALTY_GRADIENT'] not in manager1.calls
    assert FLAGS['COMPUTE_PRECISE'] in manager1.calls
    assert 'penalty_inverse' not in manager1.shared._data_dict
    assert 'penalty_gradient' not in manager1.shared._data_dict


def test_penalized_uncertainty_retains_explicit_approximation_metadata():
    result, _ = fit('exact')
    assert result['log']['uncertainty_method']=='unpenalized_information_pseudojackknife'
    assert result['log']['uncertainty_penalty_curvature_included'] is False
    assert result['log']['penalized_uncertainty_calibration']=='not_evaluated'
    assert result['log']['uncertainty_status'] in ['available','provisional_unresolved_endpoint']
    assert np.isfinite(result['parameters_se']).all()
    np.testing.assert_allclose(result['final_gradient_groups'].sum(0)+result['final_penalty_gradient'],result['final_objective_gradient'])
    np.testing.assert_allclose(result['annotation_fractions'],result['annotation_counts']/len(result['variant_h2']))
