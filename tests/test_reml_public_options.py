"""Public optimizer/Firth choices and migration defaults."""
import numpy as np
import pytest
from graphld.heritability import MethodOptions
from graphld._cli_parser import build_parser


def test_new_defaults_and_explicit_options():
    method=MethodOptions()
    assert method.optimizer=='ai' and method.information_penalty==0
    assert method.num_iterations==100 and method.convergence_tol==.001
    explicit=MethodOptions(optimizer='ai',num_iterations=7,convergence_tol=.02,information_penalty=.5)
    assert (explicit.optimizer,explicit.num_iterations,explicit.convergence_tol,explicit.information_penalty)==('ai',7,.02,.5)
    random=MethodOptions(gradient_seed=None)
    assert isinstance(random.gradient_seed,int)


@pytest.mark.parametrize('iterations',[0,-1,np.nan,np.inf,-np.inf,2.5,True])
def test_invalid_iteration_budget(iterations):
    with pytest.raises(ValueError,match='positive integer'):MethodOptions(num_iterations=iterations)


@pytest.mark.parametrize('tolerance',[0,-1,np.nan,np.inf,-np.inf])
def test_invalid_tolerance(tolerance):
    with pytest.raises(ValueError,match='convergence_tol'):MethodOptions(convergence_tol=tolerance)


@pytest.mark.parametrize('optimizer', ['trust', 'bfgs'])
def test_optimizer_validation(optimizer):
    with pytest.raises(ValueError,match='optimizer'):MethodOptions(optimizer=optimizer)

def test_cli_rejects_removed_bfgs():
    with pytest.raises(SystemExit):
        build_parser().parse_args(['reml', 'trait.sumstats', 'out', '--annot-dir', 'annotations', '--optimizer', 'bfgs'])


def test_cli_defaults_firth_alias_and_overrides():
    parser=build_parser();base=['reml','trait.sumstats','out','--annot-dir','annotations']
    args=parser.parse_args(base)
    assert (args.optimizer,args.information_penalty,args.num_iterations,args.convergence_tol)==('ai',0.,100,.001)
    assert parser.parse_args(base+['--firth']).information_penalty==1.
    assert parser.parse_args(base+['--information-penalty','.25']).information_penalty==.25
    args=parser.parse_args(base+['--optimizer','ai','--num-iterations','7','--convergence-tol','.02'])
    assert (args.optimizer,args.num_iterations,args.convergence_tol)==('ai',7,.02)


def test_convergence_csv_distinguishes_accepted_steps_from_iterations(tmp_path):
    import csv
    from graphld.cli import write_convergence_results

    # A refresh-only iteration does not add an accepted parameter point.
    results = {
        'likelihood_history': [1., 2.],
        'penalty_history': [0., 0.],
        'objective_history': [1., 2.],
        'log': {
            'optimizer': 'ai', 'num_iterations': 3,
            'trust_region_lambdas': [0., 0.],
            'evaluation_counts': {'full': 4, 'precise': 2},
            'evaluation_seconds': {'full': 1., 'precise': .5},
        },
    }
    output = tmp_path / 'convergence.csv'
    write_convergence_results(str(output), results)
    with output.open() as stream:
        rows = list(csv.reader(stream))
    assert rows[1][rows[0].index('num_iterations')] == '3'
    assert rows[3][0] == 'accepted_step'
    assert rows[4][:2] == ['0', '1.0']  # Initial point.
    assert rows[5][:2] == ['1', '2.0']
