import os, sys, subprocess, shlex, tempfile, shutil
from joblib import Parallel, delayed
from .utils import mkdir_if_not_exists, readlines, printline
from .data_structures import Proofs


# TODO remeber to give proper instalation instructions
PATH_TO_EPROVER = os.environ['EPROVER']

# TODO do scrutiny if 'conjecture' being few lines lower is handled OK
def problem_file(theorem, list_of_premises, statements, dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    input_filename = os.path.join(dirpath, theorem + '__' +
                         str(len(list_of_premises)) + '_premises' + '.E_input')
    with open(input_filename, 'w') as problem:
        print(statements[theorem].replace('axiom,', 'conjecture,'),
              file=problem)
        for p in list_of_premises:
            print(statements[p], file=problem)
    return input_filename

def problem_file_rerun(output_filename, dirpath):
    lines = readlines(output_filename)
    input_filename = output_filename.replace('.E_output', '__rerun.E_input')
    with open(input_filename, 'w') as problem:
        for l in lines:
            if 'file' in l:
                print(l, file=problem)
    return input_filename

def run_E_prover(input_filename, output_filename, cpu_time=10):
    output = open(output_filename, 'w')
    subprocess.call([
        PATH_TO_EPROVER,
        '--auto',
        '--free-numbers',
        '-s',
        '-R',
        '--cpu-limit=' + str(cpu_time),
        '--memory-limit=2000',
        '--print-statistics',
        '-p',
        '--tstp-format',
        input_filename],
        stdout=output, stderr = open(os.devnull, 'w'))
    output.close()

def used_premises(filename):
    lines = readlines(filename)
    return tuple(l.split(', ')[0].replace('fof(', '')
                    for l in lines if 'axiom' in l and 'file' in l)

def proof(theorem, ranking, statements, dirpath, params):
    cpu_time = params['cpu_time'] if 'cpu_time' in params else 10
    minimize = params['minimize'] if 'minimize' in params else True
    assert not theorem in set(ranking)
    input_filename = problem_file(theorem, ranking, statements, dirpath)
    output_filename = input_filename.replace('input', 'output')
    run_E_prover(input_filename, output_filename, cpu_time)
    premises = used_premises(output_filename)
    lines = readlines(output_filename)
    if "# Proof found!" in lines and "# SZS status Theorem" in lines:
        if rerun: # we will rerun twice!
            input_filename = problem_file_rerun(output_filename, dirpath)
            output_filename = input_filename.replace("input", "output")
            run_E_prover(input_filename, output_filename, cpu_time_rerun)
            input_filename = problem_file_rerun(output_filename, dirpath)
            output_filename = input_filename.replace("input", "output")
            run_E_prover(input_filename, output_filename, cpu_time_rerun)
            premises_rerun = used_premises(output_filename)
            if "Proof found!\n# SZS status Theorem" in \
               readlines(output_filename):
                premises = premises_rerun
            else:
                "Failed to find a proof after reruning twice."
        print("Proof of theorem {} FOUND with {} premises".format(
                theorem, len(premises)))
        return premises
    else:
        print("Proof of theorem {} NOT found with {} premises".format(
                    theorem, len(ranking)))
        return False

def proof_from_ranking(theorem, ranking, statements, dirpath, params):
    n_premises = params['n_premises'] if 'n_premises' in params else \
        [i for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] if i <= len(ranking)]
    # TODO what if len(ranking) == 0 ?
    proofs = [proof(theorem, ranking[:i], statements, dirpath, params)
              for i in n_premises]
    return [set(prf) for prf in (set(proofs) - {False})]

# wrapper for proof_from_ranking() -- useful for doing parallelization
def pfr(t, r, s, d, p): return (t, proof_from_ranking(t, r, s, d, p))

def atp_evaluation(rankings=None, statements=None, params={}, dirpath='',
                   verbose=True, logfile='', n_jobs=-1):
    if verbose or logfile:
        message = "ATP evaluation started..."
        printline(message, logfile, verbose)
    if dirpath:
        mkdir_if_not_exists(dirpath)
        dir_atp = dirpath
    else:
        dir_atp = tempfile.mkdtemp()
    with Parallel(n_jobs=n_jobs) as parallel:
        dpfr = delayed(pfr)
        proven = parallel(dpfr(thm, rankings[thm], statements, dir_atp, params)
                               for thm in rankings)
    if logfile:
        proven_n = sum([bool(i[1]) for i in proven])
        proven_avg = proven_n / len(proven)
        printline("    Number of proved theorems: {}".format(proven_n),
                  logfile, verbose)
        printline("    Percentage of proved theorems: {:.2%}".format(proven_avg),
                  logfile, verbose)
    if not dirpath:
        shutil.rmtree(dir_atp)
    return Proofs(from_dict = dict(proven))

