import subprocess, shlex, os, sys
from joblib import Parallel, delayed


def problem_file(theorem, list_of_premises, statements, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    input_filename = os.path.join(directory, theorem + "__" +
                         str(len(list_of_premises)) + "_premises" + ".E_input")
    with open(input_filename, "w") as problem:
        print(statements[theorem].replace("axiom,", "conjecture,"),
              file=problem)
        for p in list_of_premises:
            print(statements[p], file=problem)
    return input_filename

def problem_file_rerun(output_filename, directory):
    lines = lines_txt(output_filename)
    input_filename = output_filename.replace(".E_output", "__rerun.E_input")
    with open(input_filename, "w") as problem:
        for l in lines:
            if "file" in l:
                print(l, file=problem)
    return input_filename

def run_E_prover(input_filename, output_filename, cpu_time=10):
    output = open(output_filename, "w")
    subprocess.call([
        "E/PROVER/eprover",
        "--auto",
        "--free-numbers",
        "-s",
        "-R",
        "--cpu-limit=" + str(cpu_time),
        "--memory-limit=2000",
        "--print-statistics",
        "-p",
        "--tstp-format",
        input_filename],
        stdout=output, stderr = open(os.devnull, 'w'))
    output.close()

def used_premises(filename):
    lines = lines_txt(filename)
    return tuple(l.split(", ")[0].replace("fof(", "")
                    for l in lines if "axiom" in l and "file" in l)

def proof(theorem, list_of_premises, statements, directory, cpu_time,
                        rerun=True, cpu_time_rerun=1):
    assert not theorem in set(list_of_premises)
    input_filename = problem_file(theorem, list_of_premises,
                                          statements, directory)
    output_filename = input_filename.replace("input", "output")
    run_E_prover(input_filename, output_filename, cpu_time)
    premises = used_premises(output_filename)
    if "Proof found!\n# SZS status Theorem" in read_txt(output_filename):
        if rerun: # we will rerun twice!
            input_filename = problem_file_rerun(output_filename, directory)
            output_filename = input_filename.replace("input", "output")
            run_E_prover(input_filename, output_filename, cpu_time_rerun)
            input_filename = problem_file_rerun(output_filename, directory)
            output_filename = input_filename.replace("input", "output")
            run_E_prover(input_filename, output_filename, cpu_time_rerun)
            premises_rerun = used_premises(output_filename)
            if "Proof found!\n# SZS status Theorem" in read_txt(output_filename):
                premises = premises_rerun
            else:
                "Failed to find a proof after reruning twice."
        print("Proof of theorem {} FOUND with {} premises".format(
                theorem, len(premises)))
        return premises
    else:
        print("Proof of theorem {} NOT found with {} premises".format(
                theorem, len(list_of_premises)))
        return False

def proof_from_ranking(theorem, ranking_of_premises, statements,
                   directory, n_premises, cpu_time, rerun, cpu_time_rerun):
    proofs = [proof(theorem, ranking_of_premises[:i], statements,
              directory, cpu_time, rerun, cpu_time_rerun) for i in n_premises]
    return [set(prf) for prf in (set(proofs) - {False})]

# wrapper for proof_from_ranking() -- useful for doing parallelization
def prf(theorem, ranking_of_premises, statements,
        directory, n_premises, cpu_time, rerun, cpu_time_rerun):
    return (theorem, proof_from_ranking(theorem, ranking_of_premises,
      statements, directory, n_premises, cpu_time, rerun, cpu_time_rerun))

def atp_evaluation(rankings, statements,
                   directory, logfile="", n_jobs=-1, cpu_time=10,
                   n_premises=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                    rerun=True, cpu_time_rerun=1):
    n_premises = [i for i in n_premises if i <=
                len(rankings[list(rankings)[0]])]
    with Parallel(n_jobs=n_jobs) as parallel:
        dprf = delayed(prf)
        proven = parallel(
            dprf(thm, rankings[thm], statements,
                 directory, n_premises, cpu_time, rerun, cpu_time_rerun)
           for thm in rankings)
    if logfile:
        proven_n = sum([bool(i[1]) for i in proven])
        proven_avg = proven_n / len(proven)
        printline("    Number of proved theorems: {}".format(proven_n),
                  logfile, time=True)
        printline("    Part of proved theorems: {:.3f}".format(proven_avg),
                  logfile, time=True)
    return dict(proven)

