# premises
Python package for doing Machine Learning experiments for Premise Selection task.

Provides ATP evaluation with [E prover](http://wwwlehre.dhbw-stuttgart.de/~sschulz/E/E.html). Contains bunch of example experiments on data originating from [Mizar Mathematical Library](http://mizar.org/library/).

## Requirements
1. Python 3
2. `xgboost` Python package. Can be installed by running
```pip3 install xgboost```
3. The newest version of `joblib` parallelization package (`joblib-0.11.1.dev0`) so far available only on GitHub. Can be installed by running:
```pip3 install http://github.com/joblib/joblib/archive/master.zip```
(Version `joblib-0.11` does not provide `loky` backend which works properly with `xgboost`.)
4. E prover. Can be installed by running:
```
wget http://wwwlehre.dhbw-stuttgart.de/~sschulz/WORK/E_DOWNLOAD/V_2.0/E.tgz
tar -xzf E.tgz
cd E
./configure
make
```
After installation there is also needed to set `EPROVER` environment variable to make known for our package where E prover is. Assuming you are still in `E` directory run:
```export EPROVER=`realpath PROVER/eprover` ```
To make this variable permanent -- put the line above to your `.bashrc` / `.zshrc` / ... changing ``realpath PROVER/eprover``
to `'path/to/E/PROVER/eprover'`


