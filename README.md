# Comparison of timing methods and frameworks

## [line_profiler](https://github.com/rkern/line_profiler)
Provides line by line breakdown of time spent in a function

#### Usage
Install with ```pip install line_profiler```
Annotate functions to be profiled with @profile.

Example with [timingTests.py](https://github.com/QuarkStar/SPH/blob/timingFrameworks/line_profiler/timingTests.py)
```bash
cd line_profiler/
kernprof -l timingTests.py
# view results with:
python -m line_profiler timingTests.py.lprof
```
Output:
![](http://i.imgur.com/nnUkNbq.png)

## [cProfile](https://docs.python.org/2/library/profile.html)
C extension with reasonable overhead that makes it suitable for profiling long-running programs.

### Usage
cProfile ships with python, no libs to install.
```python
import cProfile
cProfile.run('someMethod())
```

Example with [cProfileTest.py](https://github.com/QuarkStar/SPH/blob/timingFrameworks/cProfile/cProfileTest.py)
Output:
```bash
cd cProfile/
python cProfileTest.py 
```
Output:
![](http://i.imgur.com/DhJfRjw.png)
