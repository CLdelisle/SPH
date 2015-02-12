# Comparison of timing methods and frameworks

## [line_profiler](https://github.com/rkern/line_profiler)
Provides line by line breakdown of time spent in a function

#### Usage
Install with ```pip install line_profiler```
Annotate functions to be profiled with @profile.

Example with [timingTests.py]()
```bash
cd line_profiler/
kernprof -l timingTests.py
# view results with:
python -m line_profiler timingTests.py.lprof
```

Example output:
![](http://i.imgur.com/nnUkNbq.png)
