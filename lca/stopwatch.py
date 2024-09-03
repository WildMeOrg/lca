from timeit import default_timer as timer
from contextlib import contextmanager

@contextmanager
def stopwatch(start_print="Start stopwatch",
                 final_print="Elapsed time %2.4f",
                 print_function=print):
    tic = timer()
    print_function(start_print)
    try:
        yield
    finally:
        toc = timer()
        print_function(final_print % (toc-tic))
