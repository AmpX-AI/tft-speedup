from __future__ import annotations

import functools
import multiprocessing as mp
from queue import Empty


def subprocessify(default_run_in_subprocess=False, start_method="spawn"):
    """A decorator factory adding a kwarg to a function that makes it run in a subprocess.

    Based on https://gist.github.com/joezuntz/e7e7764e5b591ed519cfd488e20311f1

    This can be useful when you have a function that may segfault.

    Args:
        default_run_in_subprocess: If no run_in_subprocess specified, use this as default.
        start_method: Use spawn or fork method.
    """

    def subprocessify_f(f):
        # This functools.wraps command makes this whole thing work as a well-behaved decorator,
        # maintaining the original function name and docstring.
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Two extra kwargs, one public, are used for implementation. One indicates that the user wants to run
            # in a subprocess, the other that the function is being called in the subprocess already.
            # The latter is the queue where the result gets posted to.
            run_in_subprocess = kwargs.pop("run_in_subprocess", default_run_in_subprocess)
            subprocess_timeout = kwargs.pop("subprocess_timeout", None)
            queue = kwargs.pop("__queue", None)

            # create the machinery python uses to fork a subprocess and run a function in it.
            if run_in_subprocess:
                ctx = mp.get_context(start_method)
                q = ctx.Queue()
                p = ctx.Process(target=wrapper, args=args, kwargs={"run_in_subprocess": False, "__queue": q, **kwargs})
                # Because the use of this is avoiding crashes, rather than performance / parallelization
                # we wait for the subprocess result immediately.
                p.start()
                try:
                    result = q.get(timeout=subprocess_timeout)
                    p.join()
                except Empty:
                    p.terminate()
                    raise TimeoutError("Function {} timed out with args: {}, {}".format(f, args, kwargs))
                # Pass on any exception raised in the subprocess
                if isinstance(result, BaseException):
                    raise result
                return result
            else:
                # Run the function.  Eiher we are in the subprocess already or the user
                # does not want to run in the subproc.
                try:
                    result = f(*args, **kwargs)
                except BaseException as error:
                    # If running in standard mode just raise exceptions as normal
                    if queue is None:
                        raise
                    # Otherwise we pass the exception back to the caller in place of the result
                    result = error

                # This is optional, so the function can still just be called normally with no effect.
                if queue is not None:
                    queue.put(result)

                return result

        return wrapper

    return subprocessify_f
