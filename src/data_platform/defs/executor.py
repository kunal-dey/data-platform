"""Default executor for this code location.

Small EC2 hosts (~1GB RAM) OOM-kill multiprocess children (SIGKILL →
ChildProcessCrashException). Prefer in-process execution for all jobs here.
"""

import dagster as dg


@dg.definitions
def executor_defs():
    return dg.Definitions(executor=dg.in_process_executor)
