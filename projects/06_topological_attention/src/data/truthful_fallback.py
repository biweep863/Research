"""Used by gpu_full config when no curated pairs are provided. Returns a
small built-in list so the pipeline can still run end-to-end on GPU."""


def fallback():
    return [
        {"prompt": "The capital of Japan is",
         "true_cont": " Tokyo.", "false_cont": " Madrid."},
        {"prompt": "Pi is approximately",
         "true_cont": " 3.14.", "false_cont": " 9.99."},
    ]
