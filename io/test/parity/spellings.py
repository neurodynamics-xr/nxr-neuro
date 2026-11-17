"""The old two-word spellings of nxr-neuro's command line, for the parity tests.

nxr's Python reference still speaks the old spellings (`convert subject`, `dataset create`, ...), so the tests keep them for it. nxr-neuro refuses
them (`cli/command_table.cpp`, `alias: refused`) and takes the new spelling: `to_io` rewrites the leading words of an argument list for the binary.
"""

OLD_TO_NEW = {
    ("convert", "convert"): ("import", "condition"),
    ("convert", "list"): ("import", "list"),
    ("convert", "scan"): ("import", "scan"),
    ("convert", "subject"): ("import", "subject"),
    ("convert", "surface"): ("import", "surface"),
    ("convert", "mri"): ("import", "mri"),
    ("convert", "inverse"): ("import", "inverse"),
    ("convert", "template"): ("import", "template"),
    ("convert", "templates"): ("import", "templates"),
    ("convert", "group"): ("import", "group"),
    ("convert", "resample"): ("import", "resample"),
    ("convert", "badsegments"): ("import", "badsegments"),
    ("convert", "events"): ("import", "events"),
    ("convert", "map"): ("import", "map"),
    ("dataset", "create"): ("create", "dataset"),
    ("dataset", "list"): ("list", "datasets"),
    ("dataset", "show"): ("show", "dataset"),
    ("dataset", "delete"): ("delete", "dataset"),
    ("dataset", "compact"): ("compact", "dataset"),
    ("dataset", "recover"): ("recover", "dataset"),
    ("subject", "create"): ("create", "subject"),
    ("subject", "list"): ("list", "subjects"),
    ("subject", "show"): ("show", "subject"),
    ("subject", "delete"): ("delete", "subject"),
    ("surface", "add"): ("add", "surface"),
    ("pyramid", "reduce"): ("reduce", "pyramid"),
    ("pyramid", "spatial-bands"): ("list", "spatial-bands"),
    ("measure", "query"): ("query", "measurements"),
    ("measure", "export"): ("export", "measurements"),
    ("crossspectrum", "read"): ("read", "crossspectrum"),
    ("eigen", "solve"): ("run", "eigenmodes"),
}


def to_io(args):
    """The argument list with its old two-word spelling (if any) replaced by the new one; the rest untouched."""
    args = [str(a) for a in args]
    if len(args) >= 2 and (args[0], args[1]) in OLD_TO_NEW:
        new = OLD_TO_NEW[(args[0], args[1])]
        return [*new, *args[2:]]
    return args
