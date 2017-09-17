import fuzzylite.operation as op
import fuzzylite.rule
import fuzzylite.term


class Exporter(object):
    pass


class FllExporter(Exporter):
    __slots__ = "indent", "separator"

    def __init__(self, indent="  ", separator="\n"):
        self.indent = indent
        self.separator = separator

    def term(self, term) -> str:
        return "term: %s %s %s" % (op.valid_name(term.name), term.__class__.__name__, term.parameters())

    def rule(self, rule: fuzzylite.rule.Rule) -> str:
        return "rule: %s" % rule.text
