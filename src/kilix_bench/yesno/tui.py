"""Keyboard terminal monitor for the shared YES/NO journals."""
import curses
import json
import sys
import textwrap
import time
from . import cli, worker
from .comparison import runner_labels
from .models import effort_options
from .records import read_run, save_rankings


class Monitor:
    def __init__(self, screen, folder):
        self.screen, self.folder = screen, folder
        self.page, self.cursor, self.stack = "Watch", 0, []
        self.provider = self.model = self.effort = self.message = ""
        self.group, self.path, self.detail = None, None, None
        self.invalid = False
        self.models, self.entries = [], []
        self.cache, self.refresh = None, 0
        self.red = curses.A_BOLD
        if curses.has_colors():
            curses.start_color()
            try:
                curses.use_default_colors()
                background = -1
            except curses.error:
                background = curses.COLOR_BLACK
            curses.init_pair(1, curses.COLOR_RED, background)
            self.red |= curses.color_pair(1)

    def go(self, page):
        self.stack.append((self.page, self.cursor))
        self.page, self.cursor, self.message = page, 0, ""

    def put(self, y, x, text, attr=0):
        h,w = self.screen.getmaxyx()
        if y < h and x < w-1:
            text = " ".join(str(text).splitlines())
            try:
                self.screen.addnstr(y, x, text, w-x-1, attr)
            except curses.error:
                pass

    def board(self):
        if self.cache is None or time.monotonic() >= self.refresh:
            self.cache = save_rankings(self.folder)
            self.refresh = time.monotonic()+2
        return self.cache

    def content(self):
        self.entries = []
        if self.page in ("Providers", "Models", "Effort"):
            self.entries = list(worker.PROVIDERS) if self.page == "Providers" else self.models
            return "Enter selects   Esc back", [(name, False) for name in self.entries]
        if self.page == "Rankings":
            self.entries = self.board()["models"]
            width = max(12, min(45, self.screen.getmaxyx()[1]-70))
            def format_row(fields):
                rank, provider, model, effort, runs, done, inv, correct, percent = fields
                return f"{rank:<4} {provider:<9} {model:<{width}.{width}} {effort:<10} {runs:>4} {done:>5} {inv:>7} {correct:>9} {percent:>6}"
            return format_row(cli.RANK_COLUMNS), [(format_row(cli.ranking_fields(g)), (g["invalid_answers"] or 0)>0)
                                                  for g in self.entries]
        if self.page == "Runs":
            self.entries = self.group["history"]
            return "Started (UTC)         Status        Correct/answered   Invalid   Journal", [
                (f'{(r["started"] or "-")[:19]:<20}  {r["status"]:<12}  {r["correct"]}/{r["answered"]}'
                 f'              {r["invalid"] if r["invalid"] is not None else "-"}    {r["file"]}', (r["invalid"] or 0)>0)
                for r in self.entries]
        if self.page == "Machines":
            self.entries = self.group["machine_comparisons"]
            return "Runner                         Runs  Invalid  Avg %  Wall s  s/answer", [
                (f'{runner_labels(r)["machine_label"]:<30.30} {r["runs"]:>4} {cli.invalid_count(r):>8} '
                 f'{cli.score(r["avg_percent"]):>6} {cli.score(r["mean_wall_seconds"]):>7} '
                 f'{r["mean_answer_seconds"] if r["mean_answer_seconds"] is not None else "-"}', (r["invalid_answers"] or 0)>0)
                for r in self.entries]
        if self.page == "Detail":
            r = self.detail
            info = [r.get("question", ""), r.get("context",""), "Expected  "+r.get("expected",""),
                    "Reply  "+json.dumps(r.get("raw_reply", r.get("reply","")),ensure_ascii=False),
                    "Verdict  "+r["verdict"], r.get("rationale","")]
            lines = [s for block in info if block for s in textwrap.wrap(block, max(10,self.screen.getmaxyx()[1]-7))]
            self.entries = lines
            return f'Question {r["n"]}', [(s,False) for s in lines]
        path = cli.journal(self.folder) if self.page == "Watch" else self.path
        run = read_run(path) if path and path.is_file() else None
        if not run:
            return "No question journal available.", []
        self.entries = [a for a in reversed(cli.answers(path)) if not self.invalid or a["verdict"] == "invalid"]
        status = (f'{run["provider"]} / {run["model"] or "default"}  {cli.effort_name(run)}  {run["status"]}    '
                  f'{run["answered"]}/{run["total"]} answered    {run["correct"]} correct    {run["invalid"]} invalid')
        return status, [(f'{a["n"]:>4}  {a["verdict"]:<8}  {a.get("reply","")[:18]:<18}  {a.get("question","")}',
                         a["verdict"]=="invalid") for a in self.entries]

    def draw(self):
        self.screen.erase()
        h,w = self.screen.getmaxyx()
        self.put(0, 1, "YES / NO BENCH   "+self.page, curses.A_BOLD)
        heading, rows = self.content()
        self.put(2, 2, heading, curses.A_BOLD)
        self.cursor = min(max(0,self.cursor), max(0,len(rows)-1))
        height = max(1,h-8)
        top = min(max(0,self.cursor-height//2), max(0,len(rows)-height))
        for i, (line, flagged) in enumerate(rows[top:top+height], top):
            attr = self.red if flagged else 0
            if i == self.cursor:
                attr |= curses.A_REVERSE
            self.put(4+i-top, 1, ("! " if flagged else "  ")+line, attr)
        if self.page == "Machines" and self.entries:
            labels = runner_labels(self.entries[self.cursor])
            self.put(h-4, 2, labels["cpu_label"]+"   "+labels["gpu_label"]+"   RAM "+labels["ram_label"])
        controls = {"Watch":"N New run   B Rankings   I Invalid   Enter Details   S Stop",
                    "Rankings":"Enter Runs   U Refresh   ! Invalid YES/NO reply",
                    "Runs":"Enter Answers   C Machines", "Machines":"C Runs",
                    "Answers":"I Invalid   Enter Details", "Models":"Enter Select",
                    "Providers":"Enter Select", "Effort":"Enter Start run"}.get(self.page,"")
        self.put(h-2, 1, self.message or controls)
        self.put(h-1, 1, ("Invalid replies only   " if self.invalid and self.page in ("Watch","Answers") else "")
                 +"↑/↓ Select   Esc Back   Q Quit")
        self.screen.refresh()

    def start(self):
        path = worker.start_run(self.provider,self.model,self.folder,self.effort)
        self.page,self.cursor,self.stack = "Watch",0,[]
        self.invalid,self.cache = False,None
        self.message = "Watching "+path.name

    def enter(self):
        if not self.entries:
            return
        choice = self.entries[self.cursor]
        if self.page == "Providers":
            self.provider = choice
            self.go("Models")
            self.models = []
            self.put(2, 2, "Reading available models…"); self.screen.refresh()
            try:
                self.models = cli.model_names(choice)
            finally:
                if choice != "kilix":
                    self.models = (["Use configured default"] if choice in worker.HOSTED else []) + ["Enter model ID or path…"] + self.models
        elif self.page == "Models":
            self.model = "" if choice == "Use configured default" else choice
            if choice == "Enter model ID or path…":
                self.screen.erase()
                self.put(0, 1, "Model ID or GGUF path")
                self.screen.refresh()
                self.screen.timeout(-1)
                curses.echo()
                try:
                    self.model = self.screen.getstr(2,1,4095).decode().strip()
                finally:
                    curses.noecho()
                    self.screen.timeout(700)
                if not self.model:
                    return
            self.effort = ""
            if self.provider in worker.HOSTED:
                levels = effort_options(self.provider,self.model,discover=True)[0]
                self.go("Effort")
                self.models = ["Use configured default", *levels]
            else:
                self.start()
        elif self.page == "Effort":
            self.effort = "" if self.cursor == 0 else choice
            self.start()
        elif self.page == "Rankings":
            self.group = choice; self.go("Runs")
        elif self.page == "Runs":
            self.path = cli.journal(self.folder,choice["file"]); self.go("Answers")
        elif self.page in ("Watch","Answers"):
            self.detail = choice; self.go("Detail")

    def run(self):
        self.screen.keypad(True)
        self.screen.timeout(700)
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        while True:
            try:
                self.draw()
                key = self.screen.getch()
                if key in (ord("q"),ord("Q")):
                    return 0
                if key == 27:
                    if self.stack:
                        previous,self.cursor = self.stack.pop()
                        if previous=="Models" and self.page=="Effort":
                            self.models = ["Use configured default","Enter model ID or path…",*cli.model_names(self.provider)]
                        self.page,self.message = previous,""
                    else:
                        return 0
                elif key in (curses.KEY_UP,curses.KEY_DOWN,curses.KEY_PPAGE,curses.KEY_NPAGE):
                    self.cursor += {curses.KEY_UP:-1,curses.KEY_DOWN:1,curses.KEY_PPAGE:-10,curses.KEY_NPAGE:10}[key]
                elif key in (10,13,curses.KEY_ENTER):
                    self.enter()
                elif self.page=="Watch" and key in (ord("n"),ord("N")):
                    self.go("Providers")
                elif self.page=="Watch" and key in (ord("b"),ord("B")):
                    self.cache=None; self.go("Rankings")
                elif self.page=="Rankings" and key in (ord("u"),ord("U")):
                    self.cache=None
                elif self.page in ("Watch","Answers") and key in (ord("i"),ord("I")):
                    self.invalid=not self.invalid; self.cursor=0
                elif self.page=="Watch" and key in (ord("s"),ord("S")):
                    self.message = "Stop requested." if worker.stop_run(self.folder) else "No running benchmark."
                elif self.page=="Runs" and key in (ord("c"),ord("C")):
                    self.go("Machines")
                elif self.page=="Machines" and key in (ord("c"),ord("C")):
                    self.page,self.cursor = self.stack.pop()
            except (ValueError,OSError,RuntimeError,KeyError) as error:
                self.message=str(error)
                self.put(self.screen.getmaxyx()[0]-2,1,self.message); self.screen.refresh()


def launch(folder):
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise ValueError("The TUI needs a terminal. Use rankings or status for pipeable output.")
    return curses.wrapper(lambda screen: Monitor(screen,folder).run())
