"""Exercise the checkout entry points as a user, including a real curses terminal."""
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import struct
import subprocess
import sys
import tempfile
import termios
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TerminalTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.state = Path(temp.name)
        self.folder = self.state/"bench/yesno"
        self.folder.mkdir(parents=True)
        self.env = {**os.environ, "PUZZLLM_STATE": str(self.state), "TERM":"xterm-256color", "PYTHONPATH":str(ROOT/"src")}
        rows = [{"event":"begin","provider":"codex","model":"example","effort":"high","total":2},
                {"event":"answer","n":1,"question":"Is two even?","expected":"YES","reply":"YES","verdict":"correct","seconds":1},
                {"event":"answer","n":2,"question":"Is one even?","expected":"NO","reply":"perhaps","verdict":"invalid","seconds":2},
                {"event":"end","wall_seconds":5}]
        self.name = "20261001T100000.000000Z-aabbccdd.jsonl"
        (self.folder/self.name).write_text("".join(json.dumps(r)+"\n" for r in rows))

    def cli(self, *args):
        return subprocess.check_output([ROOT/"bin/kilix-yesno",*args],env=self.env,text=True)

    def test_plain_and_machine_readable_cli_read_the_same_rankings(self):
        board=json.loads(self.cli("rankings","--json"))
        self.assertEqual(board,json.loads((self.folder/"rankings.json").read_bytes()))
        self.assertEqual(board["models"][0]["invalid_answers"],1)
        self.assertIn("! 1\tcodex\texample\thigh",self.cli("rankings"))
        self.assertEqual(json.loads(self.cli("answers",self.name,"--invalid","--json"))["reply"],"perhaps")
        self.assertEqual(json.loads(self.cli("status","--json"))["invalid"],1)
        self.assertEqual(json.loads(self.cli("runs","--json"))["file"],self.name)
        self.assertIn("high",self.cli("compare"))
        self.assertIn("needle3",self.cli("models","kilix"))
        self.assertEqual(subprocess.check_output([sys.executable,"-m","kilix_bench","yesno","rankings"],
                                                 env=self.env,text=True),self.cli("rankings"))

    def test_tui_browses_flags_and_saved_invalid_replies(self):
        master, slave = pty.openpty()
        fcntl.ioctl(slave,termios.TIOCSWINSZ,struct.pack("HHHH",32,140,0,0))
        proc = subprocess.Popen([ROOT/"bin/kilix-yesno-tui"], stdin=slave, stdout=slave, stderr=slave,env=self.env)
        os.close(slave)
        pending = b""
        def expect(word):
            nonlocal pending
            until=time.monotonic()+8
            while word not in pending and time.monotonic()<until:
                if select.select([master],[],[],0.1)[0]:
                    pending += os.read(master,65536)
            self.assertIn(word,pending)
            pending = pending.split(word,1)[1]
        try:
            expect(b"kilix-k12-computer use (yes/no)")
            os.write(master,b"d"); expect(b"BoolQ (YES/NO)")
            os.write(master,b"\x1bOB\r"); expect(b"BoolQ (YES/NO)"); expect(b"Watch")
            os.write(master,b"b"); expect(b"Rankings")
            os.write(master,b"\x1b"); expect(b"Watch")
            os.write(master,b"d"); expect(b"kilix-k12-computer use (yes/no)")
            os.write(master,b"\x1bOA\r"); expect(b"Watch")
            os.write(master,b"b"); expect(b"Rankings"); expect(b"example")
            os.write(master,b"\r"); expect(b"Runs"); expect(b"Journal")
            os.write(master,b"c"); expect(b"Machines")
            os.write(master,b"c"); expect(b"Runs")
            os.write(master,b"\r"); expect(b"Answers")
            os.write(master,b"i"); expect(b"Invalid replies only")
            os.write(master,b"\r"); expect(b"Detail"); expect(b"perhaps")
            os.write(master,b"q")
            self.assertEqual(proc.wait(timeout=5),0)
        finally:
            if proc.poll() is None:
                proc.terminate(); proc.wait(timeout=5)
            os.close(master)
