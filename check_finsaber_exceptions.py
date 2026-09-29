# check_finsaber_exceptions.py
import inspect, re, pathlib, backtest.finsaber as m, backtest.toolkit.backtest_framework_iso as f

for mod in (m, f):
    src = pathlib.Path(inspect.getsourcefile(mod)).read_text().splitlines()
    print("=====", mod.__name__)
    for i, line in enumerate(src):
        if re.search(r"on_data\(|\.run\(|except", line):
            print(f"{i+1:5d}: {line}")