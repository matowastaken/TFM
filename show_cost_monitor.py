# show_cost_monitor.py
import inspect
import backtest.toolkit.llm_cost_monitor as m

print("Fichero:", inspect.getsourcefile(m))
for name in ("add_openai_cost_from_response", "add_llm_cost", "_openai_cost"):
    fn = getattr(m, name, None)
    print("\n=====", name)
    print(inspect.getsource(fn) if fn else "NO EXISTE")