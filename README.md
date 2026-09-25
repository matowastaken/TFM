TFM/                           # repo ya existente, privado en GitHub
├── agents/
│   ├── __init__.py
│   ├── technical_agent.py     # 3.3 (← technical_prototype.py)
│   ├── sentiment_agent.py     # 3.3 (← sentiment_prototype.py)
│   ├── fundamental_agent.py   # 3.3 (← fundamental_prototype.py)
│   └── trader_agent.py        # 3.3 (← trader_prototype.py)
├── llm/
│   ├── __init__.py
│   └── call_llm.py            # 3.2
├── strategy/
│   ├── __init__.py
│   ├── orchestrator.py        # 3.4
│   └── ablation_strategy.py   # 3.5
├── config/
│   └── ablation.yaml          # 3.4, con cherry_pick_no_fundamental ya corregido
├── results/                   # solo copias consolidadas (Fase 4.3)
├── prototypes/                # scripts de Fase 2
├── tests/                     # 3.3
├── .env                       # ya existente, confirma que sigue fuera del repo
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md