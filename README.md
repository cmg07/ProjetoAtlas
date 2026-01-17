# ATLAS v5 — Premium Financial Intelligence Terminal (BR-First)

ATLAS v5 é um terminal financeiro Streamlit voltado para decisões operacionais com foco em Brasil-first, suporte offline e motores internos para análise técnica, risco e execução simulada.

## Requisitos

- Python 3.10+
- Windows, macOS ou Linux

## Instalação (Windows)

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate
pip install -r requirements.txt
```

## Execução

```powershell
streamlit run app.py
```

## Observações importantes

- O universo é carregado de `data/universe.csv` (offline). Se o arquivo não existir, o app cria um universo padrão local.
- O download de dados só ocorre quando você clica em **Analisar Ativo**.
- Fontes: Yahoo (primário), BCB (fallback para FX e juros), Bridge/Local (arquivos CSV).

## Estrutura sugerida de dados

- `data/universe.csv`: universo local (obrigatório).
- `data/bridge_prices.csv`: preços de backup com colunas `date,ticker,open,high,low,close,volume,bridge_key`.
- `data/local_prices.csv`: preços locais com colunas `date,ticker,open,high,low,close,volume`.
