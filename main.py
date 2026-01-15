import asyncio  # CORREÇÃO: Importação obrigatória para rodar funções async
from ingestion import DataIngestionISA
from agents import GeopoliticalISA, RuinTheoryISA
from advanced_modules import SupplyChainISA, MacroISA
from math_engines import GameTheoryISA, KellyCriterion

async def run_atlas():
    # 1. CAMADA DE ENTRADA (Ingestão)
    ticker_target = "PETR4.SA"
    ingestor = DataIngestionISA(ticker_target)
    data = ingestor.fetch_all()
    
    if data['history'].empty:
        print(f"❌ Erro: Dados de preço para {ticker_target} não encontrados.")
        return

    # Cálculo de retornos para os modelos estatísticos
    returns = data['history']['Close'].pct_change().dropna()
    
    # 2. PLANEJAMENTO & 3. CAMADA DE ANÁLISE (Agentes ISAs)
    # Módulo Geopolítico: EGARCH (1,1) e Efeito Alavancagem
    geo = GeopoliticalISA().analyze(returns)
    
    # Módulo Atuarial: Sparre Andersen e Probabilidade de Ruína
    info_clean = {
        'totalCash': data['info'].get('totalCash', 1),
        'totalDebt': data['info'].get('totalDebt', 1)
    }
    ruin = RuinTheoryISA().evaluate(info_clean, returns.std())
    
    # Módulo Supply Chain: Rede HT-GNN (Simulação de vizinhos)
    supply = SupplyChainISA().check_propagation()
    
    # Módulo Macro: All Weather Framework
    macro = MacroISA().get_quadrant()
    
    # 4. CAMADA DE INTEGRAÇÃO & 5. CAMADA DE DECISÃO
    # Teoria dos Jogos: Nash / Lasry-Lions
    game = GameTheoryISA().lasry_lions_balance(1000)
    
    # Dimensionamento Kelly (Segurança vs Rentabilidade)
    kelly_size = KellyCriterion().sizing(win_rate=0.55, win_loss_ratio=2.0)
    
    # --- OUTPUT INSTITUCIONAL ---
    print(f"\n" + "="*50)
    print(f"🏛️ VEREDITO FINAL ATLAS - {ticker_target}")
    print("="*50)
    print(f"1. REGIME MACRO: {macro}")
    print(f"2. GEOPOLÍTICA (EGARCH): {geo['status']} (Gamma: {geo['leverage_gamma']:.2f}x)")
    print(f"3. SOLVÊNCIA (RUÍNA): {ruin['ruin_p']:.2f}% (Risco Atuarial)")
    print(f"4. SUPPLY CHAIN (HT-GNN): {supply['risk_propagation']}")
    print(f"5. EXECUÇÃO (NASH): {game}")
    print("-" * 50)
    print(f"🎯 ALOCAÇÃO RECOMENDADA (KELLY): {kelly_size*100:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    try:
        asyncio.run(run_atlas())
    except KeyboardInterrupt:
        print("\nSessão encerrada pelo usuário.")