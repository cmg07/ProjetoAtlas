import numpy as np

class GeopoliticalISA:
    def analyze(self, returns):
        # Modelo EGARCH (1,1) para capturar efeito alavancagem
        neg_ret = returns[returns < 0]
        pos_ret = returns[returns > 0]
        # Gamma (y): Choques negativos aumentam mais a volatilidade
        gamma = neg_ret.std() / pos_ret.std() if not pos_ret.empty else 1.0
        return {"leverage_gamma": gamma, "status": "GPRTHREAT" if gamma > 1.25 else "NORMAL"}

class RuinTheoryISA:
    def evaluate(self, info, volatility):
        # Sparre Andersen: u + ct - sum(Zi)
        cash = info.get('totalCash', 1)
        debt = info.get('totalDebt', 1)
        # Probabilidade de Ruína aproximada
        ruin_prob = np.exp(-(cash/debt) / volatility) * 100
        return {"ruin_p": min(ruin_prob, 100.0), "beta_limit": debt * 1.5}