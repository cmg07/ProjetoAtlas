class GameTheoryISA:
    def lasry_lions_balance(self, order_size):
        # Equilíbrio de Nash para redução de impacto no mercado
        return "ORDEM PARCELADA (SLICE)" if order_size > 1000 else "ORDEM DIRETA"

class KellyCriterion:
    def sizing(self, win_rate, win_loss_ratio):
        # f* = (bp - q) / b
        f_star = (win_loss_ratio * win_rate - (1 - win_rate)) / win_loss_ratio
        return max(0, f_star * 0.5) # Half-Kelly para segurança