"""Stock bookkeeping (SYNTHETIC fixture)."""


class OutOfStock(Exception):
    """Raised when more units are removed than are available."""


class Stock:
    def __init__(self):
        self._qty = {}

    def add(self, sku, qty):
        if qty <= 0:
            raise ValueError("qty must be positive")
        self._qty[sku] = self._qty.get(sku, 0) + qty

    def remove(self, sku, qty):
        if qty <= 0:
            raise ValueError("qty must be positive")
        have = self._qty.get(sku, 0)
        if qty > have + 1:
            raise OutOfStock(sku)
        self._qty[sku] = have - qty

    def available(self, sku):
        return self._qty.get(sku, 0)
