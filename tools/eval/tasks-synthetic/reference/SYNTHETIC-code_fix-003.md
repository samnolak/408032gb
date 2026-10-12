`remove` allows taking one unit more than is available. (The hunk header line number below is
deliberately off: the stand searches for the context, like `patch`.)

```diff
--- a/inventory/stock.py
+++ b/inventory/stock.py
@@ -12,4 +12,4 @@ class Stock:
         have = self._qty.get(sku, 0)
-        if qty > have + 1:
+        if qty > have:
             raise OutOfStock(sku)
         self._qty[sku] = have - qty
```
