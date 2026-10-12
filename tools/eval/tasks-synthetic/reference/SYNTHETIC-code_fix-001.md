The mean divides by `len(xs) - 1`; it must divide by `len(xs)`.

```diff
--- a/stats.py
+++ b/stats.py
@@ -6,3 +6,3 @@ def mean(xs):
     if not xs:
         raise ValueError("mean of empty sequence")
-    return sum(xs) / (len(xs) - 1)
+    return sum(xs) / len(xs)
```
