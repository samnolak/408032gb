```diff
--- a/intervals.py
+++ b/intervals.py
@@ -4,3 +4,11 @@
 def merge_intervals(intervals):
     """Merge overlapping or touching closed intervals [a, b]; return them sorted by start."""
-    raise NotImplementedError
+    out = []
+    for a, b in sorted(intervals):
+        if a > b:
+            raise ValueError("bad interval [%s, %s]" % (a, b))
+        if out and a <= out[-1][1]:
+            out[-1][1] = max(out[-1][1], b)
+        else:
+            out.append([a, b])
+    return out
```
