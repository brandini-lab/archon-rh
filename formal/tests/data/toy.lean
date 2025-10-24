import Mathlib

lemma toyLemma (n : Nat) : n + 0 = n := by
  simpa using Nat.add_zero n
