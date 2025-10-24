import Mathlib

namespace ArchonRiemann

def nymanBeurlingWeight (n : Nat) : Real :=
  1 / (Nat.succ n)

lemma nymanBeurlingWeight_pos (n : Nat) : 0 < nymanBeurlingWeight n := by
  have h : (0 : Real) < (Nat.succ n : Real) := by exact_mod_cast Nat.succ_pos n
  simpa [nymanBeurlingWeight] using one_div_pos.mpr h

end ArchonRiemann
