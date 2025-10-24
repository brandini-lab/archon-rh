import Mathlib

namespace ArchonRiemann

def explicitKernel (x : Real) : Real :=
  Real.exp (-x) * Real.cos x

lemma explicitKernel_zero : explicitKernel 0 = 1 := by
  unfold explicitKernel
  simp

lemma explicitKernel_symm (x : Real) : explicitKernel (-x) = explicitKernel x := by
  unfold explicitKernel
  simp [Real.cos_neg, Real.exp_neg, mul_comm, mul_left_comm, mul_assoc]

end ArchonRiemann
